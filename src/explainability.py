# explainability.py
import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

class MaskGenerator:
    """
    Generates N randomized RISE masks of size H×W, following D-RISE:
    1. Sample h×w binary masks with p(1)=p
    2. Upsample to (h+1)*CH × (w+1)*CW via bilinear interpolation
    3. Randomly crop H×W from them
    """
    def __init__(self, H, W, h, w, p, N):
        self.H, self.W = H, W
        self.h, self.w = h, w
        self.p, self.N = p, N
        # cell size
        self.CH = H // h
        self.CW = W // w

    def generate(self) -> np.ndarray:
        # 1) sample small binary masks
        small_masks = np.random.binomial(1, self.p, size=(self.N, self.h, self.w)).astype(np.float32)
        # 2) upsample via resizing
        up_h = (self.h + 1) * self.CH
        up_w = (self.w + 1) * self.CW
        masks = np.zeros((self.N, up_h, up_w), dtype=np.float32)
        for i in range(self.N):
            masks[i] = cv2.resize(small_masks[i], (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        # 3) random crop H×W
        out = np.zeros((self.N, self.H, self.W), dtype=np.float32)
        for i in range(self.N):
            y = np.random.randint(0, up_h - self.H + 1)
            x = np.random.randint(0, up_w - self.W + 1)
            out[i] = masks[i, y:y+self.H, x:x+self.W]
        return out

def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Compute Intersection over Union for two bounding boxes [x1,y1,x2,y2].
    """
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union > 0 else 0.0

def cosine_sim(p: np.ndarray, q: np.ndarray) -> float:
    """
    Cosine similarity between two probability vectors.
    """
    num = np.dot(p, q)
    den = np.linalg.norm(p) * np.linalg.norm(q)
    return float(num/den) if den > 0 else 0.0

class DRISEExplainer:
    def __init__(self, model: YOLO, mask_gen: MaskGenerator, device: str='cuda'):
        self.model = model
        self.mask_gen = mask_gen
        self.device = device
        self.model.model.to(device)

    def _run_detector(self, img: np.ndarray):
        """
        Run YOLO detector and return list of detections:
        [(box, object_conf, class_probs), ...].
        """
        results = self.model.predict(img, device=self.device, conf=0.01)[0]
        dets = []
        C = len(self.model.names)
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            # one-hot probability vector
            P = np.zeros(C, dtype=np.float32)
            P[cls] = conf
            dets.append((np.array([x1, y1, x2, y2]), conf, P))
        return dets

    def explain(self, image: np.ndarray):
        """
        image: H×W×3 uint8 RGB image.
        returns: S of shape (T, H, W) saliency maps, one per detection.
        """
        H, W, _ = image.shape
        # 1) original detections
        Dt = self._run_detector(image)
        T = len(Dt)
        if T == 0:
            raise ValueError("No objects detected in the image.")
        # 2) generate masks
        masks = self.mask_gen.generate()  # (N, H, W)
        N = masks.shape[0]
        # 3) run detector on masked images
        Wit = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            masked = (image.astype(np.float32)/255.0) * masks[i][..., None]
            masked = (masked * 255).astype(np.uint8)
            Dp = self._run_detector(masked)
            # 4) compute weights per detection t
            for t in range(T):
                Lt, Ot, Pt = Dt[t]
                best = 0.0
                for Lp, Op, Pp in Dp:
                    sim = iou(Lt, Lp) * cosine_sim(Pt, Pp)
                    if sim > best:
                        best = sim
                Wit[i, t] = best
        # 5) aggregate saliency maps
        S = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            S[t] = (Wit[:, t][:, None, None] * masks).sum(axis=0) / N
        return S, Dt

def overlay_and_save(rgb, saliency_map, box, class_name, conf, save_path):
    """
    Overlay saliency map on image and save the result.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(rgb)
    plt.imshow(saliency_map, cmap='jet', alpha=0.5)
    x1, y1, x2, y2 = box.astype(int)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, edgecolor='white', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title(f"{class_name} ({conf:.2f})")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D-RISE explanation for object detections")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLOv8 weights (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (RGB)")
    parser.add_argument("--h", type=int, default=16, help="Mask grid height")
    parser.add_argument("--w", type=int, default=16, help="Mask grid width")
    parser.add_argument("--p", type=float, default=0.5, help="Probability for mask generation")
    parser.add_argument("--N", type=int, default=500, help="Number of masks to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    parser.add_argument("--out_dir", type=str, default="results/explain", help="Directory to save saliency maps")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.weights)
    # Read and prepare image
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Image {args.image} not found")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape

    # Create mask generator
    mask_gen = MaskGenerator(H, W, h=args.h, w=args.w, p=args.p, N=args.N)
    explainer = DRISEExplainer(model, mask_gen, device=args.device)

    # Explain
    saliency_maps, detections = explainer.explain(rgb)

    # Prepare output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Overlay and save saliency maps for each detection
    for idx, ((box, conf, P), S) in enumerate(zip(detections, saliency_maps)):
        class_id = int(np.argmax(P))
        class_name = model.names[class_id]
        save_path = os.path.join(args.out_dir, f"saliency_{idx}_{class_name}.png")
        overlay_and_save(rgb, S, box, class_name, conf, save_path)
        print(f"Saved explanation for object {idx} to {save_path}")
