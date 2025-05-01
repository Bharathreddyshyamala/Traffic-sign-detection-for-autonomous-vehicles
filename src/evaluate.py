# evaluate.py
import argparse
from ultralytics import YOLO

def evaluate_model(weights, data_config, imgsz=640, batch=16, project="results/eval", name=None, plots=True):
    """
    Evaluate a YOLOv8 model on validation data.
    """
    model = YOLO(weights)
    results = model.val(data=data_config, imgsz=imgsz, batch=batch, save_json=True,
                        project=project, name=name, plots=plots)
    print("Evaluation results (mAP50-95):", results.box.map)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset config (data.yaml)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for evaluation")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--project", type=str, default="results/eval", help="Directory to save results")
    parser.add_argument("--name", type=str, default=None, help="Name of this evaluation run")
    parser.add_argument("--no-plots", action="store_true", help="Disable saving confusion matrix and PR curve plots")
    args = parser.parse_args()

    evaluate_model(
        weights=args.weights,
        data_config=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        plots=not args.no_plots
    )
