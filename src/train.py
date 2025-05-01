# train.py
import argparse
from ultralytics import YOLO

def train_model(data_config, weights, epochs=30, batch=16, imgsz=640, project="results/train", name=None):
    """
    Train a YOLOv8 model.
    """
    model = YOLO(weights)
    model.train(data=data_config, epochs=epochs, batch=batch, imgsz=imgsz, project=project, name=name)
    print(f"Training completed. Weights and logs are saved in {project}/{name or 'exp'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset config (data.yaml)")
    parser.add_argument("--weights", type=str, default="yolov8m.pt", help="Initial weights or model path")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--project", type=str, default="results/train", help="Save directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (subfolder in project)")
    args = parser.parse_args()

    train_model(
        data_config=args.data,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name
    )
