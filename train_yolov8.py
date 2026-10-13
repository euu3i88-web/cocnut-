"""
Train a YOLOv8 detector on the supplied coconut maturity dataset.
Run locally after: pip install -r requirements.txt
Usage: python train_yolov8.py --data training_data/data.yaml --epochs 30
This script starts from pretrained yolov8n.pt weights and fine-tunes the detector.
"""
import argparse, os, shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", default="training_data/data.yaml")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args=p.parse_args()
    data_path=Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    data=yaml.safe_load(data_path.read_text())
    root=data_path.parent
    # Rewrite image directories to absolute paths, independent of current working directory.
    fixed={}
    for split in ("train","val","test"):
        if split in data:
            candidate=Path(data[split])
            fixed[split]=str((root/candidate).resolve() if not candidate.is_absolute() else candidate)
    fixed["nc"]=data["nc"]; fixed["names"]=data["names"]
    fixed_yaml=root/"data_training.yaml"
    fixed_yaml.write_text(yaml.safe_dump(fixed, sort_keys=False))
    model=YOLO("yolov8n.pt")
    model.train(data=str(fixed_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=2, patience=8,
                project="runs/coconut", name="yolov8n", exist_ok=True)
    best=Path("runs/coconut/yolov8n/weights/best.pt")
    if not best.exists():
        raise FileNotFoundError(f"Training completed but best weights not found at {best}")
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(best, "models/best.pt")
    print(f"Copied trained model to {Path('models/best.pt').resolve()}")

if __name__=="__main__":
    main()
