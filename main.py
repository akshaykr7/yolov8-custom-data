
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a new model - nano version

results = model.train(data='config.yaml', epochs=5)