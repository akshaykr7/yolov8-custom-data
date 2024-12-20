
# YOLO v8 Custom data

[Construction Safety Equipments Dataset]

([Dataset Link](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30))

**Create virtual environment**

conda create yolov8_custom
conda activate yolov8_custom

**To check GPU**

import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.current_device())  # Should return the current GPU device index
print(torch.cuda.get_device_name(0))  # Should return the GPU name
