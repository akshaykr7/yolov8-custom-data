import tensorflow as tf
import numpy
import os
import xml.etree.ElementTree as ET

classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
         'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

# print(f"Total no. of classes: ")

ann_path = r"C:/Users/arock/OneDrive - IIT Kanpur\Recent study\ML1\Codes\VS Codes\Computer Vision\Basic tutorial\New folder\yolo_annotations"

def preprocess_xml(filename):
  tree = ET.parse(filename)
  root = tree.getroot()
  size_tree = root.find('size')
  height = float(size_tree.find('height').text)  # height of image
  width = float(size_tree.find('width').text)   # .text retrieves the string between <height> and </height>.
  bounding_boxes=[]
  for object_tree in root.findall('object'):
    for bounding_box in object_tree.iter('bndbox'):
      xmin = (float(bounding_box.find('xmin').text))
      ymin = (float(bounding_box.find('ymin').text))
      xmax = (float(bounding_box.find('xmax').text))
      ymax = (float(bounding_box.find('ymax').text))
      break                                         # break -  to take only 1st bb for one object, in case of more than 1 bb
    class_name = object_tree.find('name').text
    class_dict = {classes[i]:i for i in range(len(classes))}  # {'object_name': 0, ...}
    bounding_box = [
        (xmin+xmax)/(2*width), (ymin+ymax)/(2*height), (xmax-xmin)/width,   # normalizing BB values with width, height of image according to yolo algo.
        (ymax-ymin)/height, class_dict[class_name]]
    
    # print(file_txt)
    bounding_boxes.append(bounding_box)
  
  file_txt = os.path.join(ann_path, filename[-15:-3] + "txt")

  with open(file_txt, "w") as file:
      for bounding_box in bounding_boxes:
        bounding_box = [bounding_box[-1]] + bounding_box[:-1]
        # file.write(" ".join(map(str, bounding_box)) + "\n")
                
  return None

org_ann_path = r"C:\Users\arock\OneDrive - IIT Kanpur\Recent study\ML1\Codes\Jupyter Notebooks\Computer Vision\Courses\Free Code_Camp YT - DL with tf2\deep learning for computer vision\pascal-voc-2012\VOC2012\Annotations"


for file in os.listdir(org_ann_path):
  filename = os.path.join(org_ann_path, file)
  preprocess_xml(filename)

