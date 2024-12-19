import tensorflow as tf
import numpy
import xml.etree.ElementTree as ET

classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
         'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

print(f"Total no. of classes")

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
    bounding_boxes.append(bounding_box)
  return tf.convert_to_tensor(bounding_boxes)


bb = preprocess_xml("D:/Lab Work/ML_Projects/ML_Data/Object Detection Datasets/VOC2012/Annotations/2007_000027.xml")
print(bb)