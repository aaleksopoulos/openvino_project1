import numpy as np
import os
import sys

import cv2
import time

def t_f(path, inp):
    import tensorflow as tf

inp = cv2.VideoCapture('resources/Pedestrian_Detect_2_1_1.mp4')

paths = []
paths.append('uncoverted_models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')
paths.append('uncoverted_models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb')
paths.append('uncoverted_models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb')
paths.append('uncoverted_models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb')

fw = open("no_openvino_inference_time.txt", 'a')
#fw.write("=================================================================\n")
#fw.write("================ inference time witouh OpenVINO =================\n")
#fw.write("=================================================================\n")
#for sure there is a better way that running one model at a time
#but i did not figure out how to do it
#stupid, but it works

path = paths[3]
model_list = ((path.split('/'))[1]).split('_')

model = '_'.join(model_list[:4])
print('-------------------------------model:', model)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

start_inference = time.time()
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while inp.isOpened():
            flag, image_np = inp.read()
            if not flag:
                break #video ended
            
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

total_time = time.time() - start_inference

fw.write("|model: \t\t\t|" + model + '\t\t|\n')
fw.write("|inference time (secs): \t|" + '{:07.3f}'.format(total_time) + '\t\t\t|\n')
fw.write("=================================================================\n") 
 
fw.close()