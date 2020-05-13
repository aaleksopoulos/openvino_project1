"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os, psutil
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from tracked_person import Tracked_Person
from math import pow, sqrt

import platform

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

DEBUG = True #helper variable

#NOTE Only applicable in the case of OPENVino version 2019R3 and lower
if (platform.system() == 'Windows'):
    CPU_EXTENSION = "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\\bin\intel64\Release\cpu_extension_avx2.dll"
elif (platform.system() == 'Darwin'): #MAC
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
else: #Linux, only the case of sse
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def memory_usage():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str, 
                        help="Path to image or video file, CAM if you want to use camera as input")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocess_frame(frame, width, height):
    '''
    Preprocess the image to fit the model
    '''
    frame = cv2.resize(frame, (width, height))
    frame = frame.transpose((2,0,1))
    return frame.reshape(1, 3, width, height)

def getDistance(x1, y1, centroid):
    x2, y2 = centroid
    dist = sqrt( pow((x2-x1),2) + pow((y1-y2),2) )
    if DEBUG:
        print("------ in distance --------")
        print("x1: ", x1)
        print("x2: ", x2)
        print("y1: ", y1)
        print("y2: ", y2)
        print("dist: ", dist)
        print("--------------------------")
    return dist

def get_results(in_frame, out_frame, counter, prob_threshold, widht, height, persons):
    timestamp = counter/10
    tmp_list = []
    for fr in out_frame:
        if (fr[0][0][0] == -1): #if we have not detected anything, we break out
            break
        if (fr[0][0][2]>prob_threshold) and (fr[0][0][0]==0): #if what we detected is indeed person and the probability is above the one stated
            print("---------------------------------------------------------time = ", timestamp)
            x1 = int(fr[0][0][3]*widht)
            y1 = int(fr[0][0][4]*height)
            x2 = int(fr[0][0][5]*widht)
            y2 = int(fr[0][0][6]*height)
            if DEBUG:
                print("--------------------------")
                print("calucalated x1: ", x1)
                print("calucalated x2: ", x2)
                print("calucalated y1: ", y1)
                print("calucalated y2: ", y2)
                print("--------------------------")
            if len(persons)==0:
                p = Tracked_Person(x1=x1, x2=x2, y1=y1, y2=y2)
                persons.append(p)
                if DEBUG:
                    print(p.toString())
                    print("person's centroid: ", p.getCentroid())
            else:
                for person in persons:
                    if person.isTracked():
                        dist = getDistance((x1+x2)/2, (y1+y2)/2, person.getCentroid())
                        if DEBUG:
                            print("dist: ",dist)
                        if dist<80: #number got from the current data
                            person.setX1(x1)
                            person.setX2(x2)
                            person.setY1(y1)
                            person.setY2(y2)
                            person.updateCentroid()
                            if DEBUG:
                                if (x1==person.getX1()) and (x2==person.getX2()) and (y1==person.getY1()) and (y2==person.getY2()):
                                    print("person updated succesfully")
                                else:
                                    print("values are not updated correcty")
                                    print("calucalated x1: ", x1)
                                    print("calucalated x2: ", x2)
                                    print("calucalated y1: ", y1)
                                    print("calucalated y2: ", y2)
                                    print("person x1: ", person.getX1())
                                    print("person x2: ", person.getX2())
                                    print("person y1: ", person.getY1())
                                    print("person y2: ", person.getY2())
                        else:
                            person.setTracked(False)
                            p = Tracked_Person(x1=x1, x2=x2, y1=y1, y2=y2)
                            tmp_list.append(p)
                            if DEBUG:
                                print(p.toString())
                                print("len of tmp_list: ", len(tmp_list))

            #print(fr[0][0])
            cv2.rectangle(in_frame, (x1, y1), (x2,y2), (0,255,255),1)
        if DEBUG:
            print("len of persons list: ", len(persons))
            for p in persons:
                if person.isTracked():
                    cv2.putText(img=in_frame, text=p.toString(), org=p.getCentroid(), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,255), thickness=1)

    if (len(tmp_list)!=0):
        for tmp_p in tmp_list:
            persons.append(tmp_p)
    
        if DEBUG:
            print("--- printing persons ----")
            for p in persons:
                print(p.toString())
            print("-------------------------")


    return in_frame

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    if DEBUG:
        print("probability threshold: ", prob_threshold)
        print("device: ", args.device)
        print("model_xml: ", args.model)        

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(device=args.device, model_xml=args.model, cpu_extension=None)
    ### TODO: Handle the input stream ###
    isImage = None #placeholder to check if we have an image of video input
    if (args.input).lower()=='cam':
        isImage = False
        args.input = 0
    elif (args.input).endswith('.jpg') or (args.input).endswith('bmp'):
        isImage = True #input is image
    else:
        isImage = False #we have a video stream as input
    
    if DEBUG:
        print("args.input: ", args.input)

    inp = cv2.VideoCapture(args.input)
    inp.open(args.input)

    #get the shape of the input
    width = int(inp.get(3))
    height = int(inp.get(4))
    if DEBUG:
        print("input image widht: ", width)
        print("input image height: ", height)

    #get the input shape of the networkd
    net_input_shape = infer_network.get_input_shape()
    if DEBUG:
        print("input_shape: ", net_input_shape)
        print("input_shape width: ", net_input_shape[2])
        print("input_shape height: ", net_input_shape[3] )

    if isImage:
        vid_capt = None
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_capt = cv2.VideoWriter('output_video.mp4', fourcc, 25, (width,height))


    request_id = 0
    counter = 0
    persons = []
    ### TODO: Loop until stream is over ###
    while inp.isOpened():
        
    ### TODO: Read from the video capture ###
        flag, frame = inp.read()
        if not flag:
            break #video ended
        
        #to cancel easily
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            break 
        counter +=1
    ### TODO: Pre-process the image as needed ###
        prep_frame = preprocess_frame(frame, net_input_shape[2], net_input_shape[3])
    ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(image=prep_frame,request_id=request_id)
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id=request_id)==0:
            output = infer_network.get_output(request_id=request_id)
            #if DEBUG:
                #print(output)
            ### TODO: Get the results of the inference request ###
            out_frame = get_results(frame, output, counter, prob_threshold, width, height, persons)
            ### TODO: Extract any desired stats from the results ###
        vid_capt.write(out_frame)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
    vid_capt.release()
    inp.release()
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    #to calculate execution time
    start_time = time.time()
    # Grab command line args
    args = build_argparser().parse_args()
    if DEBUG:
        print('args: ', args)
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

    mem = memory_usage()
        
    elapsed_time = time.time() - start_time
    print("total execution time in secs: ", elapsed_time)
    print("and memory used in Mb: ", mem)


if __name__ == '__main__':
    main()
