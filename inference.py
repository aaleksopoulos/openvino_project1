#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

DEBUG = False

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.core = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None

    def get_unsupported_layers(self, device):
        #get a list of the supported layers
        supported_layers = self.core.query_network(self.network, device_name=device)
        #get the required layers
        required_layers = list(self.network.layers.keys())
        #check if there are unsupported layers
        unsupported_layers = []
        for layer in required_layers:
            if layer not in supported_layers:
                unsupported_layers.append(layer)

        return unsupported_layers

    def load_model(self, device, model_xml, cpu_extension):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        #check if we have provided a valid file for the xml
        if (model_xml.endswith('.xml')) and (os.path.exists(model_xml)):
            model_bin = model_xml.replace('.xml', '.bin') #get the model bin
            if DEBUG:
                print("model found")
                print("model_xml: ", model_xml)
                print("model_bin: ", model_bin)
                print("device: ", device)
        else:
            print("There was a problem reading the xml file provided, exiting...")
            exit(1)

        #initialize the Inference Engine (NOTE using the 2019R3 version of OPENVino, as stated in the exercise)
        self.core = IECore()
  
        #initialize the Network (NOTE in the 2020R1 version of OPENVino, can be used from the IECore)
        self.network = IENetwork(model=model_xml, weights=model_bin)

        #check if there are any unsupported layers
        unsupported_layers = self.get_unsupported_layers(device)

        #if there are any unsupported layers, add CPU extension, if avaiable
        if (len(unsupported_layers)>0) and (device=='CPU'):
            print("There are unsupported layers found, will try to add CPU extension...")
            self.core.add_extension(extension_path=cpu_extension, device=device)

        #add, if provided, a cpu extension
        if (cpu_extension):
            self.core.add_extension(cpu_extension)

        #recheck for unsupported layers, and exit if there are any
        unsupported_layers = self.get_unsupported_layers(device)
        if (len(unsupported_layers)>0):
            print("After adding CPU extension, there are still unsupported layers, exiting...")
            exit(1)
        
        #load to network to get the executable network
        self.exec_network = self.core.load_network(self.network, device)

        #get the input and output blobs
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.exec_network

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image, request_id=0):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        #requiest_id is set to zero, if not specified, for pseudo async mode
        self.exec_network.start_async(request_id=request_id, inputs={self.input_blob:image})
        return

    def wait(self, request_id=0):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        infer_status = self.exec_network.requests[request_id].wait(-1)
        return infer_status

    def get_output(self, request_id=0):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        result = self.exec_network.requests[request_id].outputs[self.output_blob]
        #print('---------------------------------------- latency:', self.exec_network.requests[request_id].latency)
        return result