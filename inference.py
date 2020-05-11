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

    def load_model(self, device='CPU', model_xml, cpu_extension=None):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        #check if we have provided a valid file for the xml
        if (model_xml.endsWith('.xml') and (os.path.exists(model_xml)):
            model_bin = model_xml.replace('.xlm', '.bin') #get the model bin
        else:
            print("There was a problem reading the xml file provided, exiting...")
            exit(1)

        #initialize the Inference Engine (NOTE using the 2019R3 version of OPENVino, as stated in the exercise)
        self.core = IECore()

        #add, if provided, a cpu extension
        if (cpu_extension):
            self.core.add_extension(cpu_extension)
        
        #initialize the Network (NOTE in the 2020R1 version of OPENVino, can be used from the IECore)
        self.network = IENetwork(model=model_xml, weights=model_bin)

        #load to network to get the executable network
        self.exec_network = self.core.load_network(self.network, device)

        #get the input and output blobs
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        
        return self.network.inputs[self.input_blob].shape

    def exec_net(self):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return
