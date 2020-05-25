# Project Write-Up

 

You can use this document as a template for providing your project write-up. However, if you have a different format you prefer, feel free to use it as long as you answer all required questions.
 

As an introduction, this project is based on a repo provided to us by [Udacity](https://github.com/udacity/nd131-openvino-fundamentals-project-starter), in order to implement it as a Person Counter.

 

## Basic Info About the Aprroach

 

The project consists mainly on 3 files

 

1. tracked_persons&#46;py

It contains the implementation of Tracked_Person class, that is used to store the main attributes of each person that was tracked by the algorith

 

2. inference&#46;py

it contains the implementation of Network class, used to initialize the required classes perform inference on the model

 

3. main&#46;py

it contains the code to load the model, load the image, perform inference, send the requested data to the servers, extract some metrics and saves a videofile of the result, as well as a txt with the metrics.

 

#### Some things to consider

 

There are some things to consider about the implemented approach. At the first time a person is tracked, is stored in a list, represented by its centroid. Next, in the next frame, first we update the already known persons to their new location. To achieve that, we calculate the distances between the centroid of all currently known boxes and the new ones. Then, using the smallest distance, and if it is lower than a predefined value, we update the person's location. All other objects are tracked as new persons.

 

We also considered the case that a known person may be present in the current frame, but the model did not detect it. In order for this person not to be considered as a new person in the next frame it is detected, we store each person for 15 seconds, before actually deleting it from the person's list in order to save memory. It must be stated that this person is not used to the actual count of the persons in frame, which is achieved using the the tracked attribute of the class.

 

Another thing to consider is that the file run_info.txt contains the commannds that were used to convert each model with the model optimizer, and also to run the program. In all cases the probability threshold was 0.6, and the converted to Intermedite Representation models were placed in a folder named "models" (it is not uploaded due to limitation on file size). The folder 'final_files' contains the output of the programm, namely a video of the output of each model, a txt file that contains some metrics and the inference time of the models, if we do not use the OpenVINO toolkit.

 

#### Addtional Features

 

At this approach we added some extra functionality. If a specific person is tracked for over 10 seconds, then the bounding box of that person turns into red, otherwhise it is yellow. Additionally, if there are more than 10 persons tracked at a single frame, a warning is displayed at the top left corner of the image, showing the number of persons that were detected.

 

## Explaining Custom Layers

 

The main idea of OpenVINO toolkit is to be able to run in edge, which consists of low-end (in terms of CPU power and memory size) devices, fast and without any  loss in accuracy, or at least not any significant loss. In order to achieve that, the toolkit provides the Model Optimizer, a python script to convert the model we want to use in an intermediate representaion, which contains improvements to model size and speed.

 

The field of computer vision, and AI in general, is expanding rapidly. Many new models or techniques rise up everyday, in academia or in industry, which may add functionality or improve previous models. As we have seen, converting those models though model optimizer in the required intermediate representation is specific for each model and device we are going to use, and even after adding (at least in versions of OpenVINO prior to 2019R3) device plugins, some of the model layers may not be supported by the model optimizer. In this case, one solution is to use custom layers. Although, it must be stated that in general, is not that common to use such techniques, and is applied in the case of the layer is not in the [supported layers list](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html). In this case, that the network topology we use contains layers that are not present in that list, the model optimizer classifies them as custom. How exactly these layers are addresed, it depends on the type of the model that we are using. For example, TensorFlow models provide us with three possibilities, either to register them as extensions to the model optimizer, to replace them with a different but equivalent subgraph, or to offload the computation to TensorFlow during inference.

 

## Comparing Model Performance

 In this project, the following four models were chosen from the tensorflow model zoo.

1. [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
2. [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
3. [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
4. [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)

Before continuing on project performance, let's firts provide the code on how to convert those models in Intermediate Representation format, which is needed to run the analysis with the OpenVINO toolkit. In all the cases, the first step is to untar the model and enter the folder. The procedure will be presented for each model seperately, and we will suppose that we already have a bash window open in the folder that we downloaded them. Also, it must be noted that for the output folder was used a path in order to include those intermediate represenations to the project folder, which is specific for my system.

1. ssd_mobilenet_v1_coco

```
tar -xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz 
cd ssd_mobilenet_v1_coco_2018_01_28/
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json -o /home/atalex/udacity_openvino_exercises/openvino_project1/models/
```

2. ssd_mobilenet_v2_coco

```
tar -xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz 
cd ssd_mobilenet_v2_coco_2018_03_29/
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json -o /home/atalex/udacity_openvino_exercises/openvino_project1/models/
```

3. ssdlite_mobilenet_v2_coco

```
tar -xvzf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
cd ssdlite_mobilenet_v2_coco_2018_05_09/
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json -o /home/atalex/udacity_openvino_exercises/openvino_project1/models/
```

4. ssd_inception_v2_coco

```
tar -xvzf ssd_inception_v2_coco_2018_01_28.tar.gz 
cd ssd_inception_v2_coco_2018_01_28/
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json -o /home/atalex/udacity_openvino_exercises/openvino_project1/models/
```

It would also be useful to show how to run each model, the commands are shown below. Before running the acutal models some initializations should take place.
First, we should start the required servers, as per instructions. For each of these commands one BASH window should be open at the project direcotry
```
cd webservice/server/node-server
node ./server.js
```

```
cd webservice/ui
npm run dev
```

```
sudo ffserver -f ./ffmpeg/server.conf
```

If we have not added OpenVINO to PATH, we should initialize it first

```
source /opt/intel/openvino/bin/setupvars.sh
```
Then, we need to run the actual model. For each one, the commands are shown below

1. ssd_mobilenet_v1_coco

```
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilenet_v1_coco.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

2. ssd_mobilenet_v2_coco

```
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilenet_v2_coco.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

3. ssdlite_mobilenet_v2_coco

```
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssdlite_mobilenet_v2_coco.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

4. ssd_inception_v2_coco

```
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_inception_v2_coco.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

If we want to run all the models with one command, we could use the following

```
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilenet_v1_coco.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm; python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilenet_v2_coco.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm; python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssdlite_mobilenet_v2_coco.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm; python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_inception_v2_coco.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm; echo "all done"
```

Now, regarding model performance. At first we will compare the use of those models, in respect of the performance with and without the use of OpenVINO. Two metrics were user, model size and the time it took for each model to complete inference in the video it was provided to us. The results are shown in the tables that follow

- Model size

|  MB       | ssd_mobilenet_v1_coco | ssd_mobilenet_v2_coco | ssdlite_mobilenet_v2_coco | ssd_inception_v2_coco |
| :----: |         :----:        |         :----:        |           :----:          |         :----:        |
| with OpenVINO |        27.5        |        67.6        |          18.3          |         100.5        |
| without OpenVINO |        29.1       |        69.7        |          19.9          |         102.0        |


- Infernce time

|  sec       | ssd_mobilenet_v1_coco | ssd_mobilenet_v2_coco | ssdlite_mobilenet_v2_coco | ssd_inception_v2_coco |
| :----: |         :----:        |         :----:        |           :----:          |         :----:        |
| with OpenVINO |        50.2       |        64.0       |          26.9         |         106.0       |
| without OpenVINO |        107       |       144.6        |          137.7          |        207.2        |


Finally, it would be worthy to make another comparison, between those models. The information regarding the memory need for the execution of each model were taken using the psutil module of Python. Also, using the getSizeOf command of the sys module, we calculated tha data that was send to ffmpeg server. The results were shown in the table below. 

|         | ssd_mobilenet_v1_coco | ssd_mobilenet_v2_coco | ssdlite_mobilenet_v2_coco | ssd_inception_v2_coco |
| :----: |         :----:        |         :----:        |           :----:          |         :----:        |
| execution time (sec) |        153.4       |       167.2       |          127.9        |         203.4       |
| memory used (Mb) |        161.2      |        125.8       |          163.9        |         145.0        |
| data sent to ffserver (Mb)|        1323.4       |        1323.4       |          1323.4         |         1323.4         |

#### Limitations of Each Model

As a general rule, all models worked well and were able to identify the persons, even with the default setting of 0.6 as a probability threshold. One difference, as it can easily be stated from the table above is that the ssd_mobilenet_v2_coco in comparison with the ssd_mobilenet_v1_coco performed better in terms of memory requirements, but it was a bit slower. 

If we compare the v2 models, we can see that the the faster one was the the ssdlite model, although it had the highest memory requirements. The inception on had the slower execution time, while the mobilenet had the lowerst memory requirements, and a good balance in terms of execution time.

One other thing that was important is that, although as it was stated, all the persons were tracked, the bounding boxes were not the same. So, attention had to be payed in the tracking algorithm. For example, when the 3rd person walked away of the stand, the bounding box of the ssdlite model was quite bigger than the other models, so the centroid distance was bigger and the algorithm counted that as an extra person. So some callibration had to be made.


## Assess Model Use Cases

 Such an app could be suitable in many cases. First of all, in stores it could be usefull since the owner of the managers can be aware not only on the ammount of customers that they have at a certain time, but also if - for any case - many customers prefer certain places of their store. So they can make arrangements on their placing of their products, or even arrange product ordering in case a certain area seems to attract more people so they are not run out. Simirarly, they can arrange to place their staff to accomodate the customers.

 

Simiral use case could be considered to airports, or in general any place that attrackts a large number of people. It could also be ideal for museums, if certain exhibits attrack more people, they could be placed seperetaly so overcrowding could be avoided and help the security of both the exhibits and the persons that are admiring them.

 

Another use case could be in security. It could detect if there are person in a place, where there should not be. As an example could be a person in a store, during the time that the store is closed. Also, mainly in construction places, it could detect if there are persons in restricted or hazardous areas and notify the people in charge.

 

Regarding security, another use case, came to me because of my current work as a Traffic Control Center Operator. In tunnels, and especially in the case of fire, there are places that are designed to provide a shelter to the ones that are trapped inside the tunnel. Such an app could detect not only which of these safe places have people inside them, but also how many people, and such an information could be critical to the rescue teams, in order to be able to organize their operations optimal.

 

## Assess Effects on End User Needs

 

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. Let's check each one of these on its own

 

-Lighting: Either poor lighting, or too much light could cause the image to saturate, so the borders between the objects contained in the image are not distinct. This could cause poor perfomance of the algorithm and objects might not be identified. In order to overcome this problem, we could enhance the pictures using proper software and/or hardware, or we might combine the algorithm with other techiques, such as cannyEdgeDetection.

 

-Model accuracy is a main parameter on the effectiveness of the algorithm. Having a low accuracy could cause a lot of misclassification, namely lots of Fale Positives or True Negatives, which, in the end, diminishes the model's functionality. On the other hand, having an extremely high accuracy may be a signal of overfit, which is the case that the algorithm 'knows' extremely well the training set and performs exceptionally well with it, but in cases outside that may have poor performance.

 

-Focal leght/image size could cause disortions in the image, and the effects could be similar to the case of lighting. Again the algorithm could fail to properly detect the objects, reducing the functionality of the program. Again one solution could be use of techiques to clear up the image, although the proper techique would be to replace the hardware (camera) with a more appropriate one.

 

## Future Work

 

Of course there are several things that could be improved at this application. As was stated above, each person is tracked by its centroid. Althouhgh this approach makes it easier to handle the person, it has the drawback that is not always easy, or safe, to track a person. If the person is near the camera and moves a bit, then the distance calculated will probably be higher if the same person made the same movemented and was far in the background, since its bounding box is bigger in the former case. So, if we set the distance of the object between two frames to be stable, it could lead in errors, either if the distance is too big not to be able to track correctly the people in the back, or if the distance is to small the persons that are close to the camera may be qualified as two seperate persons. This was actual the case during the implementation of the algorithm, in at the third person, while it moved away from the stand, the bounding box was a bit higher than usual and it counted her as a different person during leaving, which was corrected by adjusting the value of the distance limit. Such behavior could be avoided using a different metric, namely the [Intersection Over Union - IoU](https://en.wikipedia.org/wiki/Jaccard_index). This metric actually is the ratio of the area of overlap to the area of union, and since it is dependant only on the size of the bounding boxes, can help overcome that problem. It should be noted that the way the Tracked_Person class is implemented, using the 4 corners of the bounding box, is rather easy to implement such a metric.

 

Also, another feature that could be added is to combine that technique with other computer vision techniques. An idea is to use it in combination with segmentation, so that we could know exactly where each person stands on, or with human pose estimation model, which could identify if a person is at risk.