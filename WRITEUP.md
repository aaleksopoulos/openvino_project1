# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

As an introduction, this project is based on a repo provided to us by [Udacity](https://github.com/udacity/nd131-openvino-fundamentals-project-starter), in order to implement
it as a Person Counter.

## Basic Info About the Aprroach

The main model consists mainly on 3 files

1. tracked_persons.py 
It contains the implementation of Tracked_Person class, that is used to store the main attributes of each person that was tracked by the algorith

2. inference&#46;py
it contains the implementation of Network class, used to initialize the required classes perform inference on the model

3. main&#46;py
it contains the code to load the model, image, perform inference, send the requested data to the servers, extract some metrics and saves a videofile of the result, as well as a txt with the metrics.

#### Some things to consider

There are some things to consider about the implemented approach. At the first time a person is tracked, is stored in a list, represented by its centroid. Next, in the next frame, first we update the already known persons to their new location. To achieve that, we calculate the distances between the centroid of all currently known boxes and the new ones. Then, using the smallest distance, and if it is lower than a predefined value, we update the person's location. All other objects are tracked as new persons.

We also considered the case that a known person may be present in the current frame, but the model did not detect it. In order for this person not to be considered as a new person in the next frame it is detected, we store each person for 15 seconds, before actually deleting it from the person's list in order to save memory. It must be stated that this person is not used to the actual count of the persons in frame, which is achieved using the the tracked attribute of the class.



## Explaining Custom Layers

The main idea of OpenVINO toolkit is to be able to run in edge, which consists of low-end (in terms of CPU power and memory size) devices, fast and without any  loss in accuracy, or at least not any significant loss. In order to achieve that, the toolkit provides the Model Optimizer, a python script to convert the model we want to use in an intermediate representaion, which contains improvements to model size and speed.

The field of computer vision, and AI in general, is expanding rapidly. Many new models or techniques rise up everyday, and sometimes, in academia or in industry, which may add functionality or improve previous models. As we have seen, converting those models though model optimizer in the required intermediate representation is specific for each model and device we are going to use, and even after adding (at least in versions of OpenVINO prior to 2019R3) device plugins, some of the model layers may not be supported by the model optimizer. In this case, one solution is to use custom layers, . Although, it must be stated that in general, is not that common to use such techniques, and is applied in the case of the layer is not in the [supported layers list](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html).

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
