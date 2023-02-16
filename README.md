# School-Waste-Reduction-System

## Overview
This software is a monitoring system designed to reduce waste and save energy in a school. The code imports various libraries for image processing, including OpenCV, NumPy, and PIL. It also imports a pre-trained model for waste classification using Keras and a YOLO object detection model for counting the number of people in a classroom.

## People Counter
The folder '/People Counter' contains the code for YOLOv8 Object Detection, which will draw a bounding box around a person in the webcam image.
For more information about YOLOv8, please refer to the following documentation from Ultralytics: https://docs.ultralytics.com/tasks/detection/

## Waste Classifier
The folder '/Waste Classifier' contains the teachable machine file ('AI Waste Classifier.tm'), which already contain the dataset for training. This file can be viewed by opening it from this website: https://teachablemachine.withgoogle.com/train/image
The dataset has been taken from Trashnet: https://github.com/garythung/trashnet

'/Evaluation Metrics' contains the Metrics evaluated from testing data. 
'image.py' will classify the images from '/testing_images'. While 'webcam.py' will classify images directly from webcam stream. 

## GUI
### Set Up
Before running 'GUI/main.py', be sure to run 'pip install -r requirements.txt', to install all the neccessary packages/dependencies needed by the program. 

The GUI folder has '/images' and '/models', which are the assets needed to run the program. 

### General Idea
The program has two main functions: display_page1() and display_page2(). The first function is responsible for displaying a window that shows the webcam feed and uses the Keras model to classify the type of waste in the image. The second function displays a window that shows the webcam feed and uses the YOLO object detection model to count the number of people in a classroom, and decide whether to turn the light ON or OFF.

### Waste Classifier Function
In the first function, the program reads from the webcam and resizes the captured image to 224x224, which is the input shape for the Keras model. It then normalizes the image and makes a prediction using the pre-trained Keras model. If the model detects an object in the image (i.e., not a "Background"), the predicted label is displayed in a label widget in the window. The program updates the webcam image in the window every 20 milliseconds using the display_cam() function.

### Lighting System Function
In the second function, the program reads from the webcam and uses the YOLO object detection model to detect people in the image. It then counts the number of people in the image and displays the count in a label widget in the window. The program updates the webcam image in the window using the display_cam() function, and if it detects at least one person in the image, it toggles the classroom lights ON at a fixed time interval.
