# School-Waste-Reduction-System

## Abstract
Please note that this software is not made for personal use, but it is directed for organizations like school to reduce waste and save energy. It covers the field of AI (particularly computer vision) and IoT, whereas it aims to create a smart system to help ensure sustainable living.

It is suggested to watch the 1 min short video clip attached, to grab a better understanding of this project. 

The lighting system can be implemented on CCTV cameras of classrooms, and the smart system can automatically decide when to turn the lights ON or OFF, depending whether there are people in the room. 

While the Waste Classifier can be applied in the canteen, since it holds the majority and most variety of school waste. A special-machine equipped trash can with a camera device on top can be installed on a designated spot. And when students throw their trash into it, the AI system will automatically classify the trash, and sort it to the corresponding type of trash bag. For example, a soda can will be sorted into a metal-containing trash bag.

This software is used to monitor and observe the performance of the smart AI system, to be implemented at school.

## People Counter
The folder '/People Counter' contains the code for YOLOv8 Object Detection, which will draw a bounding box around a person in the webcam image.
For more information about YOLOv8, please refer to the following documentation from Ultralytics: https://docs.ultralytics.com/tasks/detection/

## Waste Classifier
The folder '/Waste Classifier' contains the teachable machine file ('AI Waste Classifier.tm'), which already contain the dataset for training. This file can be viewed by opening it from this website: https://teachablemachine.withgoogle.com/train/image
The dataset has been taken from Trashnet: https://github.com/garythung/trashnet

NOTE: This model is specifically trained to resemble a trash can interior environment, so one thing to note: please change the background images data to your particular use case. Here, I have chosen the color marble green, because black/silver has a color similar to metal, and white similar to paper, so it may affect the prediction results.

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
