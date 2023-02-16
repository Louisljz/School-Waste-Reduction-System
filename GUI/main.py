# Imports
# Paths
import sys, os
# GUI
from tkinter import *
from PIL import Image, ImageTk
# Computer Vision
import cv2
import numpy as np
# Load Keras Model
from keras.models import load_model
# YOLO object detection
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Check whether file is run as exe or python script
if getattr(sys, 'frozen', False):
    folder_path = os.path.dirname(sys.executable)
else:
    folder_path = os.path.dirname(__file__)

# Set Up Folder Paths
image_path = os.path.join(folder_path, 'images/')
model_path = os.path.join(folder_path, 'models/')

# Initialize and Configure Tkinter Window
window = Tk()
window.geometry("1152x700")
window.configure(bg = "#FFFFFF")
window.title('School Waste Reduction System')
window.resizable(False, False)
icon_img = PhotoImage(file=os.path.join(image_path, 'icon.png'))
window.iconphoto(True, icon_img)

# Initialize Webcam, YOLO and Keras Model with labels
cap = cv2.VideoCapture(0)
yolo = YOLO(os.path.join(model_path, 'yolov8n.pt'))
model = load_model(os.path.join(model_path, "waste_classifier.h5"), compile=False)
class_names = open(os.path.join(model_path, "labels.txt"), "r").readlines()

# Display Waste Classifier Window
def display_page1():
    # Camera and AI Function
    def display_cam():
        # Read Camera and Resize Frame
        _, frame = cap.read()
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

        # Reshape image to the model's input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image
        image = (image / 127.5) - 1

        # Model Prediction
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction) # Get Label Index
        class_name = class_names[index] # Get Label String
        name = class_name[2:].strip() # Remove \n
        # Display Prediction Only If there is Object
        if name != 'Background': 
            message.set("AI classifies this Trash as: ")
            label.set(name)
        else:
            message.set('')
            label.set('')
        
        # Set Webcam Image in Tkinter Label
        cv2image = cv2.resize(frame, (540, 380))
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image = img)
        cam.imgtk = imgtk
        cam.configure(image=imgtk)
        # Updates Frame
        cam.after(20, display_cam)
    
    # Creates New Window
    page1 = Toplevel()
    page1.geometry('1152x700')
    page1.title('Canteen Waste Classifier')

    # Display Background
    bgimg = PhotoImage(file = os.path.join(image_path, "bg2.png"))
    bg = Label(page1, image = bgimg)
    bg.place(x = 0, y = 0)

    # Message and Prediction Display
    message = StringVar()
    message_label = Label(page1, textvariable=message, font=("Arial", 25), 
                            background='green', foreground='white')
    message_label.place(x=700, y=300)

    label = StringVar()
    class_label = Label(page1, textvariable=label, font=("Arial", 25), 
                            background='blue', foreground='white')
    class_label.place(x=770, y=350)

    # Webcam Display
    cam = Label(page1)
    cam.place(x = 100, y = 180)

    display_cam()
    page1.mainloop()

# Display Classroom Lighting Window
def display_page2():
    # Make variable global, to prevent UnboundLocalError
    global duration
    duration = 0 # For time interval between Lights ON/OFF 
    # Camera and AI Function
    def display_cam():
        global duration
        counter = 0 # Counts the number of people in every frame
        _, img = cap.read()
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # YOLO Prediction
        results = yolo.predict(img)

        for r in results:
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0] # Get Bbox Coordinates
                cls = yolo.names[int(box.cls)] # Get Label String
                conf = box.conf[0] # Get confidence score
                if cls == 'person' and conf > 0.5:
                    counter += 1 # Increments number of person
                    # Drawing bounding box
                    annotator.box_label(b, color=(0, 255, 0)) 
        
        frame = annotator.result() # Return annotated image
        # Set Webcam Image in Tkinter Label
        cv2image = cv2.resize(frame, (540, 380))
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image = img)
        cam.imgtk = imgtk
        cam.configure(image=imgtk)
        
        people_counter.set(f"Number of People in Classroom: {counter}")
        # If there is at least one person in classroom
        if counter >= 1:
            # Increments duration of the person staying in the room
            if duration < 30:
                duration += 1
        else: # If there is no person in classroom
            if duration > 0:
                # Decrements duration of the person leaving the room
                duration -= 1
        # Wait for some time, then turn on lights
        if duration == 30:
            light_status.set('Lights: ON')
        # Wait for some time, then turn off lights
        elif duration == 0:
            light_status.set('Lights: OFF')
        
        cam.after(20, display_cam) # Updates Frame

    # Creates New Window
    page2 = Toplevel()
    page2.geometry('1152x700')
    page2.title('Classroom Lighting System')

    # Display Background
    bgimg = PhotoImage(file = os.path.join(image_path, "bg3.png"))
    bg = Label(page2, image = bgimg)
    bg.place(x = 0, y = 0)

    # People Counter and Light ON/OFF Display
    people_counter = StringVar()
    counter_label = Label(page2, textvariable=people_counter, font=("Arial", 20), 
                            background='green', foreground='white')
    counter_label.place(x=675, y=300)

    light_status = StringVar()
    status_label = Label(page2, textvariable=light_status, font=("Arial", 25), 
                            background='blue', foreground='white')
    status_label.place(x=750, y=350)

    # Webcam Display
    cam = Label(page2)
    cam.place(x = 100, y = 180)

    display_cam()
    page2.mainloop()

# Tkinter Main Window (Home Page)
# Main Canvas
canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 700,
    width = 1152,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
canvas.place(x = 0, y = 0)

# Background
bg_img = PhotoImage(
    file=os.path.join(image_path, "bg1.png"))
bg = canvas.create_image(
    576.0,
    350.0,
    image=bg_img
)

# Button To Waste Classifier Window
btn1_img = PhotoImage(
    file=os.path.join(image_path, "btn1.png"))
btn1 = Button(
    image=btn1_img,
    borderwidth=8,
    highlightthickness=0,
    command=display_page1,
    relief=RAISED
)
btn1.place(
    x=260.0,
    y=258.0,
    width=256.0,
    height=250.0
)

# Button To Classroom Lighting Window
btn2_img = PhotoImage(
    file=os.path.join(image_path, "btn2.png"))
btn2 = Button(
    image=btn2_img,
    borderwidth=8,
    highlightthickness=0,
    command=display_page2,
    relief=RAISED
)
btn2.place(
    x=636.0,
    y=258.0,
    width=256.0,
    height=250.0
)

window.mainloop()
