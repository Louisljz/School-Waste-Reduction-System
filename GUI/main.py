from tkinter import *
from PIL import Image, ImageTk
import sys, os
import cv2
from keras.models import load_model
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

if getattr(sys, 'frozen', False):
    folder_path = os.path.dirname(sys.executable)
else:
    folder_path = os.path.dirname(__file__)

image_path = os.path.join(folder_path, 'images/')
model_path = os.path.join(folder_path, 'models/')


window = Tk()
cap = cv2.VideoCapture(0)

window.geometry("1152x700")
window.configure(bg = "#FFFFFF")
window.title('School Waste Reduction System')
window.resizable(False, False)

yolo = YOLO(os.path.join(model_path, 'yolov8n.pt'))
model = load_model(os.path.join(model_path, "waste_classifier.h5"), compile=False)
class_names = open(os.path.join(model_path, "labels.txt"), "r").readlines()

def display_page1():
    def display_cam():
        _, frame = cap.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        name = class_name[2:].strip()
        if name != 'Background':
            message.set("AI classifies this Trash as: ")
            label.set(name)
        else:
            message.set('')
            label.set('')
        
        cv2image = cv2.resize(frame, (540, 380))
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image = img)
        cam.imgtk = imgtk
        cam.configure(image=imgtk)
        
        cam.after(20, display_cam)
    
    page1 = Toplevel()
    page1.geometry('1152x700')
    page1.title('Canteen Waste Classifier')

    bgimg = PhotoImage(file = os.path.join(image_path, "image_2.png"))
    bg = Label(page1, image = bgimg)
    bg.place(x = 0, y = 0)

    message = StringVar()
    message_label = Label(page1, textvariable=message, font=("Arial", 25), 
                            background='green', foreground='white')
    message_label.place(x=700, y=300)

    label = StringVar()
    class_label = Label(page1, textvariable=label, font=("Arial", 25), 
                            background='blue', foreground='white')
    class_label.place(x=770, y=350)

    cam = Label(page1)
    cam.place(x = 100, y = 180)

    display_cam()
    page1.mainloop()

def display_page2():
    global duration
    duration = 0
    def display_cam():
        global duration
        counter = 0
        _, img = cap.read()
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = yolo.predict(img)

        for r in results:
            
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                cls = yolo.names[int(box.cls)]
                conf = box.conf[0]
                if cls == 'person' and conf > 0.5:
                    counter += 1
                    annotator.box_label(b, color=(0, 255, 0))
            
        frame = annotator.result() 
        cv2image = cv2.resize(frame, (540, 380))
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image = img)
        cam.imgtk = imgtk
        cam.configure(image=imgtk)

        people_counter.set(f"Number of People in Classroom: {counter}")

        if counter >= 1:
            if duration < 30:
                duration += 1
        else:
            if duration > 0:
                duration -= 1
        
        if duration == 30:
            light_status.set('Lights: ON')
        elif duration == 0:
            light_status.set('Lights: OFF')
        
        cam.after(20, display_cam)

    page2 = Toplevel()
    page2.geometry('1152x700')
    page2.title('Classroom Lighting System')

    bgimg = PhotoImage(file = os.path.join(image_path, "image_3.png"))
    bg = Label(page2, image = bgimg)
    bg.place(x = 0, y = 0)

    people_counter = StringVar()
    counter_label = Label(page2, textvariable=people_counter, font=("Arial", 20), 
                            background='green', foreground='white')
    counter_label.place(x=675, y=300)

    light_status = StringVar()
    status_label = Label(page2, textvariable=light_status, font=("Arial", 25), 
                            background='blue', foreground='white')
    status_label.place(x=750, y=350)

    cam = Label(page2)
    cam.place(x = 100, y = 180)

    display_cam()
    page2.mainloop()


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
image_image_1 = PhotoImage(
    file=os.path.join(image_path, "image_1.png"))
image_1 = canvas.create_image(
    576.0,
    350.0,
    image=image_image_1
)

button_image_1 = PhotoImage(
    file=os.path.join(image_path, "button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=display_page1,
    relief="flat"
)
button_1.place(
    x=636.0,
    y=258.0,
    width=256.0,
    height=250.0
)

button_image_2 = PhotoImage(
    file=os.path.join(image_path, "button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=display_page2,
    relief="flat"
)
button_2.place(
    x=260.0,
    y=258.0,
    width=256.0,
    height=250.0
)

window.mainloop()
