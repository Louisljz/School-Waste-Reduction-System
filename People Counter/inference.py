from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            cls = model.names[int(box.cls)]
            conf = box.conf[0]
            print(conf)
            if cls == 'person' and conf > 0.5:
                annotator.box_label(b, cls, color=(0, 0, 255))
          
    frame = annotator.result() 
    cv2.imshow('YOLO V8 Detection', frame)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
