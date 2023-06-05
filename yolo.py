from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import threading

cam1 = 0
cam2 = 0
rand = 1832


def collect_data(data, f_n):
    global rand
    global frame
    global frame2
    if f_n == 1:
        f = open("Dataset/Labels/Images"+str(rand)+".txt", "w")
        f.write(str(data))
        f.close()
        cv2.imwrite("Dataset/Images/Images"+str(rand)+".png", frame)
    if f_n == 2:
        f = open("Dataset/Labels/Images"+str(rand)+".txt", "w")
        f.write(str(data))
        f.close()
        cv2.imwrite("Dataset/Images/Images"+str(rand)+".png", frame2)
    rand += 1        

def predict(frame, results):
    cam1 = 0
    for r in results:
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            #b = box.xyxy[0].detach().cpu().numpy()  
            b = box.xywhn[0].detach().cpu().numpy()
            c = box.cls.detach().cpu().numpy()
            b_c = str(int(c)) + " " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + " " + str(b[3])
            #b_c = str(int(c)) + " " + str(int(b[0])/480) + " " + str(int(b[1])/640) + " " + str(int(b[2])/480) + " " + str(int(b[3])/640)
            if c == 0:
                cam1 = 1
                annotator.box_label(b, model.names[int(c)])
                if cam2 == 0:
                    #collect_data(box, 2)
                    collect_data(b_c, 2)
          
    frame = annotator.result() 
    print(cam1) 
    cv2.imshow('YOLO V8 Detection', frame)  

def predict2(frame, results):
    cam2 = 0
    for r in results:
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            b = box.xywhn[0].detach().cpu().numpy()  
            c = box.cls.detach().cpu().numpy()
            b_c = str(int(c)) + " " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + " " + str(b[3])
            #b_c = str(int(c)) + " " + str(int(b[0])/480) + " " + str(int(b[1])/640) + " " + str(int(b[2])/480) + " " + str(int(b[3])/640)
            cv2.circle(frame, (b[0],b[1]), 4, (0, 0, 255))
            if c == 0:
                cam2 = 1
                annotator.box_label(b, model.names[int(c)])
                if cam1 == 0:
                    #collect_data(box, 1)
                    
                    collect_data(b_c, 1)
          
    frame = annotator.result()  
    print(cam2)
    cv2.imshow('YOLO V8 Detection2', frame)      


model = YOLO('yolov8n.pt')
#model = YOLO('/home/shuvo/yolov8/runs/detect/train/weights/best.pt')
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture()
cap.set(3, 640)
cap.set(4, 480)
cap2.set(3, 640)
cap2.set(4, 480)
while True:
    _, frame = cap.read()
    _, frame2 = cap2.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    results = model.predict(img)
    results2 = model.predict(img2)
    t1 = threading.Thread(target=predict(frame, results), name='t1')
    t2 = threading.Thread(target=predict2(frame2, results2), name='t2')
    t1.start()
    t2.start()
    if cv2.waitKey(1) & 0xFF == ord(' '):
        t1.join()
        t2.join()
        break

cap.release()
cv2.destroyAllWindows()