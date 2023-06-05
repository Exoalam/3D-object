from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import threading

cam1 = 0
cam2 = 0
rand = 100

def percentage(frame_h, frame_w, h, w):
    return int((h*w)/(frame_h*frame_w)*100)

def percentage_save( f_n, cls, conf, percentage):
    global rand
    global frame
    global frame2
    if f_n == 1:
        if percentage == 1:
            f = open("PerDataset/1/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/1/Images"+str(rand)+".png", frame)  
        if percentage == 5:
            f = open("PerDataset/5/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/5/Images"+str(rand)+".png", frame)  
        if percentage == 10:
            f = open("PerDataset/10/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/10/Images"+str(rand)+".png", frame)  
        if percentage == 20:
            f = open("PerDataset/20/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/20/Images"+str(rand)+".png", frame)       
        if percentage == 30:
            f = open("PerDataset/30/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/30/Images"+str(rand)+".png", frame)  
        if percentage == 50:
            f = open("PerDataset/50/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/50/Images"+str(rand)+".png", frame)  
        if percentage == 75:
            f = open("PerDataset/75/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/75/Images"+str(rand)+".png", frame)  
        if percentage == 90:
            f = open("PerDataset/90/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/90/Images"+str(rand)+".png", frame)  
        if percentage == 100:
            f = open("PerDataset/100/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/100/Images"+str(rand)+".png", frame)  
    if f_n == 2:
        if percentage == 1:
            f = open("PerDataset/1/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/1/Images"+str(rand)+".png", frame2)  
        if percentage == 5:
            f = open("PerDataset/5/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/5/Images"+str(rand)+".png", frame2)  
        if percentage == 10:
            f = open("PerDataset/10/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/10/Images"+str(rand)+".png", frame2)  
        if percentage == 20:
            f = open("PerDataset/20/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/20/Images"+str(rand)+".png", frame2)       
        if percentage == 30:
            f = open("PerDataset/30/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/30/Images"+str(rand)+".png", frame2)  
        if percentage == 50:
            f = open("PerDataset/50/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/50/Images"+str(rand)+".png", frame2)  
        if percentage == 75:
            f = open("PerDataset/75/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/75/Images"+str(rand)+".png", frame2)  
        if percentage == 90:
            f = open("PerDataset/90/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/90/Images"+str(rand)+".png", frame2)  
        if percentage == 100:
            f = open("PerDataset/100/Labels"+str(rand)+".txt", "w")
            f.write(str(cls)+" "+str(conf)+" "+str(percentage))
            f.close()
            cv2.imwrite("PerDataset/100/Images"+str(rand)+".png", frame2)             
                         
def collect_data(data, f_n):
    global rand
    global frame
    global frame2
    if f_n == 1:
        f = open("Dataset/Labels"+str(rand)+".txt", "w")
        f.write(str(data))
        f.close()
        cv2.imwrite("Dataset/Images"+str(rand)+".png", frame)
    if f_n == 2:
        f = open("Dataset/Labels"+str(rand)+".txt", "w")
        f.write(str(data))
        f.close()
        cv2.imwrite("Dataset/Images"+str(rand)+".png", frame2)
    rand += 1        

def predict(frame, results):
    cam1 = 0
    for r in results:
        annotator = Annotator(frame)
        print(frame.shape)
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0] 
            _b = box.xywh[0].detach().cpu().numpy()
            
            c = box.cls
            if c == 41:
                cam1 = 1
                annotator.box_label(b, model.names[int(c)])
                if cam2 == 0:
                    collect_data(box, 2)
          
    frame = annotator.result() 
    print(cam1) 
    cv2.imshow('YOLO V8 Detection', frame)  

def predict2(frame, results):
    cam2 = 0
    for r in results:
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0] 
            _b = box.xywh[0].detach().cpu().numpy()
            c = box.cls
            if c == 41:
                cam2 = 1
                print(_b) 
                annotator.box_label(b, model.names[int(c)])
                if cam1 == 0:
                    collect_data(box, 1)
          
    frame = annotator.result()  
    print(cam2)
    cv2.imshow('YOLO V8 Detection2', frame)      


model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
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