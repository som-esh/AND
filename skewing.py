from confidence import write_csv_l
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

results = {}
reader = easyocr.Reader(['en'], gpu=False)


coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

cap = cv2.VideoCapture('./red old.mp4')

vehicles = [2, 3]

frame_nmr = -1
ret = True
 

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 

pts1 = np.float32([[270,82], [940,0], [570,1023], [1080,678]])
pts2 = np.float32([[0,0],[1400,0],[0,1400],[1400,1400]])

count = 0

try:
    while(cap.isOpened()):
        count+=1
        frame_nmr += 1

        ret, frame = cap.read()
        if ret == True:
            results[frame_nmr] = {}
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(frame,M,(1400,1400))
            # cv2.imshow('Output', dst)

            detections = coco_model(dst)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
            
            license_plates = license_plate_detector(dst)[0]
            

            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                if score*100 > 70:
                
                    license_plate_crop = dst[int(y1):int(y2), int(x1): int(x2), :]

                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

                    license_plate_crop = cv2.filter2D(license_plate_crop, -1, kernel)

                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
 
                    cv2.imwrite('frame/frame{:d}.jpg'.format(count), license_plate_crop_gray)
                    
                    detections = reader.readtext(license_plate_crop_gray)
                    
                    for detection in detections:
                        bbox, text, text_score = detection
                        if (text_score*100 > 40):
                            
                            write_csv_l([text, text_score])
    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
except Exception as e: 
    print(e)
    pass
 
cap.release()

cv2.destroyAllWindows()

