from confidence import write_csv_l
from ultralytics import YOLO
import cv2
import util
import string
import easyocr
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import imutils


results = {}
reader = easyocr.Reader(['en'], gpu=False)

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./red old.mp4')

vehicles = [2, 3]

# read frames
frame_nmr = -1
ret = True
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# pts1 = np.float32([[305,245], [907,0], [524,875], [1085,640]])

pts1 = np.float32([[270,82], [940,0], [570,1023], [1080,678]])
pts2 = np.float32([[0,0],[1400,0],[0,1400],[1400,1400]])

count = 0

# Read until video is completed

try:
    while(cap.isOpened()):
        count+=1
        frame_nmr += 1

        # Capture frame-by-frame
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
            
            # track_ids = mot_tracker.update(np.asarray(detections_))

            # dst = imutils.rotate(dst, 2)
            license_plates = license_plate_detector(dst)[0]
            

            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                # xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if score*100 > 70:
                
                    # cv2.imshow('Output', dst)
                    license_plate_crop = dst[int(y1):int(y2), int(x1): int(x2), :]

                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

                    # Sharpen the image 
                    license_plate_crop = cv2.filter2D(license_plate_crop, -1, kernel)

                    

                    # cv2.imwrite('/frame{:d}.jpg'.format(count), license_plate_crop)
                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    # thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 72, 255, cv2.THRESH_BINARY_INV)
                    
                    # license_plate_crop_thresh = cv2.GaussianBlur(license_plate_crop_thresh, (2, 2), 0)
                    # license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,27,15)

                    cv2.imwrite('frame/frame{:d}.jpg'.format(count), license_plate_crop_gray)
                    
                    detections = reader.readtext(license_plate_crop_gray)
                    
                    for detection in detections:
                        bbox, text, text_score = detection
                        if (text_score*100 > 40):
                            
                            write_csv_l([text, text_score])


                # cv2.imwrite('frame{:d}.jpg'.format(count), license_plate_crop)
    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
except Exception as e: 
    print(e)
    pass
 
# When everything done, release the video capture object\
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

