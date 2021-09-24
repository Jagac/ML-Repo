import cv2
import os 
import numpy as np
import face_recognition as fr
#test
test_img=cv2.imread(r'C:\Users\Administrator\Documents\GitHub\Big-and-Small-ML-Projects\Face Recognition\Test Images\test.jpeg')
faces_detected, gray_img = fr.faceDetection(test_img)
print('faces detected:', faces_detected)


faces, faceID = fr.labels_for_training_images(r'C:\Users\Administrator\Documents\GitHub\Big-and-Small-ML-Projects\Face Recognition\Training Images')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.write(r'C:\Users\Administrator\Documents\GitHub\Big-and-Small-ML-Projects\Face Recognition\traningData.yml')

face_recognizer = cv2.face_LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\Administrator\Documents\GitHub\Big-and-Small-ML-Projects\Face Recognition\traningData.yml')

name = {0: 'Jagos'} 
for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y: y+h, x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)  #predict label of the img

    fr.draw_rect(test_img, face)
    predicted_name = name[label]

    if (confidence > 35): #if confidence is greater than 35 doesn't print the name
        continue
    fr.put_text(test_img, predicted_name, x, y)
    print('Confidence', confidence)
    print('Label', label)
    resized_img = cv2.resize(test_img, (500,700))
    cv2.imshow('face detection', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows