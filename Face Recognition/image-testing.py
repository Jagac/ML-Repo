import cv2
import os 
import numpy as np
import face_recognition as fr

test_img=cv2.imread(r'C:\Users\Administrator\Documents\GitHub\Big-and-Small-ML-Projects\Face Recognition\Test Images\test.jpeg')
faces_detected, gray_img = fr.faceDetection(test_img)
print('faces detected:', faces_detected)


faces, faceID = fr.labels_for_training_images(r'C:\Users\Administrator\Documents\GitHub\Big-and-Small-ML-Projects\Face Recognition\Training Images')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.write(r'C:\Users\Administrator\Documents\GitHub\Big-and-Small-ML-Projects\Face Recognition\traningData.yml')

face_recognizer = cv2.face_LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\Administrator\Documents\GitHub\Big-and-Small-ML-Projects\Face Recognition\traningData.yml')

name = {0: 'Jagos'} 