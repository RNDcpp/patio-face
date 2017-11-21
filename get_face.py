#-*- coding:utf-8 -*-
import os
import cv2
import sys
CLASSIFIER = cv2.CascadeClassifier('lbpcascade_animeface.xml')
DIR = sys.atgv[2]
SRC = sys.argv[1]
FILES=os.listdir(SRC)

def getFaces(file_id,filename):
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = CLASSIFIER.detectMultiScale(gray_image)
    print(faces)
    for i, (x,y,w,h) in enumerate(faces):
        face_image = image[y:y+h, x:x+w]
        output_path = os.path.join(DIR,'%d_%d.jpg'%(file_id,i))
        print(output_path)
        cv2.imwrite(output_path,face_image)

for i,name in enumerate(FILES):
    print(i,name)
    getFaces(i,os.path.join(SRC,name))
