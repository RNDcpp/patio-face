#-*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import sys
CLASSIFIER = cv2.CascadeClassifier('lbpcascade_animeface.xml')
TARGET = sys.argv[1]
DIR='target'

def getFaces(file_id,filename):
    image = cv2.imread(filename)
    size=max([image.shape[1], image.shape[0]])
    tmp_img= cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    tmp_img[:image.shape[0],:image.shape[1]]=image[:,:]
    image=tmp_img
    size=1200
    image=cv2.resize(image,(size,size))
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cnt=0
    angle_cnt=0
    while cnt==0 and angle_cnt < 35:
        faces = CLASSIFIER.detectMultiScale(gray_image)
        cnt=len(faces)
        if cnt==0:
            rotation_matrix = cv2.getRotationMatrix2D((size,size), -10, 1.0)
            gray_image=cv2.warpAffine(gray_image, rotation_matrix, (size,size), flags=cv2.INTER_CUBIC)
            angle_cnt+=1;
            continue
    print(faces)
    rotation_matrix = cv2.getRotationMatrix2D(tuple([int(gray_image.shape[0]/2), int(gray_image.shape[0]/2)]), -10*angle_cnt, 1.0)
    image=cv2.warpAffine(image,rotation_matrix,(size,size),flags=cv2.INTER_CUBIC)
    for i, (x,y,w,h) in enumerate(faces):
        face_image = image[y:y+h, x:x+w]
        (fpath,ext)=TARGET.split('.')
        face_image=cv2.resize(face_image,(64,64),interpolation = cv2.INTER_AREA)
        output_path = '%s_%d.jpg'%(fpath,i)
        print(output_path)
        cv2.imwrite(output_path,face_image)


getFaces(0,TARGET)
