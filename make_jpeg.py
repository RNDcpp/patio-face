#-*- coding:utf-8 -*-
import os
import cv2
import sys
DIR = sys.argv[2]
SRC = sys.argv[1]
FILES=os.listdir(SRC)

def convert2JPEG(file_id,filename):
    name=os.path.basename(filename)
    name,ext=os.path.splitext(name)
    image = cv2.imread(filename)
    cv2.imwrite(DIR+'/'+name+'.jpg',image)

for i,name in enumerate(FILES):
    print(i,name)
    convert2JPEG(i,os.path.join(SRC,name))
