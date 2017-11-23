#-*- coding:utf-8 -*-
import os
import cv2
import sys
dname=sys.argv[1]
SRC = dname
DIR = dname+'64'
#DIR_EDGE = dname+'64_edge'
#DIR_GRAY = dname+'64_gray'
FILES=os.listdir(SRC)
for i,name in enumerate(FILES):
    print(i,name)
    image=cv2.imread(os.path.join(SRC,name))
    resized_image=cv2.resize(image,(64,64),interpolation = cv2.INTER_AREA)
    output_path = os.path.join(DIR,name)
    #output_path_edge = os.path.join(DIR_EDGE,name)
    #output_path_gray = os.path.join(DIR_GRAY,name)
    #gray_image=cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)
    #result = cv2.Canny(resized_image, 100, 200)
    cv2.imwrite(output_path,resized_image)
    #cv2.imwrite(output_path_edge,result)
    #cv2.imwrite(output_path_gray,gray_image)
