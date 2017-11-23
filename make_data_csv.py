import os
import sys
import random as rnd
PATIO_SIZE= 707
PATIO_TRAIN=607
PATIO_TEST=PATIO_SIZE-PATIO_TRAIN
NON_PATIO_SIZE= 9368
NON_PATIO_TRAIN=9268
NON_PATIO_TEST=NON_PATIO_SIZE-NON_PATIO_TRAIN
if len(sys.argv)==2:
    postfix='_'+sys.argv[1]
    print(postfix)
else:
    postfix=""
patio_faces=os.listdir('faces64%s'%(postfix))
non_patio_faces=os.listdir('faces-np64%s'%(postfix))
rnd.shuffle(patio_faces)
rnd.shuffle(non_patio_faces)

with open("train.csv","w") as f:
    cnt=0
    d=[]
    while cnt < PATIO_TRAIN*4:
        for i,name in enumerate(patio_faces[:PATIO_TRAIN]):
            cnt+=1
            if(cnt>PATIO_TRAIN*4):
                break
            d.append("faces64%s/%s,1\n"%(postfix,name))
    for i,name in enumerate(non_patio_faces[:NON_PATIO_TRAIN]):
        d.append("faces-np64%s/%s,0\n"%(postfix,name))
    rnd.shuffle(d)
    for l in d:
        f.write(l)
with open("test.csv","w") as f:
    for i,name in enumerate(patio_faces[PATIO_TRAIN:PATIO_SIZE]):
        f.write("faces64%s/%s,1\n"%(postfix,name))
    for i,name in enumerate(non_patio_faces[NON_PATIO_TRAIN:NON_PATIO_SIZE]):
        f.write("faces-np64%s/%s,0\n"%(postfix,name))


