import os
import sys
import random as rnd
PATIO_SIZE=700
PATIO_TRAIN=650
PATIO_TEST=PATIO_SIZE-PATIO_TRAIN
RND_SIZE=430
RND_TRAIN=380
RND_TEST=RND_SIZE-RND_TRAIN
if len(sys.argv)==2:
    postfix='_'+sys.argv[1]
    print(postfix)
else:
    postfix=""
patio_faces=os.listdir('faces64%s'%(postfix))
rnd_faces=os.listdir('faces-rnd64%s'%(postfix))
rnd.shuffle(patio_faces)
rnd.shuffle(rnd_faces)

with open("train.csv","w") as f:
    for i,name in enumerate(patio_faces[:PATIO_TRAIN]):
        f.write("faces64%s/%s,1\n"%(postfix,name))
    cnt=0
    while cnt < PATIO_TRAIN:
        for i,name in enumerate(rnd_faces[:RND_TRAIN]):
            cnt+=1
            if(cnt>PATIO_TRAIN):
                break
            f.write("faces-rnd64%s/%s,0\n"%(postfix,name))
with open("test.csv","w") as f:
    for i,name in enumerate(patio_faces[PATIO_TRAIN:PATIO_SIZE]):
        f.write("faces64%s/%s,1\n"%(postfix,name))
    for i,name in enumerate(rnd_faces[RND_TRAIN:RND_SIZE]):
        f.write("faces-rnd64%s/%s,0\n"%(postfix,name))


