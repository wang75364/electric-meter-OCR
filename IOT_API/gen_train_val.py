import os, random
from os.path import join, splitext

base = 'VOC2021'
source    = join('VOCdevkit',base, 'JPEGImages/')
train_txt = join('VOCdevkit',base, 'ImageSets/Main/train.txt')
val_txt   = join('VOCdevkit',base, 'ImageSets/Main/val.txt')

files = os.listdir(source)
random.shuffle(files)

f_train = open(train_txt,'a')
f_val = open(val_txt,'a')

for i, file in enumerate(files):
    name = splitext(file)[0]
    if (i >=len(files)*0.2):
        f_train.write(name+'\n')
    else:
        f_val.write(name+'\n')

f_train.close()
f_val.close()
