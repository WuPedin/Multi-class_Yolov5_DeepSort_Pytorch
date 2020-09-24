import xml.etree.ElementTree as ET
import pickle
import os
#from os import listdir, getcwd
import os
from os.path import join
import glob
import numpy as np
import shutil

# ratio of sample size in train and val
ratio = [0.8, 0.2]


# Functions
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(in_file, out_file, classes):
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        
# Main		
def main():

    # Initial
    folders = ['dataset/images/train', 'dataset/images/val', 'dataset/labels/train', 'dataset/labels/val']
    for folder in folders:
        if os.path.isdir(folder)==False:
            os.makedirs(folder)
            print('Creating %s' % folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

    # Get Classes
    file = open('yolov5/data/data.yaml', 'r')
    for line in file:
        if line.find('names')!=-1:
            line = line.split('[')[-1].split(']')[0]
            classes = line.split("'") 
            classes = [j for i,j in enumerate(classes) if i%2==1]

    # Split
    ann_files = glob.glob('dataset/annotations/*.xml')
    size = len(ann_files)
    train_size = int(size*ratio[0])
    dataset = []
    train_list = list(np.random.choice(ann_files, train_size, replace=False))
    dataset.append(train_list) # Train
    dataset.append([i for i in ann_files if i not in train_list]) # Val

    # Convert xml to txt
    for dataset_idx in range(len(dataset)):
        for file in dataset[dataset_idx]:
            file_name = os.path.basename(file).replace('.xml', '.txt')
            print('Creating %s ....' % file_name)
            in_file = open(file, 'r')
            out_file = open('dataset/labels/%s/%s' % (['train', 'val'][dataset_idx], file_name), 'w')
            convert_annotation(in_file, out_file, classes)
            in_file.close()
            out_file.close()
            os.rename('dataset/images/%s' % file_name.replace('.txt', '.jpg'), 'dataset/images/%s/%s' % (['train', 'val'][dataset_idx], file_name.replace('.txt', '.jpg')))

    print('Finised....')


if __name__=='__main__':
	main()
