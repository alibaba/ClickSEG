import numpy as np
import cv2
import os
import shutil
from PIL import  Image


def selectimagefromdir(dir_name = 'bear', select_number = 10):
    image_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/JPEGImages/Full-Resolution/' + dir_name + '/'
    mask_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/Annotations/Full-Resolution/' + dir_name + '/'

    save_image_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/IS_newdata/DAVIS/Selected/' + dir_name + '/'
    if os.path.exists(save_image_dir):
        shutil.rmtree(save_image_dir)
    os.mkdir(save_image_dir)
    image_lst = os.listdir(image_dir)
    image_lst = sorted(image_lst,reverse=True)
    step = int( len(image_lst) / select_number)
    for index in range(select_number):
        i = index * step
        image_name = image_lst[i]
        image_path = image_dir + image_name
        save_image_path = save_image_dir + image_name
        image = cv2.imread(image_path)
        cv2.imwrite(save_image_path,image)

        mask_path = mask_dir + image_name.replace('.jpg','.png')
        mask = np.array(Image.open(mask_path),dtype=np.uint8)
        #print(mask.shape, np.unique(mask)) #(1080, 1920, 3) [  0 128]
        labels = np.unique(mask)[1:]
        for j in range(len(labels)):
            save_mask_path = save_image_dir + 'object' + str(j+1) + '_' + image_name.replace('.jpg','.png')
            mask_j = (mask == labels[j]) * 255
            cv2.imwrite(save_mask_path,mask_j)



val_txt_path = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/ImageSets/2017/val.txt'
val_lst  = open(val_txt_path).readlines()
val_lst = [i.strip('\n') for i in val_lst ]
print(len(val_lst), val_lst)


for i in val_lst:
    print(i)
    selectimagefromdir(i)