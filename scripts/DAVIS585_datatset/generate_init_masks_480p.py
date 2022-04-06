import numpy as np
import cv2
import os
import shutil
from PIL import  Image
from .superpixel_augmentation import superpixel_aug
from tqdm import tqdm

def selectimagefromdir(dir_name = 'bear', select_number = 10):
    image_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/DAVIS480P/JPEGImages/480p/' + dir_name + '/'
    mask_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/DAVIS480P/Annotations/480p/' + dir_name + '/'

    save_image_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/IS_newdata/DAVIS/Selected_480P/' + dir_name + '/'
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
        #cv2.imwrite(save_image_path,image)

        mask_path = mask_dir + image_name.replace('.jpg','.png')
        mask = np.array(Image.open(mask_path),dtype=np.uint8)
        #print(mask.shape, np.unique(mask)) #(1080, 1920, 3) [  0 128]
        labels = np.unique(mask)[1:]
        for j in range(len(labels)):
            save_mask_path = save_image_dir + 'object' + str(j+1) + '_' + image_name.replace('.jpg','.png')
            mask_j = (mask == labels[j])
            if mask_j.sum() > 300:  #100 * 100:
                cv2.imwrite(save_mask_path,mask_j * 255)
                cv2.imwrite(save_image_path, image)



def generate_init_mask_stm(dir_name = 'bike-packing'):
    stm_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/IS_newdata/DAVIS/stm/'+ dir_name + '/'
    gt_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/IS_newdata/DAVIS/Selected_480P/'+ dir_name + '/'
    image_lst = os.listdir(gt_dir)
    #print(image_lst)
    mask_lst = []
    for i in image_lst:
        if '.png' in i and 'init' not in i:
            mask_lst.append(i)

    for mask_name in mask_lst:
        obj_id = int(mask_name[6])
        gt_mask_name = mask_name.split('_')[-1]
        mask = cv2.imread(gt_dir+mask_name)
        H, W = mask.shape[0], mask.shape[1]


        #print(obj_id, gt_mask_name)
        stm_mask_path = stm_dir + gt_mask_name
        stm_mask = np.array(Image.open(stm_mask_path), dtype=np.uint8)
        stm_mask = (stm_mask == obj_id) * 255

        stm_mask = ((cv2.resize(stm_mask.astype(np.uint8),(W,H)) > 128) * 255).astype(np.uint8)


        init_mask_name = 'init_stm_' + mask_name
        init_mask_path = gt_dir + init_mask_name
        cv2.imwrite(init_mask_path,stm_mask)


def generate_init_mask_sp(dir_name = 'bike-packing'):
    gt_dir = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/IS_newdata/DAVIS/Selected_480P/'+ dir_name + '/'
    image_lst = os.listdir(gt_dir)
    #print(image_lst)
    mask_lst = []
    for i in image_lst:
        if '.png' in i and 'init' not in i:
            mask_lst.append(i)

    for mask_name in tqdm(mask_lst):
        gt_mask_path= gt_dir + mask_name
        gt_mask = cv2.imread(gt_mask_path)
        image_path = gt_dir + mask_name.split('_')[-1].replace('.png','.jpg')
        image = cv2.imread(image_path)

        sp_mask = superpixel_aug(image, gt_mask[:,:,0]>128,0.85,0.75)[0]
        sp_mask = (np.stack([sp_mask,sp_mask,sp_mask],-1) * 255).astype(np.uint8)

        init_mask_name = 'init_sp_' + mask_name
        init_mask_path = gt_dir + init_mask_name
        cv2.imwrite(init_mask_path,sp_mask)

val_txt_path = '/Users/chenxi/Desktop/Ali_projects/DATA/DAVIS/ImageSets/2017/val.txt'
val_lst  = open(val_txt_path).readlines()
val_lst = [i.strip('\n') for i in val_lst ]
print(len(val_lst), val_lst)

#generate_init_mask_sp()

for i in val_lst:
    print(i)
    selectimagefromdir(i)
    generate_init_mask_stm(i)
    generate_init_mask_sp(i)