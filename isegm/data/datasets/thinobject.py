from pathlib import Path

import cv2
import numpy as np
import os
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class ThinObjectDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='img', masks_dir_name='gt',
                 init_mask_mode = None, **kwargs):
        super(ThinObjectDataset, self).__init__(**kwargs)
        self.name = 'ThinObject'
        # ====== ThinObjectDataset ======
        
        root = dataset_path #'/home/admin/workspace/project/data/datasets/ThinObject/ThinObject5K/'
        mask_dir = 'masks'
        image_dir = 'images'

        file_lst = os.listdir(os.path.join(root,mask_dir))
        mask_lst = [i for i in file_lst if '.png' in i]
        mask_image_dict = {}


        for i in mask_lst:
            image_name =  i.replace('.png','.jpg')
            image_path = os.path.join(root, image_dir, image_name) #root + image_dir + image_name
            mask_path = os.path.join(root, mask_dir, i) #root + mask_dir + i
            mask_image_dict[mask_path] = image_path
            
        self.mask_image_dict = mask_image_dict
        self.dataset_samples = list(self.mask_image_dict.keys() )

    def get_sample(self, index) -> DSample:
        mask_path = self.dataset_samples[index]
        image_path = self.mask_image_dict[mask_path]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path).astype(np.int32)
        if len(instances_mask.shape) == 3:
            instances_mask = instances_mask[:,:,0]
        instances_mask = instances_mask > 128
        instances_mask = instances_mask.astype(np.int32)


        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)
