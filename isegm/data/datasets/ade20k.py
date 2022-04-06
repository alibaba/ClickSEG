import os
import random
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_labels_with_sizes


from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import os
import json
from PIL import Image


def iter_files(root_dir, with_part = False):
    image_path_dict = {}
    print('=== Loading ADE20k data ==== ')
    for root,dirs,files in tqdm(os.walk(root_dir)):
        for file in files:
            file_name = os.path.join(root,file)
            if file_name.endswith('.jpg'):
                mask_lst = []
                seg_path = file_name.replace('.jpg','_seg.png')
                mask_lst.append(seg_path)

                if with_part == True:
                    part_1_path = file_name.replace('.jpg','_parts_1.png')
                    if os.path.exists(part_1_path):
                        mask_lst.append(part_1_path)
                        part_2_path = file_name.replace('.jpg','_parts_2.png')
                        if os.path.exists(part_2_path):
                            mask_lst.append(part_2_path)
                image_path_dict[file_name] = mask_lst
    return image_path_dict




class ADE20kDataset(ISDataset):
    def __init__(self, dataset_path = None, with_part = False,
                 **kwargs):
        super(ADE20kDataset, self).__init__(**kwargs)
        self.name = 'ADE20K'
        self.subset = 'train'
        self.dataset_path = dataset_path + '/ADE20K_2016_07_26/images/training/'
        images_dict = iter_files(self.dataset_path, with_part=with_part)
        self.dataset_samples = images_dict
        image_lst = list(self.dataset_samples.keys() )
        self.image_lst = image_lst


    def get_sample(self, index):
        image_path = self.image_lst[index]
        mask_lst = self.dataset_samples[image_path]
        if len(mask_lst) > 1:
            idx = np.random.randint(len(mask_lst))
        else:
            idx = 0
        mask_path = mask_lst[idx]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instances_mask = cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED)[:,:,0].astype(np.int32)
        instances_ids = np.unique(instances_mask)
        instances_ids = [ i for i in instances_ids if i != 0]
        return DSample(image, instances_mask, objects_ids=instances_ids, ignore_ids=[-1], sample_id=index)



