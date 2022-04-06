from pathlib import Path
import cv2
import numpy as np
import os.path as osp
import os
import json
from PIL import Image
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class YouTubeDataset(ISDataset):
    def __init__(self, dataset_path=None,
                 **kwargs):
        super(YouTubeDataset, self).__init__(**kwargs)
        self.name = 'YoutubeVOS'
        self.subset = 'train'
        #dataset_path = '/home/admin/workspace/project/data/datasets/YTBVOS'
        self.dataset_path = Path(dataset_path)
        dataset_root = self.dataset_path
        self.image_root = osp.join(dataset_root, self.subset, "JPEGImages")
        self.anno_root = osp.join(dataset_root, self.subset, "Annotations")
        meta_file = osp.join(dataset_root, self.subset, "meta.json")



        video_dirs = []

        with open(meta_file) as f:
            records = json.load(f)
            records = records["videos"]
            for video_id in records:
                video_dirs.append(video_id)
        self.video_dirs = video_dirs
        self.dataset_samples = video_dirs


    def get_sample(self, index):
        video_dir = self.video_dirs[index]
        anno_dir = osp.join(self.anno_root,video_dir)
        image_dir = osp.join(self.image_root,video_dir)
        anno_names = os.listdir(anno_dir)
        anno_name = np.random.choice(anno_names)
        image_name = anno_name.replace('png','jpg')
        anno_path = osp.join(anno_dir,anno_name)
        image_path = osp.join(image_dir,image_name)

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instances_mask = Image.open(anno_path).convert('P')
        instances_mask = np.array(instances_mask).astype(np.int32)
        #image = resize_thresh(image)
        #instances_mask = resize_thresh(instances_mask)
        instances_ids = np.unique(instances_mask)
        instances_ids = [ i for i in instances_ids if i != 0]

        #print(image.shape, instances_mask.shape, instances_ids   )
        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }

 
        #in_id = np.random.choice(instances_ids)
        #instances_mask = (instances_mask == in_id) * 255
        #print(image.shape, instances_mask.shape, instances_ids)

        return DSample(image, instances_mask, objects_ids=instances_ids, ignore_ids=[-1], sample_id=index)

def resize_thresh(image, thresh = 1280):
    h,w = image.shape[0], image.shape[1]
    image = image.astype(np.uint8)
    if w >= h:
        if w > thresh:
            w_n = thresh
            h_n = h * thresh / w
            h_n = int(h_n)
            image = cv2.resize(image,(w_n,h_n), interpolation = cv2.INTER_NEAREST)
            return image
        else:
            return image
    else:
        if h > thresh:
            h_n = thresh
            w_n = w * thresh / h
            w_n = int(w_n)
            image = cv2.resize(image,(w_n,h_n), interpolation = cv2.INTER_NEAREST)
            return image
        else:
            return image


