from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class DavisDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='img', masks_dir_name='gt',
                 init_mask_mode = None, **kwargs):
        super(DavisDataset, self).__init__(**kwargs)
        self.name = 'Davis'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}
        self.init_mask_mode = init_mask_mode

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        init_mask_dir = None
        if self.init_mask_mode == 'low':
            init_mask_dir = '/home/admin/workspace/project/data/datasets/DAVIS_Edit/DAVIS_low/'
        elif self.init_mask_mode == 'mid':
            init_mask_dir = '/home/admin/workspace/project/data/datasets/DAVIS_Edit/DAVIS_mid/'
        elif self.init_mask_mode == 'high':
            init_mask_dir = '/home/admin/workspace/project/data/datasets/DAVIS_Edit/DAVIS_high/'

        if init_mask_dir is not None:
            init_mask_path = init_mask_dir + image_name.replace('.jpg','.png')
            init_mask = cv2.imread(init_mask_path)[:,:,0] > 0
        else:
            init_mask = None

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index, init_mask=init_mask)
