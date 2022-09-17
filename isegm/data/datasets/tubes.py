from pathlib import Path
import pickle
import random
import numpy as np
import json
import cv2
import os
from copy import deepcopy
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class TubesDataset(ISDataset):
    def __init__(self, dataset_path="/home/bean/lab/SegFormer/data/all", split="train", **kwargs):
        super(TubesDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self._data_path = self.dataset_path / split
        self._images_path = self._data_path / 'images'
        self._insts_path = self._data_path / 'masks'

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        mask_name = image_name.replace("image", "mask")
        image_path = str(self._images_path / image_name)
        mask_path = str(self._insts_path / mask_name)

        image = cv2.imread(image_path)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)
