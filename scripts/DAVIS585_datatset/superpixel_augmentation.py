import cv2
import numpy as np
import os
from fast_slic import Slic


def cal_iou(mask,gt):
    intersection = np.logical_and(mask, gt).sum()
    union = np.logical_or(mask,gt).sum()
    iou = intersection / union
    return iou


def superpixel_aug(image_np, mask_np, target_iou_max , target_iou_min ):
    prob_boundary_inner_outer = [0.65,0.25,0.1]
    num_defect_decay = 0.4
    max_num_defect = 5

    #num_defect_probs = np.array([num_defect_decay ** i for i in range(max_num_defect)])
    #num_defect_probs = num_defect_probs / num_defect_probs.sum()
    num_defect_probs = [0.2,0.2,0.2,0.2,0.2]
    num_defect = np.random.choice(5, 1, p=num_defect_probs)[0] + 1
    #print(num_defect, num_defect_probs)
    kernel = np.ones((5, 5), np.uint8)
    mask = mask_np.astype(np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=3)
    dilation = cv2.dilate(mask, kernel, iterations=3)

    boundary_mask = np.logical_and(np.logical_not(erosion), dilation)
    inner_mask = np.logical_and(mask, np.logical_not(boundary_mask))
    outer_mask = np.logical_and(np.logical_not(mask), np.logical_not(boundary_mask))
    regions = [boundary_mask,inner_mask,outer_mask]
    image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)  # You can convert the image to CIELAB space if you need.

    #mask_modified = mask
    number = 0
    while(1):
       mask_modified = mask
       num_defect = np.random.choice(5, 1, p=num_defect_probs)[0] + 1
       num_defect = 10
       for i in range(num_defect):
           region_index = np.random.choice(3,1,p=prob_boundary_inner_outer)[0]
           region = regions[region_index]
           #print(mask_np.sum(),region_index,region.sum() )
           if region.sum() < 5:
              region = mask_np

           #p = np.array( [0.3,1,1,0.5,0.4,0.2])
           p = np.array([1,1,1,1,1,1])
           num_pixel = np.random.choice([50,100,200,300,500,700],1,p = p/p.sum())[0]
           slic = Slic(num_components=num_pixel, compactness=10)
           assignment = slic.iterate(image_lab)  # Cluster Map
           indices = np.argwhere(region)
           y, x = indices[np.random.randint(0, len(indices))]
           selected_patch = assignment == assignment[y, x]
           if mask_modified[y, x] == 0:
               mask_modified = np.logical_or(mask_modified, selected_patch)
           else:
               mask_modified = np.logical_and(mask_modified, np.logical_not(selected_patch))
           iou = cal_iou(mask_modified, mask_np)
           if iou >= target_iou_min and  iou <= target_iou_max:
               return mask_modified, iou
           #print('failure iou :',iou, '  number : ',i)
           if iou < target_iou_min:
               break

       iou = cal_iou(mask_modified, mask_np)
       if iou >= target_iou_min and iou <= target_iou_max:
            return mask_modified, iou
       number += 1
       if number > 30:
         print('failure iou :', iou, '  number : ', number)
         return mask_modified, iou