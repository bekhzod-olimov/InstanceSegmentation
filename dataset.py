import torch, os
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
from os import path as osp

class CustomDataset(Dataset):
    
    def __init__(self, root, transformations=None):
        
        self.root = root
        self.t = transformations
        self.ims_paths = glob(osp.join(self.root, "PNGImages/*"))
        self.masks_paths = glob(osp.join(self.root, "PedMasks/*"))
        
    def __len__(self): return len(self.ims_paths)

    def __getitem__(self, idx):
        
        # get an input image
        im = Image.open(self.ims_paths[idx]).convert("RGB")
        # get a corresponding mask
        mask = np.array(Image.open(self.masks_paths[idx]))
        # get objects in the mask
        objects = np.unique(mask)
        # remove the first object since it is a "background" class
        objects = objects[1:]
        # get mask for each class
        masks = mask == objects[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        num_objects = len(objects)
        
        # get bounding boxes
        bboxes = []
        for i in range(num_objects):
            # get locations for each object
            box = np.where(masks[i])
            xmin = np.min(box[1])
            xmax = np.max(box[1])
            ymin = np.min(box[0])
            ymax = np.max(box[0])
            bboxes.append([xmin, ymin, xmax, ymax])
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # since there is a single class, we use torch.ones()
        labels = torch.ones((num_objects, ), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((num_objects, ), dtype=torch.int64)
        
        # create a dictionary for targets
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # apply transformations
        if self.t:
            
            im, target = self.t(im, target)

            return im, target
