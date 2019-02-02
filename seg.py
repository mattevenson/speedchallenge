#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import sys
import random
import math
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append('Mask_RCNN')

from Mask_RCNN.samples.coco import coco
from Mask_RCNN.mrcnn.model import MaskRCNN

# moving object class ids are 1-9

batch_size = 4

train_paths = [f'data/train/raw/{f}' for f in os.listdir('data/train/raw')]
test_paths = [f'data/test/raw/{f}' for f in os.listdir('data/test/raw')]

def calculate_mean():
    sample = random.sample(train_paths, math.ceil(len(train_paths) * 0.05))
    sample.extend(random.sample(test_paths, math.ceil(len(test_paths) * 0.05)))

    mean = np.mean([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).mean(axis=0).mean(axis=0) for f in sample], axis=0)

    return mean

mean = calculate_mean()

class CommaConfig(coco.CocoConfig):
    NAME = 'comma.ai'

    GPU_COUNT = 1
    IMAGES_PER_GPU = batch_size

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    MEAN_PIXEL = mean

if __name__ == '__main__':
    config = CommaConfig()
    model = MaskRCNN(mode="inference", model_dir="logs", config=config)
    model.load_weights('models/mask_rcnn_coco.h5', by_name=True)

    for image_paths in [train_paths, test_paths]:
        output_dir = os.path.join(*image_paths[0].split('/')[:2], 'seg')
        os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch = []

            for j in range(batch_size):
                if i+j >= len(image_paths):
                    batch.append(batch[-1])
		        else:
			        image = cv2.cvtColor(cv2.imread(image_paths[i+j]), cv2.COLOR_BGR2RGB)
                	batch.append(image)

            batch_detections = model.detect(batch)

            for j in range(batch_size):
                mask = np.zeros((256, 512))

                detections = batch_detections[j]

                for k in range(len(detections['scores'])):
                    if 1 <= detections['class_ids'][k] <= 9:
                        mask[detections['masks'][:, :, k]] = 255

                output_name = os.path.split(image_paths[i+j])[-1]
                cv2.imwrite(os.path.join(output_dir, output_name), mask)




