#!/usr/bin/env python3

# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import sys
sys.path.append('monodepth')

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
from tqdm import tqdm

from monodepth.monodepth_model import *
from monodepth.monodepth_dataloader import *
from monodepth.average_gradients import *

checkpoint_path = 'models/model_city2kitti'
encoder = 'vgg'
input_height = 256
input_width = 512

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def main(_):
    params = monodepth_parameters(
        encoder=encoder,
        height=input_height,
        width=input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    left  = tf.placeholder(tf.float32, [2, 256, 512, 3])
    model = MonodepthModel(params, "test", left, None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    # PREDICT

    for input_dir in ['data/train/raw', 'data/test/raw']:
        image_paths = [os.path.join(input_dir, path) for path in sorted(os.listdir(input_dir))]

        print('Processing images in {}'.format(input_dir))

        output_dir = os.path.join(os.path.split(input_dir)[0], 'depth')
        os.makedirs(output_dir, exist_ok=True)

        for image_path in tqdm(image_paths):
            input_image = scipy.misc.imread(image_path, mode="RGB").astype(np.float32) / 255
            input_images = np.stack((input_image, np.fliplr(input_image)), 0)

            disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
            disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

            output_name = os.path.splitext(os.path.basename(image_path))[0]

            disp_to_img = disp_pp.squeeze()
            plt.imsave(os.path.join(output_dir, output_name), disp_to_img, cmap='plasma')

if __name__ == '__main__':
    tf.app.run()
