#!/usr/bin/env bash

git submodule update --init

mkdir data
mkdir data/train data/test
wget -P data https://github.com/commaai/speedchallenge/raw/master/data/train.mp4
wget -P data https://github.com/commaai/speedchallenge/raw/master/data/test.mp4
wget -P data https://raw.githubusercontent.com/commaai/speedchallenge/master/data/train.txt

mkdir models
gdown --id 1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da -O models/FlowNet2_checkpoint.pth.tar
chmod +x ./monodepth/utils/get_model.sh
sh ./monodepth/utils/get_model.sh model_city2kitti models
wget -P models https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
