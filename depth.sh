#!/usr/bin/env bash

mkdir data/train/depth

for file in data/train/raw/*.png; do
    python monodepth/monodepth_simple.py --image_path $file --checkpoint_path models/model_city2kitti;
done

rm data/train/raw/*.npy
mv data/train/raw/*_disp.png data/train/depth/

# for file in data/train/depth/*_disp.png; do
#     mv $file "${data/train/depth/file%_disp.png}.png";
# done

# mkdir data/test/depth

# for file in data/test/raw/*.png; do
#     python monodepth/monodepth_simple.py --image_path $file --checkpoint_path models/model_city2kitti;
# done

# rm data/test/raw/*.npy
# mv data/test/raw/*_disp.png data/test/depth/

# for file in data/test/depth/*_disp.png; do
#     mv $file "${data/test/depth/file%_disp.png}.png";
# done


