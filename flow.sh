#!/usr/bin/env bash

mkdir data/train/flow
mkdir data/test/flow

cd flownet2-pytorch

python main.py --inference --model FlowNet2 --save_flow --save ../data/train/flow --inference_dataset ImagesFromFolder --inference_dataset_root ../data/train/raw --resume ../models/FlowNet2_checkpoint.pth.tar
python main.py --inference --model FlowNet2 --save_flow --save ../data/test/flow --inference_dataset ImagesFromFolder --inference_dataset_root ../data/test/raw --resume ../models/FlowNet2_checkpoint.pth.tar

cd ..

./flow-io-opencv/cli/cli data/train/flow/inference/run.epoch-0-flow-field data/train/flow --vis-dir data/train/flow
./flow-io-opencv/cli/cli data/test/flow/inference/run.epoch-0-flow-field data/train/flow --vis-dir data/test/flow

rm data/train/flow/*.txt
rm -rf data/train/flow/inference
rm -rf data/train/flow/train
rm -rf data/train/flow/validation
rm data/train/flow/arg.txt

for file in data/train/flow/*.flo.png; do
    mv $file "${data/train/flow/file%.flo.png}.png";
done

rm data/test/flow/*.txt
rm -rf data/test/flow/inference
rm -rf data/test/flow/train
rm -rf data/test/flow/validation
rm data/test/flow/arg.txt

for file in data/test/flow/*.flo.png; do
    mv $file "${data/test/flow/file%.flo.png}.png";
done


