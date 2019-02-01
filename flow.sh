#!/usr/bin/env bash

mkdir data/train/flow
mkdir data/test/flow

cd flownet2-pytorch

python main.py --inference --model FlowNet2 --save_flow --save ../data/train/flow --inference_dataset ImagesFromFolder --inference_dataset_root ../data/train/raw --resume ../models/FlowNet2_checkpoint.pth.tar
python main.py --inference --model FlowNet2 --save_flow --save ../data/test/flow --inference_dataset ImagesFromFolder --inference_dataset_root ../data/test/raw --resume ../models/FlowNet2_checkpoint.pth.tar

cd ..

./flow-io-opencv/cli/cli data/train/flow data/train/flow --vis-dir data/train/flow
./flow-io-opencv/cli/cli data/train/flow data/test/flow --vis-dir data/test/flow

for file in data/train/flow/*.flo.png; do
    mv $file "${data/train/flow/file%.flo.png}.png";
done

rm data/train/flow/*.txt
rm data/train/flow/*.flo

for file in data/test/flow/*.flo.png; do
    mv $file "${data/test/flow/file%.flo.png}.png";
done

rm data/test/flow/*.txt
rm data/test/flow/*.flo
