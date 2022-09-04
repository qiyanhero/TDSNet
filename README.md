# TDSNet

Code release for the paper "A Task-aware Dual Similarity Network for Fine-grained Few-shot Learning", by Yan Qi, Han Sun, Ningzhong Liu, and Huiyu Zhou.

## Requirements

* python=3.6
* PyTorch=1.2+
* torchvision=0.4.2
* pillow=6.2.1
* numpy=1.18.1
* h5py=1.10.2

## Dataset

#### CUB-200-2011

* Change directory to `./filelists/CUB`
* run `source ./download_CUB.sh`

## Train
```shell
python ./train.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 1 --train_aug --gpu 0
python ./train.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0
```

## Save features

```shell
python ./save_features.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 1 --train_aug --gpu 0
python ./save_features.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0

```

## Test

```shell
python ./test.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 1 --train_aug --gpu 0
python ./test.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0
```
