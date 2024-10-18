#!/bin/bash

################################ CIFAR ###################################
datasets=CIFAR100
device=0
seed=0
model=resnet18 # PreResNet164 WideResNet16x8 resnet18 VGG16BN
schedule=cosine
wd=0.001
epoch=200
bz=256
lr=0.1
opt=SGD
std=0.001
eta=1
inc=10
for model in VGG16BN
do
for opt in ARWP
do
    for eta in 0.1
    do
    for beta in 0.99
    do
        DST=results/$opt/$datasets/cos_$opt\_cutout_beta$beta\_std$std\_inc$inc\_eta$eta\_$epoch\_$bz\_$lr\_$model\_$wd\_$datasets\_$schedule\_seed$seed
        CUDA_VISIBLE_DEVICES=$device python -u train_rwp_cos.py --datasets $datasets \
                --arch=$model --epochs=$epoch --wd=$wd --randomseed $seed --lr $lr  --cutout -b $bz \
                --save-dir=$DST/checkpoints --log-dir=$DST -p 200 --schedule $schedule --optimizer $opt --eta $eta \
                --std $std --inc $inc --beta $beta
    done
    done
done
done