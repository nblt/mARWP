#!/bin/bash

################################ CIFAR ######s#############################
datasets=CIFAR100
device=0
seed=1
model=resnet18 # resnet18 VGG16BN WideResNet16x8 WideResNet28x10
schedule=cosine
wd=0.001
epoch=200
bz=256
lr=0.1
sigma=0.015
lambda=0.5
eta=0.1
beta=0.99

DST=results/$model/$datasets/mfrwp_cutout_eta$eta\_beta$beta\_sigma$sigma\_lambda$lambda\_$epoch\_$bz\_$lr\_$model\_$wd\_$datasets\_$schedule\_seed$seed
CUDA_VISIBLE_DEVICES=$device python -u train_marwp.py --datasets $datasets \
        --arch=$model --epochs=$epoch --wd=$wd --randomseed $seed --lr $lr --sigma $sigma --lambda $lambda --cutout -b $bz \
        --save-dir=$DST/checkpoints --log-dir=$DST -p 200 --schedule $schedule --eta $eta --beta $beta
