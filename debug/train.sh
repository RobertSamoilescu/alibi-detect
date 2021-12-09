#!/bin/bash

sim=0.01
epochs=50
gpu=1

if [ ! -d "logs" ]
then
  mkdir logs
fi

nohup python train_vae.py --gpu $gpu --id 0 --epochs $epochs --sim $sim > logs/0.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 1 --epochs $epochs --sim $sim > logs/1.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 2 --epochs $epochs --sim $sim > logs/2.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 3 --epochs $epochs --sim $sim > logs/3.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 4 --epochs $epochs --sim $sim > logs/4.out 2>&1 &
