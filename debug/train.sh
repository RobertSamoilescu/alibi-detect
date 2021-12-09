#!/bin/bash

sim=0.01
epochs=25
gpu=1

if [ ! -d "logs" ]
then
  mkdir logs
fi

nohup python train_vae.py --gpu $gpu --id 0 --epochs $epochs --sim $sim > logs/0_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 1 --epochs $epochs --sim $sim > logs/1_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 2 --epochs $epochs --sim $sim > logs/2_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 3 --epochs $epochs --sim $sim > logs/3_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 4 --epochs $epochs --sim $sim > logs/4_sim=$sim.out 2>&1 &

sim=0.05
nohup python train_vae.py --gpu $gpu --id 0 --epochs $epochs --sim $sim > logs/0_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 1 --epochs $epochs --sim $sim > logs/1_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 2 --epochs $epochs --sim $sim > logs/2_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 3 --epochs $epochs --sim $sim > logs/3_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 4 --epochs $epochs --sim $sim > logs/4_sim=$sim.out 2>&1 &

sim=0.1
nohup python train_vae.py --gpu $gpu --id 0 --epochs $epochs --sim $sim > logs/0_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 1 --epochs $epochs --sim $sim > logs/1_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 2 --epochs $epochs --sim $sim > logs/2_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 3 --epochs $epochs --sim $sim > logs/3_sim=$sim.out 2>&1 &
nohup python train_vae.py --gpu $gpu --id 4 --epochs $epochs --sim $sim > logs/4_sim=$sim.out 2>&1 &


