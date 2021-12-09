#!/bin/bash

sim=0.01
epochs=1
gpu=1

python evaluate_vae.py --gpu $gpu --id 0 --epochs $epochs --sim $sim
#nohup python evaluate_vae.py --gpu $gpu --id 1 --epochs $epochs --sim $sim
#nohup python evaluate_vae.py --gpu $gpu --id 2 --epochs $epochs --sim $sim
#nohup python evaluate_vae.py --gpu $gpu --id 3 --epochs $epochs --sim $sim
#nohup python evaluate_vae.py --gpu $gpu --id 4 --epochs $epochs --sim $sim
