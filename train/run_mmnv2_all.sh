#!/bin/bash
cd ~/tallinn/train
for ds in cifar10 cifar100; do
  /usr/bin/python3.12 train_mmnv2.py $ds > ../runs/mmnv2_w4_${ds}.log 2>&1
done
