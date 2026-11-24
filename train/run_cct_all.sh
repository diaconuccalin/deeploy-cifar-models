#!/bin/bash
cd ~/tallinn/train
for ds in cifar10 cifar100; do
  /usr/bin/python3.12 train_cct.py $ds > ../runs/cct_${ds}.log 2>&1
done
