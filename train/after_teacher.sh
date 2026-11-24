#!/bin/bash
# Wait for the teacher, then run the KD experiments (CIFAR-100).
cd ~/tallinn/train
while pgrep -f "[t]rain_teacher.py" >/dev/null; do sleep 60; done
T=../runs/teacher_resnet18_cifar100/best.pt
/usr/bin/python3.12 train_mmnv2.py cifar100 --epochs 300 --wd 1e-4 --qat-epochs 0 --teacher $T --tag _kd_wd1e-4_e300 > ../runs/mmnv2_w4_cifar100_kd_wd1e-4_e300.log 2>&1 &
/usr/bin/python3.12 train_cct.py cifar100 --teacher $T --tag _kd > ../runs/cct_cifar100_kd.log 2>&1 &
wait
