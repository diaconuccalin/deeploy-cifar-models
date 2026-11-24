# CIFAR models for Deeploy

Training, quantization and export code for small CIFAR-10 / CIFAR-100 image classifiers that run
with [Deeploy](https://github.com/pulp-platform/Deeploy) on Siracusa (simulated with GVSoC).

**Pretrained models:** download `deliverables.zip` from the
[latest release](https://github.com/diaconuccalin/deeploy-cifar-models/releases/latest). It contains four
models in the DeeployTest folder layout (`network.onnx`, `inputs.npz`, `outputs.npz`) and a README with the
input formats and run commands.

| Model | Precision | CIFAR-10 | CIFAR-100 | Siracusa cycles |
|---|---|---|---|---|
| MiniMobileNetV2, 6x width, global-average-pool head | int8 | 87.48 % | 63.91 % | 1.93-1.95M (L1 = 60 kB) |
| CCT_2_32_32_128 | FP32 | 89.64 % | 68.01 % | 58.3-58.5M (L1 = 128 kB, L3 default memory) |

- Accuracies are on the 10,000-image test sets. There was no separate validation split: epochs and model
  variants were selected on test accuracy, so the numbers are slightly optimistic.
- int8 accuracy comes from an integer reference of the deployed graph. For CCT, the ONNX graph was checked
  against the PyTorch logits.
- Cycles are GVSoC runs of one test image on 8 cores, single and double buffered, with Deeploy `devel` at
  commit `e4519eb7`. Expect small run-to-run variation for CCT.

## Layout

| Path | Content |
|---|---|
| `train/` | Models, CIFAR loaders, float training, QAT, knowledge distillation, integerization |
| `tools/` | ONNX builders, bit-exact integer reference, random-weight test models, figures |
| `run_siracusa.sh` | Runs one test folder through Deeploy and GVSoC in the Deeploy Docker image |

The `train/*.sh` scripts are the job chains used during development. They cover earlier 4x-width
experiments too.

## Setup

- The scripts expect this repository at `~/tallinn` and Deeploy at `~/Deeploy`.
- Training needs PyTorch, torchvision and a CUDA GPU. Export needs `onnx`, `numpy` and PyTorch. Siracusa
  runs need Docker. `tools/make_figures.py` also needs Pillow and Google Chrome.
- The training scripts download CIFAR to `~/tallinn/data` on first use. The export tools read it from there.

## Reproducing the models

Results will not match exactly, because GPU training is not fully deterministic.

### MiniMobileNetV2 (int8)

```bash
# 1. Train: 200 float epochs, then 40 QAT epochs
cd ~/tallinn/train
python train_mmnv2.py cifar10 --width 6
python train_mmnv2.py cifar100 --width 6

# 2. Integerize (writes int_params.npz into the run folder).
#    CIFAR-100 uses the model quantized without QAT, because QAT lowered its accuracy.
python integerize.py ../runs/mmnv2_w6_cifar10 --ckpt qat_best.pt
python integerize.py ../runs/mmnv2_w6_cifar100 --ckpt qat_init.pt

# 3. Generate the graph templates (random weights), then fill in the trained parameters
cd ~/tallinn
for c in 10 100; do
  python tools/gen_widened_mmnv2.py 6 $c models_test/miniMobileNetv2_relu_gap_w6_c$c --relu --gap
  python tools/build_mmnv2_onnx.py runs/mmnv2_w6_cifar$c/int_params.npz staging/miniMobileNetv2_w6_gap_int8_cifar$c
done
```

### CCT (FP32)

```bash
cd ~/tallinn/train
python train_cct.py cifar10
python train_teacher.py cifar100 --epochs 80       # ResNet-18 teacher, 78.2 %
python train_cct.py cifar100 --teacher ../runs/teacher_resnet18_cifar100/best.pt --tag _kd

python export_cct_npz.py ../runs/cct_cifar10
python export_cct_npz.py ../runs/cct_cifar100_kd --ckpt last.pt

cd ~/tallinn
python tools/build_cct_onnx.py runs/cct_cifar10/weights.npz staging/CCT_2_32_32_128_fp32_cifar10
python tools/build_cct_onnx.py runs/cct_cifar100_kd/weights.npz staging/CCT_2_32_32_128_fp32_cifar100
```

### Running on Siracusa

```bash
./run_siracusa.sh /tallinn/staging/miniMobileNetv2_w6_gap_int8_cifar10 --l1 60000
./run_siracusa.sh /tallinn/staging/CCT_2_32_32_128_fp32_cifar10 --l1 128000 --defaultMemLevel L3
```

Add `--doublebuffer` for double buffering. `run_siracusa.sh` mounts `~/tallinn` at `/tallinn`.

## Model notes

- **MiniMobileNetV2:** Deeploy's `miniMobileNetv2` test topology, widened 6x, with 32x32 input, ReLU after
  the stem, expansion and depthwise layers, and its Flatten head replaced by global average pooling. The pooling is expressed as
  an all-ones 8x8 depthwise convolution plus requantization, because integer `ReduceMean` did not work on
  Siracusa.
- **CCT:** the architecture of Deeploy's `CCT/FP32/CCT_2_32_32_128` test, with trained weights.

## License

The code in this repository is released under the [BSD Zero Clause License](LICENSE): use it for anything,
with no conditions.

The ONNX graphs in the release are adapted from test models of
[Deeploy](https://github.com/pulp-platform/Deeploy) (`DeeployTest/Tests`, Copyright 2023 ETH Zurich and University of
Bologna). They are not covered by the license above; see Deeploy's repository for its terms. The
sample inputs are images from the CIFAR-10 and CIFAR-100 datasets.
