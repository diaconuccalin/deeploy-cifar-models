import os
import torch
import torchvision
import torchvision.transforms as T

ROOT = os.path.expanduser("~/tallinn/data")
MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}


def datasets(name, normalize=True, autoaugment=True):
    """normalize=False keeps inputs in [0,1] (pixel/255), as needed by the int8 model whose
    deployed input is the raw uint8 image."""
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    norm = [T.Normalize(MEAN[name], STD[name])] if normalize else []
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if autoaugment:
        aug.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    train = cls(ROOT, train=True, download=True, transform=T.Compose(aug + [T.ToTensor()] + norm))
    test = cls(ROOT, train=False, download=True, transform=T.Compose([T.ToTensor()] + norm))
    return train, test


def loaders(name, batch, workers, **kw):
    tr, te = datasets(name, **kw)
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True, num_workers=workers, drop_last=True,
                                        pin_memory=True, persistent_workers=True),
            torch.utils.data.DataLoader(te, 500, shuffle=False, num_workers=2, pin_memory=True))


@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        n += y.numel()
    return 100.0 * correct / n


class GPULoader:
    """Whole dataset on GPU, random crop (pad 4) + horizontal flip done on GPU. Yields
    (x in [0,1] or normalized, y). Much cheaper than CPU workers for simple augmentation."""

    def __init__(self, name, train, batch, normalize=False, device="cuda", augment=None):
        cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        ds = cls(ROOT, train=train, download=True)
        self.x = torch.from_numpy(ds.data).to(device).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        self.y = torch.tensor(ds.targets, device=device)
        self.batch, self.train = batch, train
        self.augment = train if augment is None else augment
        self.norm = None
        if normalize:
            self.norm = (torch.tensor(MEAN[name], device=device).view(1, 3, 1, 1),
                         torch.tensor(STD[name], device=device).view(1, 3, 1, 1))

    def __len__(self):
        n = len(self.y)
        return n // self.batch if self.train else (n + self.batch - 1) // self.batch

    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, device=self.y.device) if self.train else torch.arange(n, device=self.y.device)
        for i in range(len(self)):
            j = idx[i * self.batch:(i + 1) * self.batch]
            x = self.x[j].float() / 255.0
            if self.augment:
                b = x.shape[0]
                x = torch.nn.functional.pad(x, (4, 4, 4, 4))
                ox = torch.randint(0, 9, (b,), device=x.device)
                oy = torch.randint(0, 9, (b,), device=x.device)
                ar = torch.arange(32, device=x.device)
                rows = (oy[:, None] + ar[None]).view(b, 1, 32, 1).expand(b, 3, 32, 40)
                x = x.gather(2, rows)  # b,3,32,40
                cols = (ox[:, None] + ar[None]).view(b, 1, 1, 32).expand(b, 3, 32, 32)
                x = x.gather(3, cols)  # b,3,32,32
                flip = torch.rand(b, device=x.device) < 0.5
                x = torch.where(flip.view(b, 1, 1, 1), x.flip(3), x)
            if self.norm is not None:
                x = (x - self.norm[0]) / self.norm[1]
            yield x, self.y[j]
