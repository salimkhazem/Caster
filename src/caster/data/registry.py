"""Small dataset registry for CASTER experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class ArrowImageDataset(Dataset):
    """Class-balanced view over a HuggingFace-style Arrow image dataset.

    ImageNet-1k is present here only as an Arrow cache, not as an ImageFolder
    tree, but CASTER needs labelled clean source data to estimate class
    statistics. Estimating 1000 class means and a shared covariance does not
    require all 1.28M images, so this samples a fixed number per class and stops
    reading shards once every class is satisfied.

    Label convention: HuggingFace ``imagenet-1k`` indexes classes by sorted WNID,
    which is exactly the order ``torchvision.datasets.ImageFolder`` produces on
    the ImageNet-C directory tree. The two therefore align without remapping;
    this was verified over 400 samples before the loader was written, because a
    silent mismatch would corrupt every ImageNet result while looking plausible.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        num_classes: int,
        per_class: int = 50,
        transform: transforms.Compose | None = None,
        pattern: str = "*train-*.arrow",
    ) -> None:
        import glob
        import io

        import pyarrow as pa
        import pyarrow.ipc

        self._io = io
        self.transform = transform
        shards = sorted(glob.glob(str(Path(root) / "**" / pattern), recursive=True))
        if not shards:
            raise RuntimeError(f"no Arrow shards matching {pattern!r} under {root}")

        counts: dict[int, int] = {}
        self.samples: list[tuple[bytes, int]] = []
        for shard in shards:
            if len(counts) == num_classes and all(
                counts.get(c, 0) >= per_class for c in range(num_classes)
            ):
                break
            with pa.memory_map(shard, "rb") as src:
                try:
                    table = pa.ipc.open_stream(src).read_all()
                except Exception:
                    src.seek(0)
                    table = pa.ipc.open_file(src).read_all()
                labels = table.column("label").to_pylist()
                images = table.column("image").to_pylist()
            for image, label in zip(images, labels):
                if label is None or not (0 <= int(label) < num_classes):
                    continue
                label = int(label)
                if counts.get(label, 0) >= per_class:
                    continue
                payload = image.get("bytes") if isinstance(image, dict) else None
                if not payload:
                    continue
                self.samples.append((payload, label))
                counts[label] = counts.get(label, 0) + 1

        if not self.samples:
            raise RuntimeError(f"no usable samples read from {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        payload, label = self.samples[index]
        image = Image.open(self._io.BytesIO(payload)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CIFARCDataset(Dataset):
    """Loader for CIFAR-10-C / CIFAR-100-C npy files."""

    def __init__(
        self,
        root: str | Path,
        *,
        corruption: str,
        severity: int,
        transform: transforms.Compose | None = None,
    ) -> None:
        if severity < 1 or severity > 5:
            raise ValueError("CIFAR-C severity must be in [1, 5]")
        root = Path(root)
        data_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"
        if not data_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"missing CIFAR-C files under {root}")
        data = np.load(data_path)
        labels = np.load(labels_path)
        start = (severity - 1) * 10_000
        stop = severity * 10_000
        self.data = data[start:stop]
        self.labels = labels[start:stop].astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class SyntheticCorruptionDataset(Dataset):
    """Apply deterministic image corruptions before the normal eval transform."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        corruption: str,
        severity: int,
        transform: transforms.Compose | None = None,
        seed: int = 0,
    ) -> None:
        if severity < 1 or severity > 5:
            raise ValueError("synthetic corruption severity must be in [1, 5]")
        self.dataset = dataset
        self.corruption = corruption
        self.severity = int(severity)
        self.transform = transform
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        image = _to_pil_image(image)
        rng = np.random.default_rng(self.seed + int(index))
        image = _apply_synthetic_corruption(image, self.corruption, self.severity, rng)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        return transforms.ToPILImage()(image.detach().cpu()).convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _apply_synthetic_corruption(
    image: Image.Image,
    corruption: str,
    severity: int,
    rng: np.random.Generator,
) -> Image.Image:
    corruption = corruption.lower()
    image = image.convert("RGB")
    if corruption == "clean":
        return image
    if corruption == "gaussian_noise":
        sigmas = [8.0, 16.0, 28.0, 40.0, 55.0]
        array = np.asarray(image).astype(np.float32)
        array = array + rng.normal(0.0, sigmas[severity - 1], size=array.shape)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if corruption == "shot_noise":
        rates = [80.0, 40.0, 20.0, 10.0, 5.0]
        array = np.asarray(image).astype(np.float32) / 255.0
        noisy = rng.poisson(np.clip(array, 0.0, 1.0) * rates[severity - 1]) / rates[severity - 1]
        return Image.fromarray(np.clip(noisy * 255.0, 0, 255).astype(np.uint8))
    if corruption == "motion_blur":
        kernel = [
            0.0,
            0.0,
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            0.0,
            0.0,
            0.0,
        ]
        output = image
        for _ in range(severity):
            output = output.filter(ImageFilter.Kernel((3, 3), kernel, scale=1.0))
        return output
    if corruption == "brightness":
        factors = [1.25, 1.5, 1.85, 2.2, 2.6]
        return ImageEnhance.Brightness(image).enhance(factors[severity - 1])
    if corruption == "contrast":
        factors = [0.75, 0.6, 0.45, 0.3, 0.18]
        return ImageEnhance.Contrast(image).enhance(factors[severity - 1])
    raise ValueError(f"unknown synthetic corruption={corruption!r}")


def build_transform(
    *,
    image_size: int = 224,
    train: bool = False,
    dataset_family: str = "imagenet",
) -> transforms.Compose:
    if dataset_family == "cifar":
        resize: list[Any] = [] if image_size == 32 else [transforms.Resize(image_size)]
        if train:
            crop: list[Any] = (
                [transforms.RandomCrop(32, padding=4)]
                if image_size == 32
                else [transforms.RandomCrop(image_size, padding=max(4, image_size // 16))]
            )
            return transforms.Compose(
                resize
                + crop
                + [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]
            )
        return transforms.Compose(
            resize
            + [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def build_dataset(
    *,
    name: str,
    root: str | Path,
    split: str = "test",
    transform: transforms.Compose | None = None,
    max_samples: int | None = None,
    **kwargs: Any,
) -> Dataset:
    """Build a dataset by name."""
    name = name.lower()
    root = Path(root)
    if transform is None:
        family = "cifar" if "cifar" in name else "imagenet"
        transform = build_transform(dataset_family=family)
    synthetic_corruption = kwargs.get("synthetic_corruption")
    dataset_transform = None if synthetic_corruption else transform

    if name == "fake":
        size = int(max_samples or kwargs.get("size", 128))
        dataset = datasets.FakeData(
            size=size,
            image_size=(3, int(kwargs.get("image_size", 32)), int(kwargs.get("image_size", 32))),
            num_classes=int(kwargs.get("num_classes", 10)),
            transform=dataset_transform,
            random_offset=int(kwargs.get("random_offset", 0 if split == "train" else 10_000)),
        )
    elif name == "cifar10":
        dataset = datasets.CIFAR10(
            root=str(root),
            train=split == "train",
            transform=dataset_transform,
            download=bool(kwargs.get("download", False)),
        )
    elif name == "cifar100":
        dataset = datasets.CIFAR100(
            root=str(root),
            train=split == "train",
            transform=dataset_transform,
            download=bool(kwargs.get("download", False)),
        )
    elif name == "food101":
        if split not in {"train", "test"}:
            raise ValueError("Food101 split must be one of train, test")
        dataset = datasets.Food101(
            root=str(root),
            split=split,
            transform=dataset_transform,
            download=bool(kwargs.get("download", False)),
        )
    elif name == "dtd":
        if split not in {"train", "val", "test"}:
            raise ValueError("DTD split must be one of train, val, test")
        dataset = datasets.DTD(
            root=str(root),
            split=split,
            partition=int(kwargs.get("partition", 1)),
            transform=dataset_transform,
            download=bool(kwargs.get("download", False)),
        )
    elif name in {"oxford_pets", "oxford-iiit-pet", "oxford_iiit_pet"}:
        if split not in {"trainval", "test"}:
            raise ValueError("Oxford-IIIT Pet split must be one of trainval, test")
        dataset = datasets.OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="category",
            transform=dataset_transform,
            download=bool(kwargs.get("download", False)),
        )
    elif name in {"flowers102", "flowers-102"}:
        if split not in {"train", "val", "test"}:
            raise ValueError("Flowers102 split must be one of train, val, test")
        dataset = datasets.Flowers102(
            root=str(root),
            split=split,
            transform=dataset_transform,
            download=bool(kwargs.get("download", False)),
        )
    elif name in {"imagenet_arrow", "imagenet-arrow"}:
        dataset = ArrowImageDataset(
            root,
            num_classes=int(kwargs.get("num_classes", 1000)),
            per_class=int(kwargs.get("per_class", 50)),
            transform=dataset_transform,
        )
    elif name in {"cifar10c", "cifar100c"}:
        dataset = CIFARCDataset(
            root,
            corruption=str(kwargs.get("corruption", "gaussian_noise")),
            severity=int(kwargs.get("severity", 1)),
            transform=transform,
        )
    elif name in {
        "imagenet",
        "imagenet-c",
        "imagenet-r",
        "imagenet-sketch",
        "imagenet-v2",
        "pacs",
        "office-home",
        "image_folder",
    }:
        path = root / str(kwargs.get("domain", "")) if kwargs.get("domain") else root
        dataset = datasets.ImageFolder(str(path), transform=dataset_transform)
    else:
        raise ValueError(f"unknown dataset name={name!r}")

    if synthetic_corruption:
        dataset = SyntheticCorruptionDataset(
            dataset,
            corruption=str(synthetic_corruption),
            severity=int(kwargs.get("synthetic_severity", kwargs.get("severity", 1))),
            transform=transform,
            seed=int(kwargs.get("synthetic_seed", 0)),
        )

    if max_samples is not None and max_samples < len(dataset):
        return Subset(dataset, list(range(max_samples)))
    return dataset
