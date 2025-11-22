"""
Data utilities for loading datasets and preparing inputs.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Callable, Union


def get_default_transforms(
    image_size: int = 224,
    normalize: bool = True
) -> transforms.Compose:
    """
    Get default preprocessing transforms for ImageNet-style models.
    
    Args:
        image_size: Target image size
        normalize: Whether to normalize with ImageNet stats
        
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list)


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Denormalize a normalized tensor.
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    return tensor * std + mean


def load_image(
    image_path: str,
    transform: Optional[Callable] = None,
    image_size: int = 224
) -> torch.Tensor:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        transform: Optional custom transform
        image_size: Target size if using default transform
        
    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    if transform is None:
        transform = get_default_transforms(image_size)
    
    return transform(image)


def get_medical_dataset(
    dataset_name: str,
    root: str = './data',
    split: str = 'train',
    transform: Optional[Callable] = None,
    download: bool = True
) -> Dataset:
    """
    Load a medical imaging dataset.
    
    Supports:
    - ISIC (via custom loader)
    - MedMNIST datasets
    - ChestX-ray datasets (via torchxrayvision)
    
    Args:
        dataset_name: Name of the dataset
        root: Root directory for data
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transforms
        download: Whether to download if not present
        
    Returns:
        PyTorch Dataset
    """
    dataset_name = dataset_name.lower()
    
    # MedMNIST datasets
    if 'mnist' in dataset_name:
        try:
            import medmnist
            from medmnist import INFO
            
            info = INFO[dataset_name]
            DataClass = getattr(medmnist, info['python_class'])
            
            dataset = DataClass(
                split=split,
                transform=transform or get_default_transforms(),
                download=download,
                root=root
            )
            
            return dataset
        except ImportError:
            raise ImportError("Please install medmnist: pip install medmnist")
    
    # ChestX-ray datasets
    elif 'chest' in dataset_name or 'xray' in dataset_name:
        try:
            import torchxrayvision as xrv
            
            if transform is None:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
            
            if 'nih' in dataset_name:
                dataset = xrv.datasets.NIH_Dataset(
                    imgpath=f"{root}/images",
                    csvpath=f"{root}/Data_Entry_2017.csv",
                    transform=transform
                )
            elif 'chexpert' in dataset_name:
                dataset = xrv.datasets.CheX_Dataset(
                    imgpath=f"{root}/CheXpert-v1.0-small",
                    csvpath=f"{root}/CheXpert-v1.0-small/train.csv",
                    transform=transform
                )
            else:
                raise ValueError(f"Unknown chest x-ray dataset: {dataset_name}")
            
            return dataset
        except ImportError:
            raise ImportError("Please install torchxrayvision: pip install torchxrayvision")
    
    else:
        raise ValueError(
            f"Dataset {dataset_name} not supported. "
            "Supported: MedMNIST datasets, ChestX-ray datasets"
        )


class SimpleImageDataset(Dataset):
    """
    Simple dataset for a list of image paths.
    """
    
    def __init__(
        self,
        image_paths: list,
        labels: Optional[list] = None,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            image_paths: List of image file paths
            labels: Optional list of labels
            transform: Optional transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or get_default_transforms()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        
        return image


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def prepare_input(
    input_data: Union[str, Image.Image, torch.Tensor, np.ndarray],
    transform: Optional[Callable] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Prepare various input types for model inference.
    
    Args:
        input_data: Image path, PIL Image, tensor, or numpy array
        transform: Optional transform
        device: Device to move tensor to
        
    Returns:
        Preprocessed tensor ready for model
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if transform is None:
        transform = get_default_transforms()
    
    # Handle different input types
    if isinstance(input_data, str):
        # File path
        image = Image.open(input_data).convert('RGB')
        tensor = transform(image)
    elif isinstance(input_data, Image.Image):
        # PIL Image
        tensor = transform(input_data)
    elif isinstance(input_data, np.ndarray):
        # Numpy array
        image = Image.fromarray(input_data.astype('uint8'))
        tensor = transform(image)
    elif isinstance(input_data, torch.Tensor):
        # Already a tensor
        tensor = input_data
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    # Add batch dimension if needed
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor.to(device)
