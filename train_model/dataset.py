import torch
from torchvision import datasets, transforms
from timm.data import create_transform
import config

class RiceLeafDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_transforms(is_training=True):
    if is_training:
        if hasattr(config, 'USE_TIMM_AUGMENTATION') and config.USE_TIMM_AUGMENTATION:
            return create_transform(
                input_size=config.IMG_SIZE[0],
                is_training=True,
                auto_augment='rand-m9-mstd0.5-inc1',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
                interpolation='bicubic'
            )
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(config.IMG_SIZE[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE[0] + 32, config.IMG_SIZE[1] + 32)),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def get_dataloaders():
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)

    full_dataset = datasets.ImageFolder(config.DATA_DIR)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_subset, test_subset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataset = RiceLeafDataset(train_subset, transform=train_transform)
    test_dataset = RiceLeafDataset(test_subset, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    return train_loader, test_loader
