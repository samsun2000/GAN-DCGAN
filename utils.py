import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.transforms import InterpolationMode
from PIL import Image

# Directory containing the data.
root = 'D:/DCGAN-framework/emnist_jpeg/Train'

# Data augmentation
enlarged_size = int(1.1 * 32)  # 32 * 1.1 = 35.2, which we can round down to 35
canvas_size = [enlarged_size, enlarged_size]
augmented_transform = transforms.Compose([
    transforms.Resize(canvas_size, InterpolationMode.BICUBIC),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_EMNIST(params):
    # Use the augmented_transform for image loading with data augmentation
    dataset = dset.ImageFolder(root=root, transform=augmented_transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader
