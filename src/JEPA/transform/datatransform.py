from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

train_data = datasets.CIFAR10("data", train=True, transform=transform, download=True)
test_data = datasets.CIFAR10("data", transform=transform, train=False, download=True)

## batchify data

train_loader = DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=32,
    shuffle=False
)