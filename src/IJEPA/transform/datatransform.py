from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json

file = open("././parameters.json")
parameters: dict[str, int] = json.load(file)

transform = transforms.Compose([
    transforms.Resize((parameters["IMAGE_SIZE"], parameters["IMAGE_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

train_data = datasets.CIFAR10("data", train=True, transform=transform, download=True)
test_data = datasets.CIFAR10("data", transform=transform, train=False, download=True)

## batchify data

train_loader = DataLoader(
    dataset=train_data,
    batch_size=parameters["BATCH_SIZE"],
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=parameters["BATCH_SIZE"],
    shuffle=False
)