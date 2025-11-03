from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
from src.IJEPA.transform.mri_dataprocess import MRIImageDataset
from src.IJEPA.transform.cifar10dot1 import CIFAR10dot1Dataset

file = open("././parameters.json")
all_params: dict[str, int] = json.load(file)

parameters = all_params["ijepa"]
mm_params = all_params["multimodal"]

transform = transforms.Compose([
    transforms.Resize((parameters["IMAGE_SIZE"], parameters["IMAGE_SIZE"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.Resize((parameters["IMAGE_SIZE"], parameters["IMAGE_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_data = datasets.CIFAR10("data", train=True, transform=transform, download=True)
test_data = datasets.CIFAR10("data", transform=test_transform, train=False, download=True)

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

## LOAD MRI DATASET

if mm_params["LOAD_MRI"]:

    mri_dset = MRIImageDataset("MRIDATA", transform=test_transform)

    train_size = int(0.8*len(mri_dset))
    test_size = len(mri_dset) - train_size

    mri_train_dset, mri_test_dset = random_split(mri_dset, [train_size, test_size])

    mri_train_dset.dataset.transform = transform
    mri_test_dset.dataset.transform = test_transform

    mri_train_loader = DataLoader(
        dataset=mri_train_dset,
        batch_size=parameters["BATCH_SIZE"],
        shuffle=True
    )

    mri_test_loader = DataLoader(
        dataset=mri_test_dset,
        batch_size=parameters["BATCH_SIZE"],
        shuffle=False
    )

## LOAD CIFAR10.1 DATASET

if mm_params["LOAD_CIFAR101"]:

    cifar10dot1 = CIFAR10dot1Dataset("CIFAR10dot1/cifar10.1_v6_data.npy", "CIFAR10dot1/cifar10.1_v6_labels.npy", transform=test_transform)

    cifar101_test_loader = DataLoader(
        dataset=cifar10dot1,
        batch_size=parameters["BATCH_SIZE"],
    shuffle=False
)
