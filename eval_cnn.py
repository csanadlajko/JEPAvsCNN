import torch.nn as nn
import torch
from src.IJEPA.transform.datatransform import train_loader, test_loader

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.cnnmodel = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnnmodel(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LeNet().to(device)

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    loss = 0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        y_pred = model.forward(imgs)

        loss = criterion(y_pred, labels)
        loss.backward()
        optim.step()
        optim.zero_grad()

    if batch_idx == 500:
        break

    print(f"LeNet CNN loss at epoch {epoch} is: {loss}")

def eval_lenet(train_data, model):
    total, correct = 0, 0
    for batch_idx, (imgs, labels) in enumerate(train_data):
        y_pred = model.forward(imgs)
        _, predicted = torch.max(y_pred, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        if batch_idx == 200:
            break
    
    return 100 * correct, total

acc = eval_lenet(test_loader, model)

print(f"LeNet final accuracy: {acc:.4f}%")