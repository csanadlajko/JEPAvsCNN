import torch.nn as nn
import torch

class LinearProbe(nn.Module):

    def __init__(self, model, embed_dim, class_dim, prediction_head, data_loader):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(embed_dim, class_dim)
        self.prediction_head = prediction_head
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=0.003)

        # Freezing the JEPA model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = self.prediction_head(x)
        logits = self.classifier(x)
        return logits
    
    def train(self, x):
        self.model.train()
        for epoch in range(50):
            for img, label in self.data_loader:
                self.optim.zero_grad()
                target_encoding = self.model(img)
                loss = self.criterion(target_encoding, label)
                pass # TODO finish