import torch
from torch import nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 30),
            nn.ReLU(),
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model=torch.load('FirstmodelMNIST.pth', weights_only=False)

device=torch.device("cuda:0")
model.to(device)

transform = transforms.ToTensor()
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

image, label = mnist_test[0]

print("Image shape:", image.shape) 
print("Label:", label)




plt.imshow(image[0],cmap='gray')
plt.title("number")
plt.show()

data=image.reshape(1,784)

data=data.to(device)



with torch.no_grad():
    logits =model(data)
    pred=logits.argmax(1)

print(pred.item())
