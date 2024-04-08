import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from datetime import datetime

from src.utils import set_device

class SimpleMNISTClassifier:
    def __init__(self, input_size=784, hidden_size=512, output_size=10, learning_rate=0.01, force_gpu=False):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.device = set_device(force_gpu)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        self.data_loaded = False
        self.model_trained = False
        self.eval_history = {}

    def reset_device(self, machine_type):
        self.device = set_device(machine_type)
        self.model.to(self.device)
        
    def load_data(self, batch_size=64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.data_loaded = True
    
    def train(self, epochs=5):
        if not self.data_loaded:
            raise ValueError("Data is not loaded yet. Please load the data first.")
        
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs.view(-1, 28*28))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
        
        self.model_trained = True
        print('Finished Training')
    
    def evaluate(self):
        if not self.model_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images.view(-1, 28*28))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.eval_history[datetime.now()] = 100 * correct / total
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    def save_model(self, path):
        if not self.model_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        else:
            torch.save(self.model.state_dict(), path)

    def run(self):
        self.load_data()
        self.train()
        self.evaluate()