import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, datasets, models

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

learning_rate = 0.0001
batch_size = 128
n_epochs = 5

train_stl10 = torchvision.datasets.STL10(root='./data/STL10', split='train', download=True,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2470, 0.2435, 0.2616))]
                                                                      ))
test_stl10 = torchvision.datasets.STL10(root='./data/STL10', split='test', download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2470, 0.2435, 0.2616))]
                                                                     ))

train_stl10_loader = torch.utils.data.DataLoader(train_stl10, batch_size=batch_size)
test_stl10_loader = torch.utils.data.DataLoader(test_stl10, batch_size=batch_size)

model = models.resnet50(pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
  # Training
  for i, (image, label) in enumerate(train_stl10_loader):
    image, label = image.to(device), label.to(device)
    optimizer.zero_grad()

    output = model(image)
    loss = criterion(output, label)

    loss.backward()
    optimizer.step()

    total = label.size(0)
    _, prediction = torch.max(output.data, 1)
    correct = (prediction == label).sum().item()

    if i % 10 == 0:
      train_accuracy = 100.0 * correct / total
      print("Epoch {} step {}/{} : Loss {}, Accuracy {}".format(epoch, i, len(train_stl10_loader), loss, train_accuracy))

  # Test accuracy
  correct = 0
  total = 0
  with torch.no_grad():
    for i, (image, label) in enumerate(test_stl10_loader):
      image, label = image.to(device), label.to(device)
      output = model(image)

      _, pred = torch.max(output.data, 1)
      correct += (pred == label).sum().item()
      total += len(label)

  test_accuracy = 100.0 * correct / total
  print("Test Accuracy for epoch {}: {}".format(epoch, test_accuracy))
