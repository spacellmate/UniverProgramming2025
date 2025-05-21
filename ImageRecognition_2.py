import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Определение модели
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.fc = nn.Linear(24*16*16, 10)  # Изменено на 24*16*16

    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        print(out.shape)  # Отладочный вывод
        out = self.pool(out)
        print(out.shape)  # Отладочный вывод
        out = F.relu(self.bn2(self.conv2(out)))
        print(out.shape)  # Отладочный вывод
        out = out.view(out.size(0), -1)  # Преобразуем в одномерный вектор
        print(out.shape)  # Отладочный вывод
        return self.fc(out)

# Определяем трансформации
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Путь до папки, куда сохраним скачанные изображения
root = "./Data_10"

# Определяем размер батча
batch_size = 64

# Загружаем тренировочные данные
train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transformations)
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Загружаем тестовые данные
test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transformations)
test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Инициализация модели, функции потерь и оптимизатора
model = ImageModel()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Функция для тестирования точности
def test_accuracy(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Функция для отображения изображений
def show_images(images, labels, predictions):
    plt.figure(figsize=(10, 10))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f'True: {labels[i]}, Pred: {predictions[i]}')
        plt.axis('off')
    plt.show()

# Получаем 20 изображений из тестового набора
images, labels = next(iter(test_data_loader))
images = images[:20]
labels = labels[:20]

# Получаем предсказания модели
with torch.no_grad():
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

# Выводим изображения и предсказания
show_images(images, labels, predictions)

# Рассчитываем точность
accuracy = (predictions == labels).sum().item() / len(labels) * 100
print(f'Accuracy: {accuracy}%')
