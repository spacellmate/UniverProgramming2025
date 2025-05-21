import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Определяем трансформации
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Путь до папки, куда сохраним скачанные изображения
root = "./Data_10"

# Определяем размер батча
batch_size = 64  # Можно изменить размер батча по своему усмотрению

# Загружаем тренировочные данные
train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transformations)
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Загружаем тестовые данные
test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transformations)
test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Выводим информацию о загрузке данных
print("Данные успешно загружены и подготовлены.")
