import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Генерация данных
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Конвертация в тензоры PyTorch
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)  # reshape для совместимости
y_test = torch.FloatTensor(y_test).view(-1, 1)


# Определение модели
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)  # вход 5, скрытый слой 64
        self.fc2 = nn.Linear(64, 32)  # скрытый слой 32
        self.fc3 = nn.Linear(32, 16)  # скрытый слой 16
        self.fc4 = nn.Linear(16, 1)  # выход 1 (бинарная классификация)

        # Разные функции активации для каждого слоя
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
        self.act3 = nn.LeakyReLU(0.01)  # LeakyReLU с отрицательным slope=0.01
        self.output = nn.Sigmoid()  # сигмоида на выходе

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.output(self.fc4(x))
        return x


# Функция для обучения и оценки модели
def train_model(optimizer_name, learning_rate=0.01, epochs=50):
    model = NeuralNet()
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer")

    train_losses, val_losses = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        # Обучение
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Оценка на тренировочных данных
        train_preds = (outputs > 0.5).float()
        train_accuracy = (train_preds == y_train).float().mean()

        # Валидация
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            val_preds = (val_outputs > 0.5).float()
            val_accuracy = (val_preds == y_test).float().mean()

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_acc.append(train_accuracy.item())
        val_acc.append(val_accuracy.item())

    return train_losses, val_losses, train_acc, val_acc


# Сравнение оптимизаторов
adam_train_loss, adam_val_loss, adam_train_acc, adam_val_acc = train_model("Adam")
sgd_train_loss, sgd_val_loss, sgd_train_acc, sgd_val_acc = train_model("SGD")

# Визуализация
plt.figure(figsize=(12, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(adam_train_acc, label='Adam train', linestyle='--')
plt.plot(adam_val_acc, label='Adam val')
plt.plot(sgd_train_acc, label='SGD train', linestyle='--')
plt.plot(sgd_val_acc, label='SGD val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(adam_train_loss, label='Adam train', linestyle='--')
plt.plot(adam_val_loss, label='Adam val')
plt.plot(sgd_train_loss, label='SGD train', linestyle='--')
plt.plot(sgd_val_loss, label='SGD val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()