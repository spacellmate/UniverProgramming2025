# 1 пункт
# pip install torch torchvision torchaudio
#
# import torch
#
# # Проверка версии PyTorch
# print(torch.__version__)
#
# # Проверка доступности CUDA
# print(torch.cuda.is_available())

#2 пункт
import numpy as np

# Создание двух массивов 3x3 со случайными значениями
array1 = np.random.rand(3, 3)
array2 = np.random.rand(3, 3)

# Вывод массивов
print("Массив 1:")
print(array1)

print("\nМассив 2:")
print(array2)

# 3 пункт
# Сложение массивов
sum_array = array1 + array2

print("\nСумма массивов:")
print(sum_array)

#4 пункт
# Поэлементное умножение массивов
product_array = array1 * array2

print("\nРезультат поэлементного умножения:")
print(product_array)

#5 пункт
# Транспонирование второго массива
transposed_array2 = array2.T

print("\nТранспонированный массив 2:")
print(transposed_array2)

#6 пункт
# Вычисление среднего значения для каждого массива
mean_array1 = array1.mean()
mean_array2 = array2.mean()

# Вывод массивов и их средних значений
print("Среднее значение массива 1:", mean_array1)
print("Среднее значение массива 2:", mean_array2)

#7 пункт
# Вычисление максимального значения для каждого массива
max_array1 = array1.max()
max_array2 = array2.max()

# Вывод массивов и их максимальных значений
print("Максимальное значение массива 1:", max_array1)
print("Максимальное значение массива 2:", max_array2)

#8-13 пункт
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Генерация обучающего набора данных
def generate_data(num_samples):
    np.random.seed(0)
    X = np.random.rand(num_samples, 2) * 10  # num_samples примеров, каждый из двух случайных чисел от 0 до 10
    y = X[:, 0] * X[:, 1]  # Перемножение двух чисел
    return torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)

# Определение архитектуры нейросети
class MultiplicationNet(nn.Module):
    def __init__(self):
        super(MultiplicationNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),  # Полносвязный слой с 2 входами и 16 выходами
            nn.ReLU(),         # Функция активации ReLU
            nn.Linear(16, 1)   # Полносвязный слой с 16 входами и 1 выходом
        )

    def forward(self, x):
        return self.fc(x)

# Функция для обучения модели
def train_model(model, X_tensor, y_tensor, epochs=1000):
    criterion = nn.MSELoss()  # Среднеквадратичная ошибка
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Оптимизатор Adam

    for epoch in range(epochs):
        optimizer.zero_grad()  # Обнуление градиентов
        outputs = model(X_tensor)  # Прямой проход
        loss = criterion(outputs, y_tensor)  # Вычисление потерь
        loss.backward()  # Обратный проход
        optimizer.step()  # Обновление весов

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Генерация тестовых данных
def generate_test_data(num_samples=10):
    np.random.seed(1)
    X_test = np.random.rand(num_samples, 2) * 10
    y_test = X_test[:, 0] * X_test[:, 1]
    return torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1)

# Создание экземпляра модели
model = MultiplicationNet()

# Обучение модели на 10, 100 и 1000 примерах
for num_samples in [10, 100, 1000]:
    print(f"\nОбучение на {num_samples} примерах")
    X_tensor, y_tensor = generate_data(num_samples)
    train_model(model, X_tensor, y_tensor)

    # Проверка модели на тестовых данных
    X_test, y_test = generate_test_data()
    with torch.no_grad():
        test_outputs = model(X_test)
        for i in range(len(X_test)):
            print(f'Вход: {X_test[i].tolist()}, Ожидаемый выход: {y_test[i].item():.4f}, Предсказанный выход: {test_outputs[i].item():.4f}')

# Сохранение модели
torch.save(model.state_dict(), 'multiplication_net.pth')

# Загрузка модели
model.load_state_dict(torch.load('multiplication_net.pth'))

# Проверка загруженной модели на тестовых данных
print("\nПроверка загруженной модели:")
X_test, y_test = generate_test_data()
with torch.no_grad():
    test_outputs = model(X_test)
    for i in range(len(X_test)):
        print(f'Вход: {X_test[i].tolist()}, Ожидаемый выход: {y_test[i].item():.4f}, Предсказанный выход: {test_outputs[i].item():.4f}')
