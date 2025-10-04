import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Генериране на синтетични данни
def generate_data():
    """Генерира прости данни за линейна регресия"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Създаване на X данни (входни данни)
    X = torch.linspace(0, 10, 100).reshape(-1, 1)
    
    # Създаване на Y данни (целеви стойности) с шум
    # Y = 2*X + 3 + шум
    true_slope = 2.0
    true_intercept = 3.0
    noise = torch.randn(100, 1) * 0.5
    y = true_slope * X + true_intercept + noise
    
    return X, y, true_slope, true_intercept

# Дефиниране на модела
class SimpleLinearModel(nn.Module):
    """Много прост линеен модел"""
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        # Един линеен слой: y = wx + b
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_model():
    """Обучаване на модела"""
    print("🚀 Започваме обучението...")
    
    # Генериране на данни
    X, y, true_slope, true_intercept = generate_data()
    
    # Създаване на модела
    model = SimpleLinearModel()
    
    # Дефиниране на функцията за загуба и оптимизатора
    criterion = nn.MSELoss()  # Средна квадратна грешка
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Стохастичен градиентен спуск
    
    # Списък за съхранение на загубите
    losses = []
    
    # Обучение
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Предикция
        predictions = model(X)
        
        # Изчисляване на загубата
        loss = criterion(predictions, y)
        
        # Обратно разпространение
        optimizer.zero_grad()  # Нулиране на градиентите
        loss.backward()        # Изчисляване на градиентите
        optimizer.step()       # Обновяване на параметрите
        
        # Запазване на загубата
        losses.append(loss.item())
        
        # Показване на прогреса всеки 100 епохи
        if (epoch + 1) % 100 == 0:
            print(f"Епоха {epoch + 1}/{num_epochs}, Загуба: {loss.item():.4f}")
    
    return model, X, y, losses, true_slope, true_intercept

def visualize_results(model, X, y, losses, true_slope, true_intercept):
    """Визуализация на резултатите"""
    # Получаване на обучените параметри
    learned_weight = model.linear.weight.item()
    learned_bias = model.linear.bias.item()
    
    print(f"\n📊 Резултати:")
    print(f"Истински параметри: наклон = {true_slope:.2f}, отсечка = {true_intercept:.2f}")
    print(f"Научени параметри: наклон = {learned_weight:.2f}, отсечка = {learned_bias:.2f}")
    
    # Създаване на графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Графика 1: Данни и предсказания
    ax1.scatter(X.numpy(), y.numpy(), alpha=0.6, label='Данни', color='blue')
    
    # Предсказания на модела
    with torch.no_grad():
        predictions = model(X)
        ax1.plot(X.numpy(), predictions.numpy(), 'r-', linewidth=2, label='Модел')
    
    # Истинската линия
    true_line = true_slope * X.numpy() + true_intercept
    ax1.plot(X.numpy(), true_line, 'g--', linewidth=2, label='Истинска линия')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Линейна регресия')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Графика 2: Загубата през времето
    ax2.plot(losses)
    ax2.set_xlabel('Епоха')
    ax2.set_ylabel('Загуба (MSE)')
    ax2.set_title('Загубата през обучението')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Графиките са запазени като 'linear_regression_results.png'")

def main():
    """Главна функция"""
    print("🎯 Добре дошли в първия PyTorch проект!")
    print("=" * 50)
    
    # Проверка дали PyTorch работи
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA налично: {torch.cuda.is_available()}")
    
    # Обучение на модела
    model, X, y, losses, true_slope, true_intercept = train_model()
    
    # Визуализация на резултатите
    visualize_results(model, X, y, losses, true_slope, true_intercept)
    
    print("\n✅ Проектът завърши успешно!")
    print("Това беше елементарен пример за линейна регресия с PyTorch.")

if __name__ == "__main__":
    main()
