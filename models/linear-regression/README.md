# Linear Regression Model 📈

[![CI/CD](https://github.com/mimitheone/PyTorch/workflows/PyTorch%20Linear%20Regression%20CI/CD/badge.svg)](https://github.com/mimitheone/PyTorch/actions)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

A beginner-friendly PyTorch implementation of linear regression - one of the most fundamental machine learning algorithms.

## 🎯 What This Model Does

This model learns the relationship between input data (X) and output data (Y) using the equation: **Y = 2*X + 3 + noise**

The model starts with random parameters and gradually learns the correct slope (2) and intercept (3) through training.

## 🚀 Quick Start

```bash
# Navigate to the model directory
cd models/linear-regression

# Install dependencies
pip install -r requirements.txt

# Run the model
python simple_linear_regression.py
```

## 📊 Expected Results

After training, you should see something like:
```
📊 Results:
True parameters: slope = 2.00, intercept = 3.00
Learned parameters: slope = 2.00, intercept = 3.02
```

The model learns the parameters very accurately!

## 🧠 How It Works

### 1. Data Generation
```python
X = torch.linspace(0, 10, 100).reshape(-1, 1)  # Input data
y = 2*X + 3 + noise                             # Target values with noise
```

### 2. Model Definition
```python
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # y = wx + b
    
    def forward(self, x):
        return self.linear(x)
```

### 3. Training Loop
```python
criterion = nn.MSELoss()                    # Loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Optimizer

for epoch in range(1000):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    optimizer.zero_grad()  # Zero gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update parameters
```

## 📈 Real-World Examples

Linear regression is used in many real-world scenarios:

### 📚 Education
- **X** = hours studied before a test
- **Y** = test score
- 👉 The more you study, the higher your score

### 🏋️ Fitness
- **X** = number of workouts per week
- **Y** = strength (maximum bench press weight)
- 👉 The more you train, the stronger you get

### 🚗 Cars
- **X** = age of the car (years)
- **Y** = market price
- 👉 The older the car, the lower the price

### 🌡️ Weather
- **X** = month of the year (1–12)
- **Y** = average temperature (°C)
- 👉 Temperature changes with seasons

### 💻 Technology
- **X** = years of programming experience
- **Y** = number of solved coding problems
- 👉 More experience → more solved problems

### 💊 Health
- **X** = number of cigarettes per day
- **Y** = lung capacity (%)
- 👉 More smoking → lower lung capacity

## 🔧 Key Concepts

- **Tensors**: Basic data structures in PyTorch
- **Modules**: Building blocks for neural networks
- **Loss Functions**: Measure how well the model predicts
- **Optimizers**: Update model parameters
- **Backpropagation**: Calculate gradients for training

## 📁 Files

- `simple_linear_regression.py` - Main implementation
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## 🎓 Learning Objectives

After working with this model, you'll understand:
- How to create a simple neural network in PyTorch
- The basics of training a model
- How to visualize training results
- The concept of linear relationships in data

## 🔗 Related Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Linear Regression Explained](https://en.wikipedia.org/wiki/Linear_regression)

---

**Note**: This model is designed for educational purposes and is perfect for PyTorch beginners.
