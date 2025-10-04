# First PyTorch Project 🚀

This is an elementary project for PyTorch beginners that demonstrates the fundamentals of machine learning through linear regression.

## What is Linear Regression?

Linear regression is one of the simplest and most fundamental machine learning algorithms. It finds the best straight line that fits through your data points. The goal is to predict a continuous value (like price, temperature, or score) based on one or more input features.

### The Math Behind It

Linear regression tries to find the relationship between input (X) and output (Y) using the equation:

**Y = mX + b**

Where:
- **m** (slope) = how much Y changes when X increases by 1
- **b** (intercept) = the value of Y when X = 0

## Real-World Examples

Here are some practical examples where linear regression is commonly used:

### 📚 Education
- **X** = hours studied before a test
- **Y** = test score
- 👉 The more you study, the higher your score. Linear regression might say: "On average, each additional hour of studying adds +0.3 points to the test score."

### 🏋️ Fitness
- **X** = number of workouts per week
- **Y** = strength (maximum bench press weight)
- 👉 The more you train, the stronger you get. The model might say: "Each additional weekly workout adds +5 kg to the maximum bench press."

### 🚗 Cars
- **X** = age of the car (years)
- **Y** = market price (USD/EUR/whatever)
- 👉 The older the car, the lower the price. Linear regression might say: "Each additional year reduces the price by about -1500."

### 🌡️ Weather
- **X** = month of the year (1–12)
- **Y** = average temperature (°C)
- 👉 In some climates, the temperature increases almost linearly from winter to summer. Regression might say: "On average, each month adds +2°C."

### 💻 Technology
- **X** = years of programming experience
- **Y** = number of solved tasks on HackerRank/LeetCode
- 👉 More experience → more solved tasks. The model might say: "Each year of experience adds +50 solved tasks."

### 💊 Health
- **X** = number of cigarettes per day
- **Y** = lung capacity (%)
- 👉 Here the relationship is negative. Linear regression might show: "Each cigarette reduces lung capacity by -0.8%."

## What This Project Does

This project creates and trains a simple linear model for regression. The model learns the relationship between input data (X) and output data (Y) in the format: **Y = 2*X + 3 + noise**

The model starts with random parameters and gradually learns the correct slope (2) and intercept (3) through training.

## Project Structure

```
Pytorch/
├── requirements.txt              # Project dependencies
├── simple_linear_regression.py  # Main code
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## Installation

1. **Install Python** (recommended version 3.8 or newer)

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv pytorch_env
   source pytorch_env/bin/activate  # On macOS/Linux
   # or
   pytorch_env\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project

Simply execute:

```bash
python simple_linear_regression.py
```

## What You'll See

1. **PyTorch Information** - version and CUDA availability
2. **Training Progress** - loss every 100 epochs
3. **Results** - comparison between true and learned parameters
4. **Visualizations**:
   - Data points, model predictions, and true line
   - Loss curve over time
5. **Saved Image** - `linear_regression_results.png`

## Code Explanation

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

## Key Concepts

- **Tensors**: Basic data structures in PyTorch
- **Modules**: Building blocks for neural networks
- **Loss Functions**: Measure how well the model predicts
- **Optimizers**: Update model parameters
- **Backpropagation**: Calculate gradients for training

## Expected Results

After training, you should see something like:
```
📊 Results:
True parameters: slope = 2.00, intercept = 3.00
Learned parameters: slope = 2.00, intercept = 3.02
```

The model learns the parameters very accurately!

## Next Steps

Once you understand this example, you can experiment with:

1. **Different Data** - try various functions and relationships
2. **Parameter Tuning** - different learning rates, number of epochs
3. **More Complex Models** - multiple layers, different activation functions
4. **Real Data** - use actual datasets from the internet

## Useful Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch Book](https://pytorch.org/deep-learning-with-pytorch)

---

**Note**: This project is created for educational purposes and is suitable for complete beginners in PyTorch and machine learning.


