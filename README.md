# SVM from Scratch with CuPy GPU Support

### Overview
This project is a custom implementation of a Support Vector Machine (SVM) classifier, built from scratch in Python. It leverages [CuPy](https://cupy.dev/) to support GPU acceleration, making it suitable for handling larger datasets efficiently by utilizing NVIDIA CUDA-capable GPUs.

### Features
- **Kernel Functions**: Supports multiple kernel functions, including linear, polynomial, radial basis function (RBF), and sigmoid.
- **Hinge Loss Calculation**: Implements hinge loss for the SVM optimization objective.
- **Stochastic Gradient Descent**: Uses SGD to optimize the dual form of the SVM.
- **CuPy GPU Acceleration**: Supports CUDA-based operations via CuPy for enhanced performance.

### Requirements
Ensure you have Python 3.x installed and a CUDA-capable NVIDIA GPU for GPU acceleration. Install the required libraries as follows:

```bash
pip install cupy
```

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/GoldSharon/SVM-from-scratch.git
   cd SVM-from-scratch
   ```
2. Install the dependencies listed in the `requirements.txt` file (optional).

### Usage
To use this SVM model, import the `SVM` class and fit it to your data as shown below.

```python
import cupy as cp
from svm import SVM  # Replace with your filename if different

# Load your data (example with random data)
X = cp.random.rand(100, 2)  # 100 samples, 2 features
y = cp.random.randint(0, 2, 100)  # Binary labels (0 or 1)

# Initialize and train the model
model = SVM(learning_rate=0.01, num_of_iter=1000, lambda_parameter=0.01, kernel='linear')
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(predictions)
```

### Project Structure
- `svm.py`: Contains the SVM class with kernel functions and training methods.
- `README.md`: Overview of the project, including setup and usage instructions.

### How It Works
The SVM model in this project is trained using stochastic gradient descent on the dual form of the SVM problem. The model supports different kernel functions, allowing it to classify both linearly and non-linearly separable data.

1. **Hinge Loss**: Minimizes the hinge loss function, which allows the model to focus on maximizing the margin while penalizing misclassifications.
2. **Kernel Trick**: Allows mapping input data into higher-dimensional spaces for better separability.
3. **CuPy Acceleration**: CuPy is used to handle matrix operations on the GPU, enabling faster computation on large datasets.

### Customizing the Kernel
The model includes several kernel functions that you can choose from:
- **Linear Kernel**: Basic linear separability.
- **Polynomial Kernel**: Adds non-linearity with a configurable degree.
- **RBF (Gaussian) Kernel**: Handles non-linear boundaries with a configurable gamma.
- **Sigmoid Kernel**: Adds non-linear decision boundaries with a configurable gamma and r.

To use a different kernel, specify it during initialization:
```python
model = SVM(kernel='rbf')
```

### License
This project is licensed under the MIT License.

---
