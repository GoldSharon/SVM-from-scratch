import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) classifier.

    Attributes:
        learning_rate (float): The learning rate for weight updates.
        num_of_iter (int): The number of iterations for training.
        lambda_parameter (float): The regularization parameter.
        weights (ndarray): The weights of the SVM model.
        bias (float): The bias term of the SVM model.
        rows (int): The number of samples in the training data.
        cols (int): The number of features in the training data.
    """

    def __init__(self, learning_rate=0.01, num_of_iter=1000, lambda_parameter=0.01):
        """
        Initializes the SVM with the given parameters.

        Parameters:
            learning_rate (float): The learning rate for the optimizer.
            num_of_iter (int): The number of iterations for training.
            lambda_parameter (float): The regularization strength.
        """
        self.learning_rate = learning_rate
        self.num_of_iter = num_of_iter
        self.lambda_parameter = lambda_parameter
        self.weights = None
        self.bias = None
        self.rows = None
        self.cols = None

    def fit(self, X, y):
        """
        Fit the SVM model to the training data.

        Parameters:
            X (ndarray): Training data of shape (num_samples, num_features).
            y (ndarray): Target labels of shape (num_samples,), where labels should be 0 or 1.
        """
        self.rows, self.cols = X.shape
        self.weights = np.zeros(self.cols)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero

        # Convert labels to -1 and 1
        y_mod = np.where(y == 0, -1, 1)

        # Training loop
        for i in range(self.num_of_iter):
            for index in range(self.rows):
                # Compute the decision value
                decision_value = y_mod[index] * (self.weights @ X[index] - self.bias)

                # Update weights and bias based on the decision value
                if decision_value >= 1:
                    dw = 2 * self.lambda_parameter * self.weights  # Regularization term
                    db = 0
                else:
                    dw = 2 * self.lambda_parameter * self.weights - X[index] * y_mod[index]
                    db = y_mod[index]

                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
            X (ndarray): Input data of shape (num_samples, num_features).

        Returns:
            ndarray: Predicted labels (0 or 1) for each input sample.
        """
        # Compute the output
        output = X @ self.weights + self.bias
        
        # Apply the sign function to get predicted labels
        predicted_label = np.sign(output)
        
        # Convert predicted labels from -1 and 1 to 0 and 1
        y_pred = np.where(predicted_label <= -1, 1, 0)
        
        return y_pred
