# Multilayer Perceptron (MLP) with PyTorch
This project involves implementing a Multilayer Perceptron (MLP) using the PyTorch library. An MLP is a type of feedforward artificial neural network that consists of at least three layers: an input layer, one or more hidden layers, and an output layer. Except for the input nodes, each node is a neuron that uses a non-linear activation function to transform its input.

## Key Features

### Stopping Criteria
Training the model involves iterative adjustments to the weights based on the input data and the errors produced. Two primary criteria are used to decide when to stop training:
1. **Convergence**: Training stops when the changes in the weights and the resulting changes in error become negligible.
2. **Avoid Overfitting**: Training stops if the validation error increases over a specified number of epochs, which indicates that the model is starting to memorize the training data rather than learning to generalize from it.

### Learning Rate Adjustment
The learning rate determines how much the weights are adjusted with each training step. If the validation error increases, the learning rate is reduced to allow the model to approach the optimal weights with smaller, more precise steps. This helps in preventing the model from overshooting the optimal point during training.

### Hyperparameter Tuning
Hyperparameters are critical in shaping the performance and efficiency of the neural network. The following hyperparameters are tuned during this project:
- **Number of Hidden Layers**: Determines the depth of the network.
- **Number of Neurons per Layer**: Determines the width of the network.
- **Activation Functions**: Non-linear functions applied to the outputs of neurons.
- **Number of Training Epochs**: Total number of passes through the training dataset.
- **Learning Rate and Adjustment Strategy**: Controls the size of steps during optimization.
- **Regularization Coefficient**: Helps in preventing overfitting by penalizing large weights.
- **Batch Size**: Number of training examples used in one iteration.
- **Optimizer**: Algorithm used for updating the weights.
- **Dropout Rate**: Fraction of neurons randomly set to zero during training to prevent overfitting.



## Initial Network Configuration

### Evaluation and Selection
- **Single-layer Network**: Models with a single hidden layer and sigmoid activation function performed poorly.
- **Multi-layer Network**: Networks with 1, 2, and 3 hidden layers were tested. Two hidden layers showed better performance and were selected for further tuning.
- **Neuron Count**: Initially set at 512 neurons per layer, which yielded good results.

### Learning Rate and Regularization
- **Optimal Learning Rates**: Determined to be between 1e-5 and 1e-4, based on experiments with different values.
- **Regularization**: Effective regularization coefficients were found to be 1e-5 and 1e-4.
- **Optimizer Comparison**: Adam optimizer outperformed SGD and RMSprop.

### Batch Size and Normalization
- **Batch Size**: Smaller batch sizes lead to better learning but increase training time.
- **Batch Normalization**: Applied to improve gradient flow and allow for faster convergence.

### Dropout and Activation Functions
- **Dropout**: Rates higher than 0.1 led to underfitting. A dropout rate of 0.1 in the second layer was optimal.
- **Activation Functions**: Leaky ReLU combined with tanh was effective, preventing the dead neuron problem associated with standard ReLU.

  
## Final Model Configuration
- **Architecture**: Two hidden layers with 512 neurons each.
- **Activation Functions**: Leaky ReLU for the first hidden layer and tanh for the second.
- **Optimizer**: Adam with a learning rate of 1e-4 and a regularization coefficient of 1e-5.
- **Batch Size**: 128
- **Dropout Rate**: 0.1 in the second hidden layer.


## Evaluating the Model
After training, the model's performance is evaluated on the test dataset. The test script will load the model, run the data through it, and print the loss and accuracy.


![](https://github.com/SheidaAbedpour/MLP-MNIST-handwritten-digit-Classification/blob/main/pictures/20.PNG)
