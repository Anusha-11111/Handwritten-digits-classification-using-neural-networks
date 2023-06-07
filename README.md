# Handwritten-digits-classification-using-neural-networks
The model utilizes a combination of forward propagation and backpropagation algorithms, along with appropriate activation functions and loss functions, to train and optimize the neural network.
The MNIST dataset has been taken from keras consists of a large collection of handwritten digits from 0 to 9. The dataset is split into two main parts: a training set and a test set. The training set contains 60,000 images, while the test set contains 10,000 images. Each image in the dataset is a 28x28 grayscale image, representing a handwritten digit.
Data Preprocessing:
The MNIST dataset consists of grayscale images of handwritten digits. Prior to training the neural network, the pixel values of the images are normalized to a range of 0 to 1 by dividing each pixel value by 255.0. This normalization ensures that the input data is within a consistent range and aids in the convergence of the model.
Neural Network Architecture:
The neural network architecture consists of an input layer with 784 neurons (flattened 28x28 image), a hidden layer with ReLU activation function, and an output layer with a softmax activation function. The model aims to learn the mapping between input images and their corresponding digit labels.
Here we used Relu activation function for hidden layer : The Rectified Linear Unit (ReLU)  is a non-linear function that introduces non-linearity to the neural network, enabling it to learn complex patterns and representations.
The ReLU function is defined as follows:  ReLU(x) = max(0, x)
In other words, for any given input x, the ReLU function returns the maximum value between 0 and x. If x is negative, ReLU outputs 0, otherwise, it outputs x itself. This means that ReLU "activates" or turns on the neurons for positive inputs and keeps them turned off for negative inputs.

The softmax activation function is used particularly for multiclassfication tasks hence I have at the out put layer which efficiently computes during back propagation.
Mathematically, the softmax function is defined as follows for an input vector : softmax(x) = (e^x) / (Σ e^x_i)
In simpler terms, softmax exponentiates each element of the input vector and divides it by the sum of the exponentiated values of all elements. This normalization ensures that the resulting values range between 0 and 1 and sum up to 1, representing a valid probability distribution.

Loss Function and Optimization:
The loss function used in this model is the Categorical Cross-Entropy Loss, which measures the dissimilarity between the predicted class probabilities and the true labels. Backpropagation is employed to compute the gradients of the loss with respect to the model parameters, allowing for parameter updates.
The backpropagation step is automatically handled by the model.fit() method. When you call model.fit(), it performs backpropagation to compute the gradients of the loss function with respect to the network's parameters (weights and biases). Then, it updates these parameters using an optimization algorithm (in this model, the 'adam' optimizer) based on the computed gradients.

Training and Parameter Updates:
The model parameters, including weights and biases, are initialized randomly. The training process involves iteratively propagating the input forward through the network, computing the loss, and then propagating the gradients backward to update the parameters. The update equations for the model parameters involve the learning rate (α) and the gradients computed during backpropagation.
The process of forward propagation, loss calculation, backpropagation, and parameter updates is repeated for each batch of training data in every epoch specified in the model.fit() function.

Evaluation and Discussion: 
           The algorithm used is a feedforward neural network with multiple layers, including an input layer, hidden layers with ReLU activation, and an output layer with sigmoid activation. It employs forward propagation for inference and backpropagation for gradient computation and parameter updates.
Initial settings: The model parameters, including the weights and biases, are randomly initialized using suitable distributions. The specific initialization details are provided by the Keras library.
Parameter updates on epochs: The model uses the Adam optimizer, which adapts the learning rate for each parameter during training. The update equations for the model parameters, such as w1, b1, w2, b2, etc., are automatically handled by the optimizer.
Evaluating the model,the code runs 10 epochs, Difference between the test loss and training loss can be an indication of overfitting or underfitting, while we got low loss values shows  training and testing indicating the best model performance.
Training Loss: 0.015334345400333405 and Test Loss: 0.09627578407526016

And the Training Accuracy: 0.994866669178009(99%) shows the model is trained accurately  and Test Accuracy: 0.9763000011444092(97%)  shows the correctly predicted labels compared to the total number of test samples.

The performance metrics: 
Specificity: Also known as the True Negative Rate, it measures the model's ability to correctly identify negative instances. A specificity of 1.0 means that the model has correctly classified all negative instances.
Sensitivity: Also known as the True Positive Rate or Recall, it measures the model's ability to correctly identify positive instances. A sensitivity of 1.0 means that the model has correctly classified all positive instances.
Precision: It measures the model's ability to correctly classify positive predictions. A precision of 1.0 means that all positive predictions made by the model were correct.
Recall: Another term for sensitivity, it measures the model's ability to correctly identify positive instances.
Accuracy: It measures the overall correctness of the model's predictions by considering both true positive and true negative predictions. An accuracy of 0.9763 means that the model achieved an overall accuracy of 97.63%.
F1-Score: It is a weighted average of precision and recall, providing a balance between the two metrics. An F1-Score of 0.9763 indicates a high balance between precision and recall, implying a good overall performance of the model.
Hence, the model seems to have achieved high accuracy and balanced precision and recall, indicating good performance on the classification task.
The confusion matrix displays the number of samples predicted in each class and revealed insights of good performance where digonal values having high weightage when compared with other values.

In conclusion, the implemented neural network model successfully classified handwritten digits with high accuracy. The combination of appropriate activation functions, loss functions, and optimization techniques enabled the model to learn and generalize from the MNIST dataset. The findings from this project contribute to the field of digit classification and demonstrate the power of neural networks in image recognition tasks.
