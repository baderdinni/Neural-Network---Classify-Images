# image_classifier.py

This script trains a neural network to classify handwritten digits using the MNIST dataset.

Dependencies

* Python 3.6+
* Pytorch
* Torchvision
* Scikit-learn
* Matplotlib

Usage

1. Clone the repository.
2. Install the dependencies.
3. Run the script: python image_classifier.py

Model

The model is a simple feedforward neural network with 3 layers:
* An input layer with 784 nodes (28*28 pixels).
* A hidden layer with 512 nodes and ReLU activation.
* A hidden layer with 256 nodes and ReLU activation.
* An output layer with 10 nodes (one for each digit).

The model is trained using the Adam optimizer and the cross-entropy loss function.

Data

The MNIST dataset is downloaded automatically by the script.

Performance

The model achieves a test accuracy of over 97%.

Example

The following command will train the model and evaluate it on the test set:

python image_classifier.py


The output of the script will include the test loss, test accuracy, and classification report.

Author

[Vasista]
