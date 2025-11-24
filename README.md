# Deep Learning

This repository contains a collection of Jupyter Notebooks demonstrating various Deep Learning concepts, algorithms, and architectures using TensorFlow and Keras. The projects range from simple neural networks for binary classification to more complex Convolutional Neural Networks (CNNs) for image classification, Recurrent Neural Networks (RNNs) for sequence data, and Natural Language Processing (NLP) tasks.

## Repository Structure

The repository consists of the following notebooks:

1.  `01_Simple_Neural_Network.ipynb`
2.  `02_Multi_layer_Perceptron.ipynb`
3.  `03_Cnn.ipynb`
4.  `04_Image_Classification_using_CNN.ipynb`
5.  `05_Rnn.ipynb`
6.  `06_PaddyDetection.ipynb`
7.  `07_Simple_text_classification.ipynb`

## Detailed Notebook Descriptions

### 1. Simple Neural Network (`01_Simple_Neural_Network.ipynb`)
*   **Description:** Demonstrates the creation of a basic feedforward neural network for binary classification to predict passenger survival.
*   **Dataset:** Titanic Dataset.
*   **Preprocessing:** Handling missing values, one-hot encoding categorical features, standard scaling of numerical features.
*   **Model Architecture:**
    *   Input Layer: Dense (10 units, ReLU activation)
    *   Output Layer: Dense (1 unit, Sigmoid activation)
*   **Training:** Adam optimizer, Binary Crossentropy loss, 100 epochs.

### 2. Multi-Layer Perceptron (`02_Multi_layer_Perceptron.ipynb`)
*   **Description:** Implements a Multi-Layer Perceptron (MLP) for a regression task to predict tip amounts.
*   **Dataset:** Tips Dataset.
*   **Preprocessing:** One-hot encoding, train-test split.
*   **Model Architecture:**
    *   Input Layer: Dense (64 units, ReLU activation)
    *   Hidden Layer: Dense (32 units, ReLU activation)
    *   Output Layer: Dense (1 unit, Linear activation)
*   **Key Concepts:** Regression with Neural Networks, Early Stopping callback to prevent overfitting.

### 3. Introduction to CNN (`03_Cnn.ipynb`)
*   **Description:** Provides an introduction to Convolutional Neural Networks (CNNs) and applies them to handwritten digit classification.
*   **Dataset:** MNIST Handwritten Digits.
*   **Model Architecture:**
    *   Conv2D (10 filters, 3x3 kernel, ReLU)
    *   Conv2D (10 filters, 3x3 kernel, ReLU)
    *   MaxPooling2D
    *   Conv2D (10 filters, 3x3 kernel, ReLU)
    *   Conv2D (10 filters, 3x3 kernel, ReLU)
    *   MaxPooling2D
    *   Flatten
    *   Dense (100 units, Softmax activation)
*   **Key Concepts:** Convolutional layers, Pooling layers, Flattening.

### 4. Image Classification using CNN (`04_Image_Classification_using_CNN.ipynb`)
*   **Description:** Builds a CNN model to classify fashion items.
*   **Dataset:** Fashion MNIST.
*   **Preprocessing:** Pixel normalization (scaling to [0, 1]), reshaping input data.
*   **Model Architecture:**
    *   Conv2D (32 filters, 3x3 kernel, ReLU)
    *   MaxPooling2D
    *   Conv2D (64 filters, 3x3 kernel, ReLU)
    *   MaxPooling2D
    *   Flatten
    *   Dense (10 units, Softmax activation)
*   **Results:** Evaluates model accuracy on test data.

### 5. Recurrent Neural Networks (`05_Rnn.ipynb`)
*   **Description:** Explores Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks for processing sequential data.
*   **Dataset:** Synthetic data (for binary classification) and Flights dataset (for time series forecasting).
*   **Model Architectures:**
    *   **SimpleRNN:** SimpleRNN layer (32 units) -> Dense output.
    *   **LSTM:** LSTM layer (32 units) -> Dense output.
*   **Key Concepts:** Sequence processing, Time series forecasting, Vanishing gradient problem.

### 6. Paddy Disease Detection (`06_PaddyDetection.ipynb`)
*   **Description:** A practical application of Deep Learning to classify paddy leaf diseases into 10 categories.
*   **Dataset:** Paddy Leaf Disease Dataset (10 classes).
*   **Preprocessing:** `image_dataset_from_directory` for efficient data loading, Rescaling layer (1./255).
*   **Model Architecture:**
    *   Rescaling Layer
    *   Conv2D (128 filters) + MaxPooling2D
    *   Conv2D (64 filters) + MaxPooling2D
    *   Conv2D (32 filters) + MaxPooling2D
    *   Conv2D (16 filters) + MaxPooling2D
    *   Flatten
    *   Dropout (0.25)
    *   Dense (128 units, ReLU)
    *   Output Layer: Dense (10 units, Softmax)
*   **Key Concepts:** Real-world image classification, Data pipelines, Regularization with Dropout.

### 7. Simple Text Classification (`07_Simple_text_classification.ipynb`)
*   **Description:** Performs sentiment analysis (positive/negative) on movie reviews using a binary classification model.
*   **Dataset:** IMDB Large Movie Review Dataset.
*   **Preprocessing:** Custom standardization (HTML stripping), `TextVectorization` layer for tokenization and integer encoding.
*   **Model Architecture:**
    *   Embedding Layer (16 dimensions)
    *   Dropout (0.2)
    *   GlobalAveragePooling1D
    *   Dropout (0.2)
    *   Output Layer: Dense (1 unit)
*   **Training:** Binary Crossentropy loss (from logits), Adam optimizer.
*   **Key Concepts:** NLP, Word Embeddings, Text Vectorization, Global Average Pooling for variable-length sequences.

## Getting Started

To run these notebooks, you will need Python installed along with the following libraries:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

You can then launch Jupyter Notebook or JupyterLab to explore and execute the code:

```bash
jupyter notebook
```
