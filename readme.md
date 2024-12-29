# TensorFlow Deep Dive: Theory and Implementation

Welcome to the **TensorFlow Deep Dive** repository, a collection of theoretical and practical notebooks that cover fundamental and advanced TensorFlow concepts to solve Machine Learning problems. This repository is organized to facilitate progressive learning and the application of advanced techniques in various domains.

---

## Repository Structure

The repository is organized as follows:

```
Theory_notebooks/
└── Extras/
└── Sides/
└── All notebooks with the code
└── helper_functions.py
└── requiremets.txt
```

- **Extras**: Additional files that complement the main content.
- **Sides**: Auxiliary resources and explanations.
- **All notebooks with the code**: Notebooks with the main content.
- **Helper_functions**: Utility functions used in the notebooks.
- **requirements.txt**: Requierements needed to run the code in your machine.

---

## Notebook Contents

### **1. TensorFlow Fundamentals**
- Introduction to tensors.
- Using GPU to accelerate processes.
- Using the `@tf.function` decorator.
- Differences between eager and graph execution.

### **2. Regression with TensorFlow using Dense Layers**
- Modifying shapes to input data into models.
- Steps to create an ANN.
- Saving the model.
- Calculating total and trainable parameters of a model.
- Detailed explanations of:
  - Backpropagation.
  - Dropout.
  - Vanishing gradient problem.
- Visualizing regression results using Matplotlib.
- Implementing custom loss functions.

### **3. Classification with TensorFlow using Dense Layers**
- Architecture of classification problems.
- Explanation of loss functions and optimizers.
- Key evaluation metrics for classification models.
- Using confusion matrices and ROC curves for model evaluation.

### **4. Introduction to Convolutional Neural Networks (CNNs)**
- Fundamentals and architecture of CNNs.
- Binary and multiclass classification models for images.
- Data augmentation techniques to improve model generalization.
- Visualizing feature maps from CNN layers.

### **5-7. Transfer Learning and Fine Tuning**
- Using pre-trained models like EfficientNet and ResNet.
- Feature extraction and fine-tuning.
- Model comparison with TensorBoard.
- Accelerating data processing with prefetching.
- Comprehensive exercises and practical cases.
- Tips for selecting pre-trained models for specific tasks.

### **8-9. Introduction to NLP with TensorFlow**
- Tokenization and embedding processes for text data.
- Detailed explanation of sequential problems and recurrent neural networks (RNN):
  - LSTM.
  - CONV1D.
  - GRU.
  - Bidirectional RNN.
- Project applying NLP techniques.
- Using pre-trained word embeddings like GloVe and Word2Vec.

### **10. Time Series Forecasting**
- Using CNNs, RNNs, and ensemble models for forecasting.
- Key uncertainties in time series problems.
- Introduction to N-BEATS for state-of-the-art representation.
- Model training best practices and limitations.
- Evaluating forecasting models using MAE, MASE, MAPE and other metrics.

---

## Requirements

- Python 3.12+
- TensorFlow 2.18
- Jupyter Notebook
- Additional libraries specified in `requirements.txt`

Note: If you are using Windows, you need to enable the Windows Subsystem for Linux (WSL) to run the code in this repository. This is because the versions of TensorFlow-GPU, CUDA, and cuDNN used are not compatible with native Windows environments.

---

## Installation

1. Clone or download the repository:
   ```bash
   git clone git@github.com:Garcialejan/Tensorflow_deep_learning_theory.git
   cd Tensorflow_deep_learning_theory
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks:
   ```bash
   jupyter notebook
   ```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

