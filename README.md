# **Car Price Prediction Using TensorFlow**

This repository contains a Jupyter Notebook that demonstrates the implementation of a TensorFlow model for predicting car prices. The dataset used is a second-hand car dataset from Kaggle, and the project explores essential data preprocessing steps, model building, and performance evaluation.

## **Summary**
This tutorial explores the end-to-end process of predicting car prices using TensorFlow. It covers data preprocessing, model building, and evaluation, providing a comprehensive guide for applying machine learning techniques to real-world data.

### **Highlights**
- **Data Preparation**: Load and preprocess the dataset by selecting relevant features, handling missing values, and normalizing the data to make it suitable for model training.
- **Model Building**: Construct a neural network using TensorFlow, defining layers such as normalization, dense layers, and input layers, and compile the model using appropriate loss functions and optimizers.
- **Training and Evaluation**: Split the dataset into training, validation, and test sets, train the model on the training set, and evaluate its performance on the test set using metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
- **Performance Visualization**: Visualize the model’s predictions and compare them against the actual values to assess its accuracy and identify areas for improvement.

## **Methodology**
### **Data Preprocessing**
- **Import the Dataset**: Load the dataset and convert it into a TensorFlow-compatible format.
- **Shuffle the Data**: Shuffle the data to ensure a good mix of samples during training.
- **Normalize the Features**: Normalize the features to improve the model's performance.
- **Split the Data**: Divide the data into training, validation, and test sets.

### **Model Construction**
- **Define the Neural Network**: Use TensorFlow’s `Sequential` API to define the neural network architecture.
- **Add Layers**: Incorporate layers such as `Normalization`, `Dense`, and `InputLayer`.
- **Compile the Model**: Use the `Adam` optimizer and `MeanSquaredError` loss function to compile the model.

### **Training and Testing**
- **Train the Model**: Train the model using the training set and validate it on the validation set.
- **Evaluate the Model**: Test the model's performance on the test set using MSE and RMSE metrics.

### **Visualization**
- **Plot Predictions**: Compare the model’s predictions against the actual values.
- **Explore Improvements**: Consider tuning hyperparameters or using more complex architectures for better results.

## **Execution**

To execute this project, follow these steps:

1. **Clone the repository:**
   ```bash
   https://github.com/parmarsunny125/Car_price_prediction.git
   cd CarPricePrediction
