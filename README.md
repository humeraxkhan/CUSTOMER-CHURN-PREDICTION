# CUSTOMER-CHURN-PREDICTION
Customer Churn Prediction

This repository contains a Python-based implementation for predicting customer churn using a machine learning model. The project involves data preprocessing, visualization, and the use of a neural network model built with TensorFlow/Keras.
Features

    Data Preprocessing:
        Cleaning and handling missing or inconsistent values.
        Encoding categorical features and scaling numerical columns.
    Visualization:
        Histograms for tenure and monthly charges in relation to churn.
    Modeling:
        Neural network with 2 hidden layers trained to classify customer churn.
        Evaluation using accuracy, confusion matrix, and classification report.

Steps to Run

    Clone this repository:

git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

Install required libraries:

pip install pandas numpy matplotlib seaborn tensorflow scikit-learn

Replace the dataset path in the script:

df = pd.read_csv(r"C:\path\to\customer_churn.csv")

Run the script:

    python CUSTOMER_CHURN_PREDICTIONN.py

Results

    Model Accuracy: 86% on the test set.
    Metrics:
        Confusion matrix and classification report provide insights into the model's performance.

Future Improvements

    Optimize hyperparameters for better performance.
    Explore additional features or external datasets.
    Implement more advanced models for comparison.
