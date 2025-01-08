import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def plot_correlation():
    train_data = pd.read_csv('data/development.csv')
    target_column = train_data.columns[2]
    for feature in train_data.columns:
        if feature != target_column:  
            plt.scatter(train_data[feature], train_data['age'])
            plt.xlabel(feature)
            plt.ylabel('age')
            plt.show()


def calculate_correlation():
    train_data = pd.read_csv('data/development.csv', na_values=['ND']).dropna()
    
    # Calculate the correlation coefficient
    corr_coef = train_data.select_dtypes(include=['int64', 'float64']).corrwith(train_data['age'])

    # Print the correlation coefficients
    print(corr_coef)


def lasso_feature_selection():
    train_data = pd.read_csv('data/development.csv')

    # Split the data into features (X) and target (y)
    X = train_data.drop('age', axis=1).select_dtypes(include=['int64', 'float64'])
    y = train_data['age']

    # Split the data into training and testing sets
    X_train, y_train, = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create a Lasso object with alpha=0.5
    lasso = Lasso(alpha=0.5)

    # Fit the Lasso model to the training data
    lasso.fit(X_train_scaled, y_train)

    # Get the coefficients
    coeffs = lasso.coef_

    # Select the features with non-zero coefficients
    selected_features = [i for i, coeff in enumerate(coeffs) if coeff != 0]

    # Get the feature names
    feature_names = X.columns

    # Print the selected features
    print("Selected features:")
    for i in selected_features:
        print(feature_names[i])