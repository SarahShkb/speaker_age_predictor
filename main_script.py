import pandas as pd
from algorithms.feature_selection import plot_correlation, calculate_correlation,lasso_feature_selection
from algorithms.random_forest_regression import random_forest_regression

def main():
    print("This is the main function")
    train_data = pd.read_csv('data/development.csv', na_values=['ND']).dropna()
    random_forest_regression(train_data)

if __name__ == '__main__':
    main()