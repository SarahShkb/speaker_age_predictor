import pandas as pd
from algorithms.feature_selection import plot_correlation, calculate_correlation,lasso_feature_selection

def main():
    print("This is the main function")
    train_data = pd.read_csv('data/development.csv', na_values=['ND']).dropna()
    lasso_feature_selection(train_data)

if __name__ == '__main__':
    main()