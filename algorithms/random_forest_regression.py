import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from .preprocessing import z_score_normalization


def random_forest_regression(data):

    # Split the data into training and testing sets
    X = data[['hnr', 'num_pauses', 'silence_duration', 'mean_pitch', 'num_words','Id']]
    y = data['age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    X_train_scaled = z_score_normalization(X_train)
    X_test_scaled = z_score_normalization(X_test)

    # Create a random forest regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = rf.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse:.2f}')

    # Use the model to make predictions on the evaluation set
    evaluation_df = pd.read_csv('data/evaluation.csv')[['Id','hnr', 'num_pauses', 'silence_duration', 'mean_pitch', 'num_words']]
    evaluation_X = evaluation_df
    evaluation_X_scaled = z_score_normalization(evaluation_X)
    evaluation_pred = rf.predict(evaluation_X_scaled)

    # Save the predictions to a CSV file
    submission_df = pd.DataFrame({'Id': evaluation_df['Id'], 'Predicted': evaluation_pred})
    print(submission_df)