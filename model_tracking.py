import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import mlflow
from joblib import dump

# Load the data
df = pd.read_csv('output.csv')

# Splitting the data into train and validation sets
train, val = train_test_split(df, test_size=0.15, random_state=42)

# Separating the target variable and predictors
X_train = train.drop('Price', axis=1)
print(f"The training columns are: {X_train.columns}")
y_train = train['Price']
X_val = val.drop('Price', axis=1)
y_val = val['Price']

# List of models to train
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(n_estimators=1000),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=1000)
}

# Variable to store the best model and its performance
best_model = None
lowest_rmse = float('inf')

# Start MLflow experiment
mlflow.set_experiment('Car Price Prediction Models')

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on validation set
        predictions = model.predict(X_val)
        
        # Calculate metrics
        rmse = root_mean_squared_error(y_val, predictions)
        
        # Check if this model has the lowest RMSE
        if rmse < lowest_rmse:
            best_model = model
            best_model_name = name
            lowest_rmse = rmse
        
        # Log ALL model parameters
        params = model.get_params()
        mlflow.log_params(params)
        
        # Log the mean squared error
        mlflow.log_metric('RMSE', rmse)
        
        # Log model
        mlflow.sklearn.log_model(model, f"model_{name}")
        
        # Print out the model and its MSE for easy viewing
        print(f'Model: {name}')
        print(f'RMSE: {rmse}\n')

mlflow.end_run()

# Retrain the best model on the entire dataset
X = df.drop('Price', axis=1)
y = df['Price']
best_model.fit(X, y)

# Save the best model to a local file
model_path = "final_model.pkl"
dump(best_model, model_path)

print(f"Best Model: {best_model_name} with RMSE: {lowest_rmse}")
print(f"Model saved to {model_path}")