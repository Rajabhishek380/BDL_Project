from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np

# Define a Pydantic model for the input data
class CarFeatures(BaseModel):
    Transmission: int
    Mileage: int
    Owner_Type: int
    Engine: int
    Power: int
    Car_Age: int

# Load the pre-trained model from a file
def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# Initialize the FastAPI app
app = FastAPI()

# Load the model from the specified path
try:
    model = load_model("final_model.pkl")
except HTTPException as he:
    print(he.detail)

@app.get("/")
def read_root():
    return {"Message": "Welcome to the Car Price Prediction API. Use the /predict endpoint to make predictions."}

@app.post("/predict/")
def predict_car_price(features: CarFeatures):
    try:
        # Convert the input data into a numpy array
        input_data = np.array([[features.Transmission, features.Mileage, features.Owner_Type,
                                features.Engine, features.Power, features.Car_Age]])
        
        # Make a prediction using the model
        prediction = model.predict(input_data)
        return {"predicted_price": prediction[0]}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

# Run the server using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
