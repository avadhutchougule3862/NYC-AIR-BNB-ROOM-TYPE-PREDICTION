from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="NYC Airbnb Room Type Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = [
    "latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "neighbourhood_group", "neighbourhood",
]

# Load pre-trained model pipeline
model = joblib.load("Model_Pipeline.pkl")


class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    price: float = Field(..., gt=0, description="Price per night")
    minimum_nights: int = Field(..., ge=1, le=365, description="Minimum nights required")
    number_of_reviews: int = Field(..., ge=0, description="Total number of reviews")
    reviews_per_month: float = Field(..., ge=0, description="Average reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description="Host listing count")
    availability_365: int = Field(..., ge=0, le=365, description="Availability days in a year")
    neighbourhood_group: str = Field(..., min_length=1, description="Borough name")
    neighbourhood: str = Field(..., min_length=1, description="Specific neighbourhood name")


@app.get('/')
def greet():
    return {"message": "API is running successfully!"}


@app.post('/predict')
def predict(features: Features):
    # Use model_dump() instead of dict() for Pydantic V2
    input_data = pd.DataFrame([features.model_dump()], columns=COLUMNS)
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Map class labels to probabilities
    prob_dict = {
        cls: round(float(prob), 4)
        for cls, prob in zip(model.classes_, probabilities)
    }

    return {
        "predicted_room_type": prediction,
        "probabilities": prob_dict
    }