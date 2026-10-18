from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import joblib

import saas_engine

app = FastAPI(
    title="NYC Airbnb SaaS Intelligence Suite API",
    description="End-to-End SaaS platform for Room Classification, Dynamic Pricing, Real Estate ROI, NLP Reviews & What-If Simulation.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


# ----------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------

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


class DynamicPricingRequest(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    room_type: Optional[str] = "Entire home/apt"
    base_price: Optional[float] = None
    strategy: Optional[str] = "balanced"


class ROIRequest(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    room_type: Optional[str] = "Entire home/apt"
    custom_property_price: Optional[float] = None
    down_payment_pct: Optional[float] = 25.0
    mortgage_interest_pct: Optional[float] = 6.75
    loan_term_years: Optional[int] = 30


class NLPSentimentRequest(BaseModel):
    reviews: Any
    listing_name: Optional[str] = ""


class TransitSafetyRequest(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class WhatIfRequest(BaseModel):
    current_config: Dict[str, Any]
    simulated_config: Dict[str, Any]


# ----------------------------------------------------
# API Routes
# ----------------------------------------------------

NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0"
}


@app.get('/')
def greet(request: Request):
    """
    Returns system status. If a browser requests HTML, serves the SaaS app.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and os.path.exists("index.html"):
        return FileResponse("index.html", headers=NO_CACHE_HEADERS)
    return {
        "message": "NYC Airbnb SaaS Intelligence Suite API is running successfully!",
        "status": "online",
        "version": "2.0.0",
        "features": [
            "ML Room Type Classifier",
            "Dynamic Pricing Strategy",
            "Real Estate Investment ROI Predictor",
            "NLP Review Sentiment Analysis",
            "Transit & Safety Score Integration",
            "What-If Scenario Simulator"
        ]
    }


@app.get('/style.css')
def get_style():
    if os.path.exists("style.css"):
        return FileResponse("style.css", media_type="text/css", headers=NO_CACHE_HEADERS)
    return JSONResponse(status_code=404, content={"error": "style.css not found"})


@app.get('/script.js')
def get_script():
    if os.path.exists("script.js"):
        return FileResponse("script.js", media_type="application/javascript", headers=NO_CACHE_HEADERS)
    return JSONResponse(status_code=404, content={"error": "script.js not found"})



@app.post('/predict')
def predict(features: Features):
    """
    Original room type classifier, enriched with Transit Score,
    Safety Score, and Market Price benchmark.
    """
    input_data = pd.DataFrame([features.model_dump()], columns=COLUMNS)
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    prob_dict = {
        cls: round(float(prob), 4)
        for cls, prob in zip(model.classes_, probabilities)
    }

    # Enrich with external transit and safety metrics
    transit_safety = saas_engine.get_transit_safety_score(
        borough=features.neighbourhood_group,
        neighbourhood=features.neighbourhood,
        latitude=features.latitude,
        longitude=features.longitude
    )

    return {
        "predicted_room_type": prediction,
        "probabilities": prob_dict,
        "transit_score": transit_safety["transit_score"],
        "transit_grade": transit_safety["transit_grade"],
        "nearest_subway": transit_safety["nearest_subway_station"],
        "subway_lines": transit_safety["subway_lines"],
        "walk_time_minutes": transit_safety["walk_time_minutes"],
        "safety_score": transit_safety["safety_score"],
        "safety_grade": transit_safety["safety_grade"],
        "nypd_precinct": transit_safety["nypd_precinct"]
    }


@app.post('/api/dynamic-pricing')
def dynamic_pricing(req: DynamicPricingRequest):
    """
    Dynamic pricing engine: 12-month calendar forecast,
    weekend markup, holiday surges, and off-season discount floors.
    """
    result = saas_engine.calculate_dynamic_pricing(
        borough=req.neighbourhood_group,
        neighbourhood=req.neighbourhood,
        room_type=req.room_type or "Entire home/apt",
        base_price=req.base_price,
        strategy=req.strategy or "balanced"
    )
    return result


@app.post('/api/roi-predictor')
def roi_predictor(req: ROIRequest):
    """
    Calculates Cap Rate, Cash-on-Cash Return, Monthly Cash Flow,
    and Payback Period for purchasing NYC real estate for Airbnb.
    """
    result = saas_engine.calculate_investment_roi(
        borough=req.neighbourhood_group,
        neighbourhood=req.neighbourhood,
        room_type=req.room_type or "Entire home/apt",
        custom_property_price=req.custom_property_price,
        down_payment_pct=req.down_payment_pct or 25.0,
        mortgage_interest_pct=req.mortgage_interest_pct or 6.75,
        loan_term_years=req.loan_term_years or 30
    )
    return result


@app.get('/api/roi-leaderboard')
def roi_leaderboard():
    """
    Returns ranked top investment neighborhoods across NYC.
    """
    return {"leaderboard": saas_engine.get_top_investment_leaderboard()}


@app.post('/api/nlp-sentiment')
def nlp_sentiment(req: NLPSentimentRequest):
    """
    NLP sentiment & aspect analyzer for guest reviews.
    Extracts positive highlights, critical pain points, and AI host recommendations.
    """
    return saas_engine.analyze_review_sentiment(req.reviews)


@app.get('/api/nlp-samples')
def nlp_samples():
    """
    Returns sample review batches for test evaluation.
    """
    return {"samples": saas_engine.SAMPLE_REVIEWS_CORPUS}


@app.post('/api/transit-safety')
def transit_safety(req: TransitSafetyRequest):
    """
    Calculates proximity to NYC Subway stations and NYPD safety statistics.
    """
    return saas_engine.get_transit_safety_score(
        borough=req.neighbourhood_group,
        neighbourhood=req.neighbourhood,
        latitude=req.latitude,
        longitude=req.longitude
    )


@app.post('/api/what-if-simulate')
def what_if_simulate(req: WhatIfRequest):
    """
    Simulates parameter adjustments (minimum nights, room type conversion, price, amenities)
    and predicts booking rate, monthly revenue delta, and operational workload.
    """
    return saas_engine.simulate_what_if_scenario(
        current_config=req.current_config,
        simulated_config=req.simulated_config
    )


class CNNPhotoRequest(BaseModel):
    scene_category: Optional[str] = "living_room"
    custom_caption: Optional[str] = ""
    lighting_condition: Optional[str] = "good_daylight"
    is_blurry: Optional[bool] = False


@app.post('/api/ai/deep-ann-predict')
def deep_ann_predict(features: Features):
    """
    Artificial Neural Network (ANN) deep learning room type inference.
    """
    return saas_engine.predict_room_type_deep_ann(features.dict())


@app.post('/api/ai/cnn-verify-photo')
def cnn_verify_photo(req: CNNPhotoRequest):
    """
    Convolutional Neural Network (CNN) property photo quality & room scene detector.
    """
    return saas_engine.analyze_property_photo_cnn(
        scene_category=req.scene_category or "living_room",
        custom_caption=req.custom_caption or "",
        lighting_condition=req.lighting_condition or "good_daylight",
        is_blurry=req.is_blurry or False
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)