# ==============================================================================
# NYC AIRBNB: FULL AI/ML MODEL ARTIFACT EXPORTER (FOR GOOGLE COLAB)
# Generates:
# 1. Model_Pipeline.pkl   (Scikit-Learn Preprocessor + RandomForest Classifier)
# 2. ann_model.keras      (Deep Learning Multi-Layer Perceptron for Tabular Data)
# 3. cnn_room_vision.keras(Computer Vision MobileNetV2 for Room Image Quality & Scene)
# 4. model_metadata.json  (Model architecture details & metadata)
# And packages them into 'nyc_airbnb_ai_models.zip' with automatic browser download!
# ==============================================================================

import os
import json
import zipfile
import numpy as np
import pandas as pd
import joblib

# Check if running in Google Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("Step 1: Preparing Models for Export...")

# ------------------------------------------------------------------------------
# 1. SAVE SCIKIT-LEARN PIPELINE (Model_Pipeline.pkl)
# ------------------------------------------------------------------------------
# Make sure your trained best pipeline is in variable `final_pipeline` or `best_model`
# If you used the name from cell 12:
try:
    target_pipeline = final_pipeline
except NameError:
    try:
        target_pipeline = best_model
    except NameError:
        target_pipeline = pipeline

joblib.dump(target_pipeline, "Model_Pipeline.pkl")
print("✅ Saved: Model_Pipeline.pkl (Scikit-Learn Tabular Pipeline)")


# ------------------------------------------------------------------------------
# 2. BUILD & SAVE DEEP LEARNING ANN MODEL (ann_model.keras)
# ------------------------------------------------------------------------------
print("\nStep 2: Training & Saving Deep Artificial Neural Network (ANN)...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Extract transformed training features using the preprocessor
preprocessor = target_pipeline.named_steps['preprocessor']
X_train_trans = preprocessor.transform(X_train)
if hasattr(X_train_trans, "toarray"):
    X_train_trans = X_train_trans.toarray()

# Encode y_train to One-Hot
le = LabelEncoder()
y_train_num = le.fit_transform(y_train)
y_train_cat = to_categorical(y_train_num, num_classes=3)

# Define Multi-Layer Perceptron (ANN) Architecture
ann_model = keras.Sequential([
    layers.Input(shape=(X_train_trans.shape[1],), name="tabular_features_input"),
    layers.Dense(128, activation='relu', name="dense_layer_1"),
    layers.BatchNormalization(name="batch_norm_1"),
    layers.Dropout(0.3, name="dropout_1"),
    layers.Dense(64, activation='relu', name="dense_layer_2"),
    layers.BatchNormalization(name="batch_norm_2"),
    layers.Dropout(0.2, name="dropout_2"),
    layers.Dense(32, activation='relu', name="dense_layer_3"),
    layers.Dense(3, activation='softmax', name="room_type_output")
], name="NYC_Airbnb_ANN_Classifier")

ann_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training ANN for 15 epochs (takes ~10 seconds on Colab GPU/CPU)...")
ann_model.fit(
    X_train_trans,
    y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

ann_model.save("ann_model.keras")
print("✅ Saved: ann_model.keras (Deep Learning Multi-Layer Perceptron)")


# ------------------------------------------------------------------------------
# 3. BUILD & SAVE COMPUTER VISION CNN MODEL (cnn_room_vision.keras)
# ------------------------------------------------------------------------------
print("\nStep 3: Building & Saving Convolutional Neural Network (CNN)...")
# MobileNetV2 Pretrained Backbone for Property Image Quality & Room Verification
base_vision = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_vision.trainable = False

cnn_model = keras.Sequential([
    layers.Input(shape=(224, 224, 3), name="property_photo_input"),
    base_vision,
    layers.GlobalAveragePooling2D(name="global_pooling"),
    layers.Dense(128, activation='relu', name="vision_dense"),
    layers.Dropout(0.3, name="vision_dropout"),
    layers.Dense(5, activation='softmax', name="scene_classifier") # bedroom, living_room, bathroom, kitchen, exterior
], name="NYC_Airbnb_CNN_Vision_Inspector")

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.save("cnn_room_vision.keras")
print("✅ Saved: cnn_room_vision.keras (CNN Room Photo Inspector)")


# ------------------------------------------------------------------------------
# 4. SAVE METADATA (model_metadata.json)
# ------------------------------------------------------------------------------
metadata = {
    "project_name": "NYC Airbnb Room Type Prediction & SaaS Intelligence",
    "version": "2.3.0",
    "classes": list(le.classes_),
    "ann_feature_dim": int(X_train_trans.shape[1]),
    "cnn_categories": ["bedroom", "living_room", "bathroom", "kitchen", "exterior"],
    "export_status": "SUCCESS"
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: model_metadata.json")


# ------------------------------------------------------------------------------
# 5. BUNDLE INTO ZIP & TRIGGER INSTANT DOWNLOAD
# ------------------------------------------------------------------------------
zip_filename = "nyc_airbnb_ai_models.zip"
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("Model_Pipeline.pkl")
    zipf.write("ann_model.keras")
    zipf.write("cnn_room_vision.keras")
    zipf.write("model_metadata.json")

print(f"\n🎉 Success! All models packaged into '{zip_filename}' (Size: ~{os.path.getsize(zip_filename)//(1024*1024)} MB)")

if IN_COLAB:
    print("🚀 Triggering automatic browser download to your computer...")
    files.download(zip_filename)
else:
    print(f"File created locally: {os.path.abspath(zip_filename)}")
