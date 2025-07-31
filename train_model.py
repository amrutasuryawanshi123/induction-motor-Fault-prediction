import os
import cv2
import numpy as np
import torch
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

# === Paths ===
CSV_PATH = "synthetic_maintenance_log.csv"
IMAGE_BASE_DIR = r"D:\projects\thermal_induction_motor\fault detection and severity prediction\IR-Motor-bmp"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "fault_label_encoder.pkl"
ACTION_ENCODER_PATH = "action_label_encoder.pkl"

# Model output paths
FAULT_MODEL_PATH = "fault_model.pkl"
SEVERITY_MODEL_PATH = "severity_model.pkl"
ACTION_MODEL_PATH = "action_model.pkl"
COST_MODEL_PATH = "cost_model.pkl"
DOWNTIME_MODEL_PATH = "downtime_model.pkl"

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Label mapping if needed ===
def map_fault_label(label):
    label = label.lower()
    if "noload" in label or "no load" in label:
        return "Healthy"
    elif "rotor" in label:
        return "Rotor fault"
    elif "fan" in label:
        return "Cooling fan fault"
    elif any(x in label for x in ["a", "b", "c", "winding", "stator"]):
        return "Stator fault"
    else:
        return label

df['Fault_Type'] = df['Fault_Type'].apply(map_fault_label)

# === Setup ResNet18 Feature Extractor ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
resnet = resnet18(weights=weights)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(tensor)
    return features.cpu().numpy().reshape(-1)

# === Prepare features and labels ===
features_list = []
fault_labels = []
severity_labels = []
action_labels = []
cost_values = []
downtime_values = []

print("🔍 Extracting features from images...")
for idx, row in df.iterrows():
    folder = row['Folder']
    image_path = os.path.join(IMAGE_BASE_DIR, folder, row['Image_Name'])
    if not os.path.exists(image_path):
        print(f"⚠️ Skipping missing image: {image_path}")
        continue

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        continue

    try:
        features = extract_features(img)
        features_list.append(features)
        fault_labels.append(row['Fault_Type'])
        severity_labels.append(row['Severity'])
        action_labels.append(row['ActionTaken'])
        cost_values.append(row['Cost'])
        downtime_values.append(row['Downtime_Days'])
        print(f"✅ Processed: {row['Image_Name']}")
    except Exception as e:
        print(f"❌ Error processing {row['Image_Name']}: {e}")

# === Convert to arrays ===
X = np.vstack(features_list)
y_fault = np.array(fault_labels)
y_severity = np.array(severity_labels)
y_action = np.array(action_labels)
y_cost = np.array(cost_values)
y_downtime = np.array(downtime_values)

# === Encode categorical targets ===
fault_encoder = LabelEncoder()
action_encoder = LabelEncoder()
y_fault_encoded = fault_encoder.fit_transform(y_fault)
y_action_encoded = action_encoder.fit_transform(y_action)

# Save encoders
joblib.dump(fault_encoder, ENCODER_PATH)
joblib.dump(action_encoder, ACTION_ENCODER_PATH)
print("✅ Encoders saved")

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print("✅ Scaler saved")

# === Train Models ===

# Fault Classification
fault_model = RandomForestClassifier(n_estimators=100, random_state=42)
fault_model.fit(X_scaled, y_fault_encoded)
joblib.dump(fault_model, FAULT_MODEL_PATH)
print("✅ Fault type classifier saved")

# Severity Regression
severity_model = RandomForestRegressor(n_estimators=100, random_state=42)
severity_model.fit(X_scaled, y_severity)
joblib.dump(severity_model, SEVERITY_MODEL_PATH)
print("✅ Severity regressor saved")

# Action Prediction
action_model = RandomForestClassifier(n_estimators=100, random_state=42)
action_model.fit(X_scaled, y_action_encoded)
joblib.dump(action_model, ACTION_MODEL_PATH)
print("✅ Action recommender model saved")

# Cost Regression
cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
cost_model.fit(X_scaled, y_cost)
joblib.dump(cost_model, COST_MODEL_PATH)
print("✅ Cost predictor saved")

# Downtime Regression
downtime_model = RandomForestRegressor(n_estimators=100, random_state=42)
downtime_model.fit(X_scaled, y_downtime)
joblib.dump(downtime_model, DOWNTIME_MODEL_PATH)
print("✅ Downtime predictor saved")
