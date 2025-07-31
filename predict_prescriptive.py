import os
import cv2
import numpy as np
import torch
import joblib
import pandas as pd
from datetime import datetime
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

# === Paths ===
TEST_IMAGE_DIR = r"d:\projects\thermal_induction_motor\fault detection and severity prediction\testing"
OUTPUT_DIR = r"D:\projects\thermal_induction_motor\prescriptive maintenance\prescriptive_outputs"
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "prescriptive_predictions.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Model Paths ===
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "fault_label_encoder.pkl"
ACTION_ENCODER_PATH = "action_label_encoder.pkl"
FAULT_MODEL_PATH = "fault_model.pkl"
SEVERITY_MODEL_PATH = "severity_model.pkl"
ACTION_MODEL_PATH = "action_model.pkl"
COST_MODEL_PATH = "cost_model.pkl"
DOWNTIME_MODEL_PATH = "downtime_model.pkl"

# === Load Models and Tools ===
scaler = joblib.load(SCALER_PATH)
fault_encoder = joblib.load(ENCODER_PATH)
action_encoder = joblib.load(ACTION_ENCODER_PATH)
fault_model = joblib.load(FAULT_MODEL_PATH)
severity_model = joblib.load(SEVERITY_MODEL_PATH)
action_model = joblib.load(ACTION_MODEL_PATH)
cost_model = joblib.load(COST_MODEL_PATH)
downtime_model = joblib.load(DOWNTIME_MODEL_PATH)

# === ResNet Feature Extractor ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(image):
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.cpu().numpy().reshape(-1)

# === Rule-based Reasoning ===
def get_reason(fault_type):
    fault_type = fault_type.lower()
    if "stator" in fault_type:
        return "Insulation degradation due to thermal, electrical, mechanical, or environmental stresses."
    elif "rotor" in fault_type:
        return "Manufacturing defects, mechanical issues, or environmental stresses during operation."
    elif "cooling" in fault_type:
        return "Lack of airflow, bearing failure, or electrical issues affecting the cooling system."
    elif "healthy" in fault_type:
        return "No fault detected. Motor is operating within normal thermal parameters."
    else:
        return "Unidentified fault. Further diagnostics required."

def get_recommendation(fault_type, severity):
    fault_type = fault_type.lower()
    if "stator" in fault_type:
        if severity >= 50:
            return "Replace stator windings immediately."
        elif severity >= 30:
            return "Schedule winding inspection and repair."
        else:
            return "Monitor windings in next maintenance cycle."
    elif "rotor" in fault_type:
        return "Inspect rotor and replace damaged bars or balance shaft."
    elif "cooling" in fault_type:
        return "Clean fan, unblock vents, and replace damaged blades if necessary."
    elif "healthy" in fault_type:
        return "No maintenance required at this time."
    else:
        return "Run diagnostics to identify the fault."

def get_next_step(severity):
    if severity >= 50:
        return "Immediate shutdown and corrective action required."
    elif severity >= 30:
        return "Schedule maintenance within 24 hours."
    else:
        return "Continue monitoring during routine checks."

# === Prediction Loop ===
VALID_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png')
results = []

for img_name in os.listdir(TEST_IMAGE_DIR):
    if not img_name.lower().endswith(VALID_EXTENSIONS):
        continue

    img_path = os.path.join(TEST_IMAGE_DIR, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"⚠️ Could not load image: {img_path}")
        continue

    try:
        features = extract_features(image)
        scaled_features = scaler.transform([features])

        fault_pred = fault_model.predict(scaled_features)[0]
        fault_label = fault_encoder.inverse_transform([fault_pred])[0]

        severity = severity_model.predict(scaled_features)[0]
        action_pred = action_model.predict(scaled_features)[0]
        action_label = action_encoder.inverse_transform([action_pred])[0]

        cost = cost_model.predict(scaled_features)[0]
        downtime = downtime_model.predict(scaled_features)[0]

        # Generate reason and recommendation
        reason = get_reason(fault_label)
        recommendation = get_recommendation(fault_label, severity)
        next_step = get_next_step(severity)

        # Save annotated image
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(annotated, f"{fault_label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated, f"Severity: {round(severity, 2)}%", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        pred_name = f"pred_{img_name}"
        save_path = os.path.join(OUTPUT_DIR, pred_name)
        cv2.imwrite(save_path, annotated)

        results.append({
            "Image": pred_name,
            "Fault_Type": fault_label,
            "Severity": round(severity, 2),
            "ActionTaken": action_label,
            "Estimated_Cost": round(cost, 2),
            "Estimated_Downtime_Days": round(downtime, 2),
            "Reason": reason,
            "Recommendation": recommendation,
            "Next_Step": next_step
        })

        print(f"✅ Saved: {save_path}")

    except Exception as e:
        print(f"❌ Failed on {img_name}: {e}")

# === Save to CSV ===
pd.DataFrame(results).to_csv(CSV_OUTPUT_PATH, index=False)
print(f"\n📄 Prescriptive maintenance predictions saved to:\n{CSV_OUTPUT_PATH}")
