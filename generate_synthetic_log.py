# generate_synthetic_log.py

import os
import pandas as pd
import random
from datetime import datetime, timedelta

DATASET_PATH = r"D:\projects\thermal_induction_motor\fault detection and severity prediction\IR-Motor-bmp"
OUTPUT_CSV = "synthetic_maintenance_log.csv"

def map_fault_label(folder_name):
    name = folder_name.lower()
    if "noload" in name:
        return "Healthy"
    elif "fan" in name:
        return "Cooling fan fault"
    elif "rotor" in name:
        return "Rotor fault"
    elif any(x in name for x in ["a", "b", "c", "winding"]):
        return "Stator fault"
    else:
        return "Unknown"

def extract_severity(folder_name):
    for token in folder_name.split('&'):
        digits = ''.join(filter(str.isdigit, token))
        if digits:
            return int(digits)
    return 0

def simulate_action_and_cost(fault, severity):
    if fault == "Healthy":
        return "No action", "No issue", 0, 0
    elif severity >= 50:
        return "Component Replacement", "Resolved", random.randint(3000, 8000), random.randint(1, 3)
    elif severity >= 30:
        return "Repair + Calibration", "Resolved", random.randint(1000, 3000), random.randint(1, 2)
    elif severity >= 10:
        return "Scheduled Inspection", "Resolved", random.randint(300, 1000), 0
    else:
        return "Monitoring Only", "Monitoring", 0, 0

rows = []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    fault_type = map_fault_label(folder)
    severity = extract_severity(folder)

    for img in os.listdir(folder_path):
        if not img.lower().endswith((".bmp", ".jpg", ".png")):
            continue

        image_name = img
        action, resolution, cost, downtime = simulate_action_and_cost(fault_type, severity)
        repair_start = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 100))
        repair_end = repair_start + timedelta(days=downtime)

        rows.append({
            "Image_Name": image_name,
            "Folder": folder,
            "Fault_Type": fault_type,
            "Severity": severity,
            "ActionTaken": action,
            "ResolutionStatus": resolution,
            "Cost": cost,
            "Downtime_Days": downtime,
            "RepairStartDate": repair_start.date(),
            "RepairEndDate": repair_end.date()
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Synthetic maintenance log saved: {OUTPUT_CSV}")
