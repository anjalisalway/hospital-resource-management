import csv
import random
from datetime import datetime

# Hospital names and locations (realistic coordinates for a metropolitan area)
hospitals = [
    {"name": "City General Hospital", "lat": 40.7580, "lon": -73.9855},
    {"name": "St. Mary's Medical Center", "lat": 40.7489, "lon": -73.9680},
    {"name": "Riverside Community Hospital", "lat": 40.7614, "lon": -73.9776},
    {"name": "Memorial Healthcare Center", "lat": 40.7505, "lon": -73.9934},
    {"name": "Metropolitan Hospital", "lat": 40.7738, "lon": -73.9524},
    {"name": "Lakeside Medical Center", "lat": 40.7410, "lon": -73.9937},
    {"name": "Northern Regional Hospital", "lat": 40.7831, "lon": -73.9712},
    {"name": "Sunset Valley Hospital", "lat": 40.7364, "lon": -73.9808},
    {"name": "Parkview Medical Complex", "lat": 40.7690, "lon": -73.9595},
    {"name": "Eastside General Hospital", "lat": 40.7529, "lon": -73.9540},
    {"name": "Westbrook Health Center", "lat": 40.7456, "lon": -74.0047},
    {"name": "Highland Medical Institute", "lat": 40.7789, "lon": -73.9820},
    {"name": "Oakwood Community Hospital", "lat": 40.7382, "lon": -73.9730},
    {"name": "Central Care Hospital", "lat": 40.7555, "lon": -73.9733},
    {"name": "Greenfield Medical Center", "lat": 40.7658, "lon": -73.9910},
    {"name": "Harborview Hospital", "lat": 40.7421, "lon": -73.9568},
    {"name": "Summit Regional Medical", "lat": 40.7712, "lon": -73.9651},
    {"name": "Bay Area Hospital", "lat": 40.7345, "lon": -73.9889},
    {"name": "Valley View Medical Center", "lat": 40.7601, "lon": -73.9845},
    {"name": "Crescent Hills Hospital", "lat": 40.7478, "lon": -73.9712}
]

# Equipment types with typical quantities
equipment_types = [
    "Ventilators", "MRI Machines", "CT Scanners", "X-Ray Machines",
    "Ultrasound Machines", "ECG Machines", "Defibrillators",
    "Infusion Pumps", "Patient Monitors", "Surgical Tables",
    "Anesthesia Machines", "Dialysis Machines", "ICU Beds",
    "Emergency Beds", "Operating Rooms", "Wheelchairs",
    "Stretchers", "Oxygen Concentrators", "CPAP Machines", "Nebulizers"
]

# Staff roles
staff_roles = [
    "Emergency Physician", "Surgeon", "Anesthesiologist", "Radiologist",
    "Cardiologist", "Neurologist", "Pediatrician", "General Practitioner",
    "ICU Nurse", "Emergency Nurse", "Surgical Nurse", "General Nurse",
    "Lab Technician", "Radiologic Technologist", "Pharmacist",
    "Respiratory Therapist", "Physical Therapist", "Administrative Staff"
]

# Shift timings
shifts = [
    {"name": "Morning", "start": "06:00", "end": "14:00"},
    {"name": "Evening", "start": "14:00", "end": "22:00"},
    {"name": "Night", "start": "22:00", "end": "06:00"}
]

# Initialize data lists for CSV files
hospitals_data = []
departments_data = []
equipment_data = []
staff_shifts_data = []
alerts_data = []
specializations_data = []

# Generate hospital data
for idx, hospital in enumerate(hospitals):
    hospital_id = f"HOSP_{idx+1:03d}"
    hospital_name = hospital["name"]
    
    # Determine hospital size (small, medium, large)
    hospital_size = random.choice(["small", "medium", "large"])
    
    if hospital_size == "small":
        bed_capacity = random.randint(50, 150)
        staff_multiplier = 0.6
        equipment_multiplier = 0.5
    elif hospital_size == "medium":
        bed_capacity = random.randint(150, 300)
        staff_multiplier = 1.0
        equipment_multiplier = 1.0
    else:
        bed_capacity = random.randint(300, 500)
        staff_multiplier = 1.5
        equipment_multiplier = 1.5
    
    # Add hospital data
    hospitals_data.append({
        "hospital_id": hospital_id,
        "hospital_name": hospital_name,
        "latitude": hospital["lat"],
        "longitude": hospital["lon"],
        "address": f"{random.randint(100, 9999)} Medical Pkwy, City, ST {random.randint(10000, 99999)}",
        "hospital_size": hospital_size,
        "total_bed_capacity": bed_capacity,
        "operational_hours": "24/7",
        "emergency_services": "Yes",
        "trauma_level": random.choice(["Level I", "Level II", "Level III"]),
        "last_updated": datetime.now().isoformat()
    })
    
    # Generate departments
    dept_types = [
        {"name": "Emergency", "pct": 0.15, "occ_min": 0.6, "occ_max": 0.95},
        {"name": "ICU", "pct": 0.1, "occ_min": 0.7, "occ_max": 0.95},
        {"name": "General Ward", "pct": 0.4, "occ_min": 0.5, "occ_max": 0.85},
        {"name": "Surgical", "pct": 0.15, "occ_min": 0.4, "occ_max": 0.8},
        {"name": "Pediatric", "pct": 0.1, "occ_min": 0.45, "occ_max": 0.75},
        {"name": "Maternity", "pct": 0.1, "occ_min": 0.5, "occ_max": 0.8}
    ]
    
    for dept in dept_types:
        departments_data.append({
            "hospital_id": hospital_id,
            "hospital_name": hospital_name,
            "department_name": dept["name"],
            "bed_count": int(bed_capacity * dept["pct"]),
            "occupancy_rate": round(random.uniform(dept["occ_min"], dept["occ_max"]) * 100, 2)
        })
    
    # Generate equipment inventory
    for equip in equipment_types:
        if equip in ["Ventilators"]:
            base_qty = random.randint(10, 30)
        elif equip in ["MRI Machines", "CT Scanners"]:
            base_qty = random.randint(1, 4)
        elif equip in ["X-Ray Machines", "Ultrasound Machines"]:
            base_qty = random.randint(2, 8)
        elif equip in ["ICU Beds"]:
            base_qty = random.randint(20, 80)
        elif equip in ["Emergency Beds"]:
            base_qty = random.randint(15, 50)
        elif equip in ["Operating Rooms"]:
            base_qty = random.randint(4, 12)
        elif equip in ["Wheelchairs", "Stretchers"]:
            base_qty = random.randint(20, 60)
        else:
            base_qty = random.randint(5, 25)
        
        total_qty = int(base_qty * equipment_multiplier)
        available_qty = random.randint(int(total_qty * 0.3), total_qty)
        in_use_qty = total_qty - available_qty
        maintenance_qty = random.randint(0, max(1, int(total_qty * 0.1)))
        
        equipment_data.append({
            "hospital_id": hospital_id,
            "hospital_name": hospital_name,
            "equipment_type": equip,
            "total_quantity": total_qty,
            "available_quantity": available_qty,
            "in_use_quantity": in_use_qty,
            "maintenance_quantity": maintenance_qty,
            "equipment_condition": random.choice(["Excellent", "Good", "Fair"]),
            "last_updated": datetime.now().isoformat()
        })
    
    # Generate staff
    for role in staff_roles:
        if "Physician" in role or "Surgeon" in role or "ologist" in role:
            base_count = random.randint(3, 8)
        elif "Nurse" in role:
            base_count = random.randint(15, 40)
        elif "Technician" in role or "Technologist" in role:
            base_count = random.randint(5, 15)
        else:
            base_count = random.randint(4, 12)
        
        total_count = int(base_count * staff_multiplier)
        
        for shift in shifts:
            staff_on_shift = random.randint(
                int(total_count * 0.25), 
                int(total_count * 0.4)
            )
            staff_shifts_data.append({
                "hospital_id": hospital_id,
                "hospital_name": hospital_name,
                "staff_role": role,
                "shift_name": shift["name"],
                "shift_start_time": shift["start"],
                "shift_end_time": shift["end"],
                "staff_count": staff_on_shift,
                "total_staff_in_role": total_count
            })
    
    # Generate specializations
    hospital_specializations = random.sample([
        "Cardiology", "Neurology", "Orthopedics", "Oncology",
        "Pediatrics", "Obstetrics", "Emergency Medicine", "Surgery"
    ], k=random.randint(3, 6))
    
    for spec in hospital_specializations:
        specializations_data.append({
            "hospital_id": hospital_id,
            "hospital_name": hospital_name,
            "specialization": spec
        })
    
    # Generate alerts (30% chance equipment shortage, 20% chance staff shortage)
    if random.random() < 0.3:
        critical_equipment = random.choice(equipment_types)
        alerts_data.append({
            "hospital_id": hospital_id,
            "hospital_name": hospital_name,
            "alert_type": "Equipment Shortage",
            "resource_name": critical_equipment,
            "severity": random.choice(["High", "Medium"]),
            "alert_message": f"Low availability of {critical_equipment}",
            "alert_timestamp": datetime.now().isoformat(),
            "is_resolved": "No"
        })
    
    if random.random() < 0.2:
        critical_role = random.choice(staff_roles)
        alerts_data.append({
            "hospital_id": hospital_id,
            "hospital_name": hospital_name,
            "alert_type": "Staff Shortage",
            "resource_name": critical_role,
            "severity": "High",
            "alert_message": f"Urgent need for {critical_role}",
            "alert_timestamp": datetime.now().isoformat(),
            "is_resolved": "No"
        })

# Write to CSV files
def write_csv(filename, data, fieldnames):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"✓ Created {filename} ({len(data)} rows)")

# Write all CSV files
write_csv('hospitals.csv', hospitals_data, 
          ['hospital_id', 'hospital_name', 'latitude', 'longitude', 'address', 
           'hospital_size', 'total_bed_capacity', 'operational_hours', 
           'emergency_services', 'trauma_level', 'last_updated'])

write_csv('departments.csv', departments_data,
          ['hospital_id', 'hospital_name', 'department_name', 'bed_count', 'occupancy_rate'])

write_csv('equipment_inventory.csv', equipment_data,
          ['hospital_id', 'hospital_name', 'equipment_type', 'total_quantity', 'available_quantity',
           'in_use_quantity', 'maintenance_quantity', 'equipment_condition', 'last_updated'])

write_csv('staff_shifts.csv', staff_shifts_data,
          ['hospital_id', 'hospital_name', 'staff_role', 'shift_name', 'shift_start_time',
           'shift_end_time', 'staff_count', 'total_staff_in_role'])

write_csv('resource_alerts.csv', alerts_data,
          ['hospital_id', 'hospital_name', 'alert_type', 'resource_name', 'severity',
           'alert_message', 'alert_timestamp', 'is_resolved'])

write_csv('hospital_specializations.csv', specializations_data,
          ['hospital_id', 'hospital_name', 'specialization'])

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total Hospitals: {len(hospitals_data)}")
print(f"Total Departments: {len(departments_data)}")
print(f"Total Equipment Records: {len(equipment_data)}")
print(f"Total Staff Shift Records: {len(staff_shifts_data)}")
print(f"Total Active Alerts: {len(alerts_data)}")
print(f"Total Specializations: {len(specializations_data)}")
print(f"\nTotal Beds in System: {sum(h['total_bed_capacity'] for h in hospitals_data)}")
print(f"Hospitals with Alerts: {len(set(a['hospital_id'] for a in alerts_data))}")

print("\n" + "="*80)
print("CSV FILES CREATED:")
print("="*80)
print("1. hospitals.csv - Main hospital information")
print("2. departments.csv - Department details by hospital")
print("3. equipment_inventory.csv - Equipment availability")
print("4. staff_shifts.csv - Staff allocation across shifts")
print("5. resource_alerts.csv - Active resource shortage alerts")
print("6. hospital_specializations.csv - Hospital specializations")

# Display sample data from hospitals.csv
print("\n" + "="*80)
print("SAMPLE DATA (hospitals.csv - first 3 rows):")
print("="*80)
for i, hospital in enumerate(hospitals_data[:3]):
    print(f"\nHospital {i+1}:")
    for key, value in hospital.items():
        print(f"  {key}: {value}")