import random
import numpy as np
import pandas as pd

# ثبات النتائج
random.seed(42)
np.random.seed(42)

NUM_ROWS = 10000

crane_types = {
    "Tower Crane": {"capacity": 8000, "boom_min": 40, "boom_max": 80},
    "Mobile Crane": {"capacity": 5000, "boom_min": 20, "boom_max": 50},
    "Crawler Crane": {"capacity": 12000, "boom_min": 40, "boom_max": 90},
    "Rough Terrain Crane": {"capacity": 7000, "boom_min": 25, "boom_max": 60},
}

load_types = {
    "Steel Beam": (500, 3000, 1, 4),
    "Concrete Block": (1000, 5000, 1, 3),
    "Shipping Container": (1500, 4500, 10, 18),
    "Glass Panel": (200, 1200, 6, 14),
    "Steel Pipe": (300, 2500, 2, 6),
    "Machinery": (800, 4000, 2, 8),
    "Wind Turbine Blade": (1500, 3500, 15, 30),
    "Precast Wall": (1000, 4500, 5, 12),
}

rows = []

for _ in range(NUM_ROWS):

    ###################################################
    # Crane
    ###################################################

    crane_type = random.choice(list(crane_types.keys()))
    crane = crane_types[crane_type]

    capacity = crane["capacity"]

    boom_length = round(
        random.uniform(
            crane["boom_min"],
            crane["boom_max"]
        ),1
    )

    boom_angle = random.randint(25,75)

    lifting_radius = round(
        random.uniform(
            boom_length*0.3,
            boom_length*0.85
        ),1
    )

    hook_height = round(
        boom_length*np.sin(np.radians(boom_angle))
        + random.uniform(-2,2),1
    )

    ###################################################
    # Load
    ###################################################

    load_type = random.choice(list(load_types.keys()))

    wmin,wmax,amin,amax = load_types[load_type]

    load_weight = random.randint(wmin,wmax)

    load_area = round(random.uniform(amin,amax),2)

    ###################################################
    # Weather
    ###################################################

    wind_speed = round(
        min(np.random.weibull(2.0)*7,30),
        1
    )

    gust_speed = round(
        wind_speed*random.uniform(1.05,1.40),
        1
    )

    wind_direction = random.randint(0,359)

    temperature = round(random.uniform(-5,45),1)

    humidity = random.randint(25,100)

    pressure = round(random.uniform(990,1035),1)

    visibility = random.randint(500,10000)

    rain = round(max(0,np.random.normal(1,3)),1)

    ###################################################
    # Crane condition
    ###################################################

    crane_tilt = round(
        abs(np.random.normal(0.2,0.3)),
        2
    )

    operator_exp = random.randint(1,30)

    ###################################################
    # Risk Score
    ###################################################

    load_ratio = load_weight/capacity

    risk = 0

    # الرياح
    risk += (wind_speed/30)*35

    # الهبات
    risk += (gust_speed/40)*15

    # نسبة الحمولة
    risk += load_ratio*20

    # مساحة الحمولة
    risk += (load_area/30)*10

    # نصف القطر
    risk += (lifting_radius/boom_length)*10

    # ميل الرافعة
    risk += min(crane_tilt/2,1)*5

    # المطر
    risk += min(rain/20,1)*3

    # قلة الرؤية
    risk += (1-visibility/10000)*2

    # خبرة المشغل تقلل الخطورة قليلاً
    risk -= operator_exp*0.15

    risk = max(0,min(100,risk))

    ###################################################
    # Label
    ###################################################

    if risk < 35:
        safety = "Safe"
    elif risk < 60:
        safety = "Warning"
    else:
        safety = "Unsafe"

    rows.append([
        crane_type,
        capacity,
        boom_length,
        boom_angle,
        lifting_radius,
        hook_height,
        load_type,
        load_weight,
        load_area,
        wind_speed,
        gust_speed,
        wind_direction,
        temperature,
        humidity,
        pressure,
        visibility,
        rain,
        crane_tilt,
        operator_exp,
        round(risk,2),
        safety
    ])

columns = [
    "crane_type",
    "crane_capacity_kg",
    "boom_length_m",
    "boom_angle_deg",
    "lifting_radius_m",
    "hook_height_m",
    "load_type",
    "load_weight_kg",
    "load_area_m2",
    "wind_speed_ms",
    "gust_speed_ms",
    "wind_direction_deg",
    "temperature_c",
    "humidity_percent",
    "air_pressure_hpa",
    "visibility_m",
    "rain_mm",
    "crane_tilt_deg",
    "operator_experience_years",
    "risk_score",
    "safety_status"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv("crane_safety_10000.csv", index=False)

print(df.head())

print("\nShape :", df.shape)

print("\nSafety Distribution")
print(df["safety_status"].value_counts())

print("\nDataset saved as crane_safety_10000.csv")