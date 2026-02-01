import os
import csv

BASE_DIR = "dataset/images"

calorie_map = {
    "aloo_gobi": 180,
    "aloo_tikki": 220,
    "bandar_laddu": 150,
    "Barbecue_chicken": 300,
    "bhindi_masala": 160,
    "biryani": 420,
    "Brownies": 280,
    "Butter_Chicken": 350,
    "chana_masala": 240,
    "chapati": 120,
    "chicken burger": 320,
    "chicken kabab": 270,
    "Chicken Mashroam white rice": 380,
    "Chicken_Biryani": 450
}

rows = []

for food in os.listdir(BASE_DIR):
    food_path = os.path.join(BASE_DIR, food)
    if food not in calorie_map:
        continue
    for img in os.listdir(food_path):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            rows.append([
                f"images/{food}/{img}",
                food,
                calorie_map[food]
            ])

with open("dataset/labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "food_type", "calories"])
    writer.writerows(rows)

print("✅ labels.csv created successfully")
