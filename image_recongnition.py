import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

folder_path = r"C:\Users\user1\Documents\GitHub\coddy_image_recognition\images"
total_contours = 0
results = []

for index, filename in enumerate(os.listdir(folder_path),start=1):
    img_path = os.path.join(folder_path, filename)
    image = cv2.imread(img_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define orange range (you may tune this slightly)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count_contours = len(contours)
    total_contours += count_contours
    print(f"The number of orange blobs for image {index} - {filename}: {count_contours}")

    results.append({
        "image_number": index,
        "filename" : filename,
        "blob_count" : count_contours
    })

print(f"Total Orange blobs: {total_contours}")

df = pd.DataFrame(results)

print(df)

df["month_year"] = df["filename"].str.extract(r'_(.*)\.png')

print(df)

df["date"] = pd.to_datetime(df["month_year"], format="%b%Y", errors="coerce")

print(df)

# df = df.sort_values("date")
# print(df)

plt.plot(df["date"], df["blob_count"])
plt.title("Number of Orange blob over time")
plt.xlabel("Date")
plt.ylabel("Number of Orange Blob")
plt.xticks(rotation = 45)
plt.show()



