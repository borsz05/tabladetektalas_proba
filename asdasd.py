from ultralytics import YOLO
import cv2

# 1. Modell betöltése (a saját best.pt fájlod)
model = YOLO("best.pt")

# 2. Predikció futtatása a képen
# A 'conf' a magabiztossági küszöb (0.5 = 50%)
results = model.predict("sakktabla12.jpg", conf=0.5)[0]

# 3. Az eredmények kirajzolása (ez automatikusan ráteszi a maszkot és a keretet)
annotated_frame = results.plot()

# 4. Megjelenítés
# cv2.imshow("YOLOv11 Szegmentacio", annotated_frame)

# Mentés fájlba (opcionális)
cv2.imwrite("eredmeny.jpg", annotated_frame)
