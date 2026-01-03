import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. MODELL ÉS KONFIGURÁCIÓ ---
model = YOLO("best_nano.pt")
output_size = 600

# Ideális belső rácspontok generálása (7x7-es belső háló a 8x8-as táblán)
# Ezek a pontok egy tökéletes 600x600-as képen a mezők találkozásai
margin = output_size / 8
ideal_grid_points = []
for y in range(1, 8):
    for x in range(1, 8):
        ideal_grid_points.append([x * margin, y * margin])
ideal_grid_points = np.array(ideal_grid_points, dtype=np.float32)

def find_saddle_points(gray, mask):
    """Nyeregpontok keresése a maszkolt területen belül."""
    res = cv2.cornerHarris(gray, 2, 3, 0.04)
    res = cv2.dilate(res, None)
    ret, dst = cv2.threshold(res, 0.01 * res.max(), 255, 0)
    dst = np.uint8(dst)
    dst = cv2.bitwise_and(dst, dst, mask=mask)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    return centroids[1:]

def get_stable_homography(src_points):
    """
    RANSAC alapú homográfia becslés.
    Kiszűri azokat a pontokat, amik nem a tábla síkjában vannak.
    """
    # Legalább 4 pont kell a homográfiához, de a stabil RANSAC-hoz 10+ javasolt
    if len(src_points) < 8: 
        return None
    
    # Mivel nem tudjuk pontosan melyik talált pont melyik ideális rácspont, 
    # fix kameránál a legegyszerűbb, ha a talált pontokat a legközelebbi 
    # elvárt rácsponthoz rendeljük, vagy bázisvektorokkal szűrünk.
    
    # Ebben a példában feltételezzük, hogy a szegmentáció alapján már 
    # nagyjából behatároltuk a pontokat.
    # A RANSAC 5.0-ás küszöbbel kidobja a bábuk tetején talált pontokat.
    
    # Megjegyzés: A findHomography-hoz src és dst listák azonos hosszúságúak kell legyenek.
    # Itt egy egyszerűsített illesztést alkalmazunk:
    M, mask = cv2.findHomography(src_points, ideal_grid_points[:len(src_points)], cv2.RANSAC, 5.0)
    return M

cap = cv2.VideoCapture("test_video_1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(frame, conf=0.6, verbose=False)[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if results.masks is not None:
        mask_data = results.masks.data[0].cpu().numpy()
        mask = cv2.resize((mask_data * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]))
        
        # 1. Nyeregpontok detektálása
        pts = find_saddle_points(gray, mask)
        
        # 2. Homográfia számítása RANSAC-kal a pontok alapján
        # Ha elegendő pontunk van, a findHomography stabilabb lesz, mint a 4 sarok
        if len(pts) >= 4:
            # Itt a pts-t rendezni/szűrni kell, hogy illeszkedjen az ideal_grid_points-hoz
            # Egyelőre maradjunk a biztonságos 4-sarkos alapnál, ha kevés a belső pont
            
            poly = results.masks.xy[0].astype(np.int32)
            hull = cv2.convexHull(poly)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) == 4:
                # Nyers sarokpontok a szegmentációból
                src_corners = approx.reshape(4, 2).astype(np.float32)
                # Képzeletbeli cél sarkok
                dst_corners = np.array([[0,0], [output_size,0], [output_size,output_size], [0,output_size]], dtype="float32")
                
                # Rendezés (fontos!)
                def order_points(pts):
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    return rect
                
                ordered_src = order_points(src_corners)
                
                # Stabil homográfia: findHomography RANSAC-kal (src, dst, módszer, küszöb)
                M, _ = cv2.findHomography(ordered_src, dst_corners, cv2.RANSAC, 5.0)
                
                if M is not None:
                    warped = cv2.warpPerspective(frame, M, (output_size, output_size))

                    # 3. Rács kirajzolása a stabilizált képre
                    for i in range(9):
                        pos = int(i * output_size / 8)
                        cv2.line(warped, (0, pos), (output_size, pos), (0, 255, 0), 1)
                        cv2.line(warped, (pos, 0), (pos, output_size), (0, 255, 0), 1)

                    cv2.imshow("Warped (RANSAC stabil)", warped)

    cv2.imshow("Original", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()