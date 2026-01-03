import cv2 as cv
import numpy as np
import os
from sklearn.cluster import MeanShift

# ArUco beállítások
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
aruco_params = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

# Kép betöltése - Ellenőrizd az elérési utat!
img_path = '../sakktabla9.jpg'
img = cv.imread(img_path)


# JAVÍTOTT MEGJELENÍTÉS: Automatikus skálázás, hogy ne lógjon le az alja
def show_resized(name, image):
    screen_max_height = 900  # Itt állítsd be, mekkora legyen max a magasság a monitorodon
    h, w = image.shape[:2]
    
    # Kiszámoljuk a skálát úgy, hogy a magasság ne lépje túl a limitet
    if h > screen_max_height:
        scale = screen_max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv.resize(image, (new_w, new_h))
    else:
        resized = image
        
    cv.imshow(name, resized)


def get_lattice_vectors(points):
    """Kiszámolja a két domináns irányvektort a pontokból."""
    displacements = []
    pts = np.array(points)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j:
                displacements.append(pts[j] - pts[i])
    
    displacements = np.array(displacements)
    # Szűrünk a tipikus mezőméretekre (800x800-as képen kb. 80-120 pixel)
    filtered = displacements[(np.linalg.norm(displacements, axis=1) > 60) & 
                             (np.linalg.norm(displacements, axis=1) < 140)]
    
    if len(filtered) < 10: return None, None

    ms = MeanShift(bandwidth=15)
    ms.fit(filtered)
    centers = ms.cluster_centers_
    
    # Kiválasztjuk a leginkább vízszintes és függőleges vektort
    v_h = centers[np.argmin(np.abs(centers[:, 1]))]
    v_v = centers[np.argmin(np.abs(centers[:, 0]))]
    return v_h, v_v

def reconstruct_lattice(detected_points, v_horiz, v_vert):
    pts = np.array(detected_points)
    
    # 1. JAVÍTOTT KEZDŐPONT: 
    # A warped képen a sakktábla bal felső sarka elméletileg a (0,0)
    # Keressük meg azt a detektált pontot, ami a legközelebb van a (0,0)-hoz,
    # DE eltoljuk egy kicsit befelé, hogy ne a tábla szélét, hanem az első sarkot kapjuk el
    dists_from_origin = np.linalg.norm(pts - np.array([0, 0]), axis=1)
    best_start = pts[np.argmin(dists_from_origin)]

    # 2. Elméleti rács generálása
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            grid[i, j] = best_start + i * v_vert + j * v_horiz
            
    # 3. GLOBÁLIS FINOMÍTÁS (ITERATÍV)
    # Kétszer futtatjuk le, hogy a rács "beüljön" a helyére
    for _ in range(2):
        total_offset = np.array([0.0, 0.0])
        valid_matches = 0
        for row in grid:
            for gp in row:
                diffs = pts - gp
                dists = np.linalg.norm(diffs, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < 50: # Megengedőbb küszöb az elején
                    total_offset += diffs[min_idx]
                    valid_matches += 1
        
        if valid_matches > 0:
            grid += (total_offset / valid_matches)
        
    return grid

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
corners, ids, _ = detector.detectMarkers(gray)
annotated_img = img.copy()

if ids is not None:
    cv.aruco.drawDetectedMarkers(annotated_img, corners, ids)
    print(f"Talált markerek ID-i: {ids.flatten()}")

    if len(ids) == 4:
        # Koordináták kinyerése ID alapján (0,1,2,3)
        det_pts = {ids[i][0]: np.mean(corners[i][0], axis=0) for i in range(len(ids))}
        
        try:
            # 4 sarokpont sorrendben (Ügyelj rá, hogy a markereid jól legyenek elhelyezve!)
            pts_src = np.array([det_pts[0], det_pts[1], det_pts[2], det_pts[3]], dtype="float32")
            pts_dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype="float32")

            # Homográfia mátrix kiszámítása
            matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
            # Transzformáció az eredeti (rajzmentes) képen
            warped = cv.warpPerspective(img, matrix, (800, 800))

            cv.imshow("Felulnezet (Menteshez nyomj 's'-t)", warped)
            
            # Megvárjuk a billentyűleütést
            key = cv.waitKey(0) & 0xFF 
            if key == ord('s'):
                cv.imwrite('ures_tabla.jpg', warped)
                print("Siker! A 'ures_tabla.jpg' elmentve a mappádba.")
            
        except KeyError as e:
            print(f"Hiba: Hiányzik a(z) {e} ID-jú marker a képről!")
    else:
        print(f"Hiba: 4 marker kellene, de csak {len(ids)}-t találtam.")
else:
    print("Hiba: Egyetlen ArUco markert sem találtam a képen!")


#1. Szürkeárnyalat és élek
gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray_warped, 50, 150, apertureSize=3)

# 2. Vonalak keresése (HoughLinesP a szakaszokhoz)
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# 3. Itt rajzold ki a vonalakat, hogy lásd, megtalálja-e a rácsot
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(warped, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imshow("Talalt vonalak", warped)
cv.waitKey(0)
cv.destroyAllWindows()



# (Feltételezzük, hogy a 'warped' változóban ott a 800x800-as kép)

def filter_and_cluster_lines(lines, img_size=800):
    if lines is None: return [], []
    
    horizontals = []
    verticals = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Vonal hossza és dőlésszöge alapján szűrünk
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 10 or angle > 170: # Vízszintes (0 fok környéke)
            horizontals.append(int((y1 + y2) / 2))
        elif 80 < angle < 100: # Függőleges (90 fok környéke)
            verticals.append(int((x1 + x2) / 2))
    
    # Csoportosítás (Cluster): Ha több vonal van egymás mellett 15 pixelen belül, átlagoljuk
    def cluster(points):
        if not points: return []
        points.sort()
        groups = []
        if len(points) > 0:
            current_group = [points[0]]
            for i in range(1, len(points)):
                if points[i] - points[i-1] < 15: # 15 pixel a küszöb
                    current_group.append(points[i])
                else:
                    groups.append(int(np.mean(current_group)))
                    current_group = [points[i]]
            groups.append(int(np.mean(current_group)))
        return groups

    return cluster(horizontals), cluster(verticals)

# --- FELDOLGOZÁS ---

gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
# Próbáljuk meg a Gauss-blur-t, hogy a fa erezete ne zavarjon
blurred = cv.GaussianBlur(gray_warped, (5, 5), 0)
edges = cv.Canny(blurred, 50, 150)

lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=20)

h_lines, v_lines = filter_and_cluster_lines(lines)

# Kirajzolás az ellenőrzéshez
result_img = warped.copy()

for y in h_lines:
    cv.line(result_img, (0, y), (800, y), (0, 255, 0), 2) # Zöld = tiszta vízszintes
for x in v_lines:
    cv.line(result_img, (x, 0), (x, 800), (255, 0, 0), 2) # Kék = tiszta függőleges

print(f"Talált tiszta vonalak: Vízszintes: {len(h_lines)}, Függőleges: {len(v_lines)}")

# --- METSZÉSPONTOK ---
# Ha megvan a 9-9 vonal (vagy legalább a legtöbb), kiszámoljuk a pontokat
points = []
if len(h_lines) >= 2 and len(v_lines) >= 2:
    for y in h_lines:
        for x in v_lines:
            points.append([x, y]) # Lista formátum a transzformációhoz
            cv.circle(result_img, (x, y), 5, (0, 0, 255), -1)

cv.imshow("Tisztitott Racs", result_img)
cv.waitKey(0)
cv.destroyAllWindows()

# --- 4. VISSZAVETÍTÉS ÉS MEGJELENÍTÉS ---
annotated_img = img.copy()

if len(points) > 0:
    M_inv = np.linalg.inv(matrix)
    # OpenCV-nek megfelelő formátum: (1, N, 2)
    pts_to_project = np.array([points], dtype='float32')
    projected_pts = cv.perspectiveTransform(pts_to_project, M_inv)[0]

    for pt in projected_pts:
        px, py = int(pt[0]), int(pt[1])
        cv.circle(annotated_img, (px, py), 10, (0, 0, 255), -1) # Piros pont
        cv.circle(annotated_img, (px, py), 12, (255, 255, 255), 2) # Fehér keret

# Megjelenítés átméretezve, hogy lásd az egészet!
show_resized("Eredeti kep raccsal", annotated_img)
cv.imshow("Kivasalt tabla", warped)
cv.waitKey(0)
cv.destroyAllWindows()



# 2. Rács rekonstrukció az új módszerrel
if len(points) > 10:
    vh, vv = get_lattice_vectors(points)
    if vh is not None and vv is not None:
        # KÉNYSZERÍTJÜK az irányokat:
        # vh legyen az, ami jobbra mutat (x pozitív)
        if vh[0] < 0: vh = -vh
        # vv legyen az, ami lefelé mutat (y pozitív)
        if vv[1] < 0: vv = -vv
        
        # Biztosítsuk, hogy vh a vízszintesebb, vv a függőlegesebb
        if abs(vh[1]) > abs(vh[0]): 
            vh, vv = vv, vh
            
        grid_9x9 = reconstruct_lattice(points, vh, vv)
        
        # Átalakítjuk a grid-et listává a visszavetítéshez
        final_points = grid_9x9.reshape(-1, 2)
        
        # --- VISSZAVETÍTÉS ---
        M_inv = np.linalg.inv(matrix)
        pts_to_project = np.array([final_points], dtype='float32')
        projected_pts = cv.perspectiveTransform(pts_to_project, M_inv)[0]

        annotated_img = img.copy()
        for pt in projected_pts:
            px, py = int(pt[0]), int(pt[1])
            cv.circle(annotated_img, (px, py), 8, (255, 255, 255), -1) # Fehér alap
            cv.circle(annotated_img, (px, py), 5, (0, 255, 0), -1)   # Zöld pont (rekonstruált)

        show_resized("Rekonstrualt racs az eredetin", annotated_img)
        cv.imshow("Kivasalt tabla", warped)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Nem sikerült meghatározni a bázisvektorokat.")


