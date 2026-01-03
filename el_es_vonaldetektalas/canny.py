import cv2
import numpy as np

def filter_lines(lines, rho_threshold=40, theta_threshold=np.pi/18):
    """Egyesíti az egymáshoz túl közel lévő vonalakat."""
    if lines is None or len(lines) == 0: return []
    
    # Rendszerezés rho szerint
    lines = sorted(lines, key=lambda x: x[0])
    filtered_lines = []
    
    if len(lines) > 0:
        filtered_lines.append(lines[0])
        for i in range(1, len(lines)):
            rho, theta = lines[i]
            prev_rho, prev_theta = filtered_lines[-1]
            
            # Ha a távolság vagy a szög jelentősen eltér, ez egy új vonal
            if abs(rho - prev_rho) > rho_threshold:
                filtered_lines.append(lines[i])
                
    return filtered_lines

# 1. Betöltés és élek (ahogy eddig csináltad)
img = cv2.imread('../sakktabla2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)

# 2. Vonalak detektálása
raw_lines = cv2.HoughLines(edges, 1, np.pi/180, 120)

horizontals = []
verticals = []

if raw_lines is not None:
    for line in raw_lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        if angle < 45 or angle > 135:
            verticals.append((rho, theta))
        else:
            horizontals.append((rho, theta))

# 3. SZŰRÉS: Ne legyen 50 vonalunk egy élre
v_lines = filter_lines(verticals)
h_lines = filter_lines(horizontals)

print(f"Talált függőleges vonalak: {len(v_lines)}")
print(f"Talált vízszintes vonalak: {len(h_lines)}")

# 4. Metszéspontok és Homográfia (csak ha megvan a minimum 9-9 vonal)
def get_intersect(l1, l2):
    rho1, theta1 = l1
    rho2, theta2 = l2
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    try:
        res = np.linalg.solve(A, b)
        return int(res[0].item()), int(res[1].item())
    except: return None

points = []
for h in h_lines:
    for v in v_lines:
        p = get_intersect(h, v)
        if p:
            points.append(p)
            cv2.circle(img, p, 5, (0, 0, 255), -1)

# Eredmény megjelenítése
cv2.imshow('Chessboard Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# --- 1. A pontok megtisztítása és a 4 sarok keresése ---
# Alakítsuk a pontokat egyetlen NumPy tömbbé
all_points = np.array(points, dtype=np.float32)

# Keressük meg a pontfelhő köré írható legszűkebb idomot (Konvex burok)
hull = cv2.convexHull(all_points)

# Egyszerűsítsük le az idomot, amíg csak 4 sarka nem marad
# Ez a függvény addig finomít, amíg egy négyszöget nem kapunk
epsilon = 0.02 * cv2.arcLength(hull, True)
approx_corners = cv2.approxPolyDP(hull, epsilon, True)

# Ha több vagy kevesebb, mint 4 pontot talált, vegyük a 4 legszélső pontot
if len(approx_corners) != 4:
    # Alternatív megoldás: a pontok összege és különbsége alapján
    # (x+y min/max és x-y min/max megadja a 4 sarkot)
    s = all_points.sum(axis=1)
    diff = np.diff(all_points, axis=1)
    
    top_left = all_points[np.argmin(s)]
    bottom_right = all_points[np.argmax(s)]
    top_right = all_points[np.argmin(diff)]
    bottom_left = all_points[np.argmax(diff)]
    
    board_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
else:
    # A megtalált 4 pontot sorbarendezzük (óra járása szerint)
    board_corners = approx_corners.reshape(4, 2).astype("float32")

# --- 2. Homográfia (Kivasalás) ---
# Határozzuk meg a cél-kép méretét (legyen pl. 800x800 pixel)
side = 800
destination_corners = np.array([
    [0, 0],
    [side - 1, 0],
    [side - 1, side - 1],
    [0, side - 1]
], dtype="float32")

# Számítsuk ki a transzformációs mátrixot
# Mivel a board_corners sorrendje fontos, egy kis rendezés kellhet
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

ordered_src = order_points(board_corners)
M = cv2.getPerspectiveTransform(ordered_src, destination_corners)

# Hajtsuk végre a kivasalást
warped = cv2.warpPerspective(img, M, (side, side))

# --- 3. Megjelenítés ---
cv2.imshow('Kivasalt tabla', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

