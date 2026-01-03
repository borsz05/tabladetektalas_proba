import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('../sakktabla.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Deriváltak
Ix = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
Iy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

Ixx = cv.Sobel(Ix, cv.CV_64F, 1, 0, ksize=3)
Iyy = cv.Sobel(Iy, cv.CV_64F, 0, 1, ksize=3)
Ixy = cv.Sobel(Ix, cv.CV_64F, 0, 1, ksize=3)

# Hessian determináns
detH = Ixx * Iyy - Ixy**2

# Erősség (kontraszt)
response = np.abs(detH)

kernel_size = 10
local_max = cv.dilate(response, np.ones((kernel_size, kernel_size)))

nms_mask = (response == local_max)

# Küszöbök (EZEK FONTOSAK)
det_thresh = -5000   # nyeregpont kell
resp_thresh =1000000    # legyen erős

mask = (detH < det_thresh) & (response > resp_thresh) & nms_mask

ys, xs = np.where(mask)

# Pontlista (x, y) formában
points = np.column_stack((xs, ys)).astype(np.float64)


# Vizualizáció
vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
for x, y in zip(xs, ys):
    cv.circle(vis, (x, y), 2, (0,255,0), -1)

cv.imshow("saddle corners", vis)
cv.waitKey(0)




# === kNN vektorok ===
diffs = points[:, None, :] - points[None, :, :]
dists = np.linalg.norm(diffs, axis=2)

k = 8
idx = np.argsort(dists, axis=1)[:, 1:k+1]
vectors = np.take_along_axis(diffs, idx[:, :, None], axis=1)

# összelapítás hisztogramhoz
vecs = vectors.reshape(-1, 2)

# =========================
# + HISZTOGRAM + MEGJELENÍTÉS
# =========================
dx = vecs[:, 0]
dy = vecs[:, 1]

# (opcionális) szűrés: ne legyen benne 0-hoz közeli és túl nagy vektor
max_dist = 150.0
dist = np.hypot(dx, dy)
mask = (dist > 2.0) & (dist < max_dist)
dx_f = dx[mask]
dy_f = dy[mask]

# 2D hisztogram (accumulator)
bins = 180
H, xedges, yedges = np.histogram2d(
    dx_f, dy_f,
    bins=bins,
    range=[[-max_dist, max_dist], [-max_dist, max_dist]]
)

# Megjelenítés
plt.figure(figsize=(8, 7))
plt.imshow(
    H.T, origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
)
plt.title("2D elmozdulási hisztogram (kNN vektorok)")
plt.xlabel("dx (px)")
plt.ylabel("dy (px)")
plt.axhline(0, linestyle="--", linewidth=0.7)
plt.axvline(0, linestyle="--", linewidth=0.7)
plt.colorbar(label="szavazatok")
plt.show()

# (opcionális) 1D hisztogramok is sokat segítenek a "L" (mezőméret) látásához
plt.figure(figsize=(8, 4))
plt.hist(np.abs(dx_f), bins=120)
plt.title("|dx| eloszlás (szűrt kNN vektorok)")
plt.xlabel("|dx| (px)")
plt.ylabel("db")
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(np.abs(dy_f), bins=120)
plt.title("|dy| eloszlás (szűrt kNN vektorok)")
plt.xlabel("|dy| (px)")
plt.ylabel("db")
plt.show()



from scipy.ndimage import maximum_filter

def get_basis_vectors(H, xedges, yedges, num_peaks=15):
    # 1. Lokális maximumok keresése (Peak Finding)
    # A maximum_filter megnézi minden pixel környezetét
    neighborhood_size = 5
    data_max = maximum_filter(H, neighborhood_size)
    peak_mask = (H == data_max) & (H > 0)
    
    # Csúcsok koordinátáinak és értékeinek kinyerése
    y_idx, x_idx = np.where(peak_mask)
    peak_values = H[y_idx, x_idx]
    
    # Visszaalakítás dx, dy koordinátákká
    peaks = []
    for i in range(len(x_idx)):
        dx_val = (xedges[x_idx[i]] + xedges[x_idx[i]+1]) / 2
        dy_val = (yedges[y_idx[i]] + yedges[y_idx[i]+1]) / 2
        # Kihagyjuk a középső (0,0) pontot
        if np.hypot(dx_val, dy_val) > 20: 
            peaks.append([dx_val, dy_val, peak_values[i]])
            
    # Sorbarendezés erősség szerint
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:num_peaks]
    
    # 2. Harmonikus pontozás (opcionális, de ajánlott)
    # Minden jelöltre megnézzük, van-e csúcs a 2x-es, 3x-os távolságban
    scored_peaks = []
    for p in peaks:
        v = np.array([p[0], p[1]])
        score = p[2]
        for n in [2, 3]: # Harmonikusok ellenőrzése
            target = n * v
            # Megkeressük a legközelebbi csúcsot a többszörösnél
            for other in peaks:
                dist = np.linalg.norm(np.array([other[0], other[1]]) - target)
                if dist < 15: # Ha közel van a várt harmonikushoz
                    score += other[2] * 0.5 # Bónusz pont
        scored_peaks.append((v, score))
    
    scored_peaks = sorted(scored_peaks, key=lambda x: x[1], reverse=True)
    
    # 3. Két egymásra merőleges bázisvektor kiválasztása
    v1 = scored_peaks[0][0]
    v2 = None
    for i in range(1, len(scored_peaks)):
        candidate = scored_peaks[i][0]
        # Skaláris szorzat a merőlegesség ellenőrzéséhez (koszinusz hasonlóság)
        cos_sim = np.abs(np.dot(v1, candidate) / (np.linalg.norm(v1) * np.linalg.norm(candidate)))
        if cos_sim < 0.5: # Ha az elhajlás elég nagy (közel merőleges)
            v2 = candidate
            break
            
    return v1, v2, peaks

# Futtatás
v1, v2, all_peaks = get_basis_vectors(H, xedges, yedges)

print(f"Bázisvektor 1: {v1}")
print(f"Bázisvektor 2: {v2}")

# --- Megjelenítés ---
plt.figure(figsize=(8, 7))
plt.imshow(H.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.scatter([p[0] for p in all_peaks], [p[1] for p in all_peaks], color='red', s=20, label='Csúcs jelöltek')

if v1 is not None:
    plt.arrow(0, 0, v1[0], v1[1], color='yellow', width=2, head_width=8, label='V1 (Bázis)')
if v2 is not None:
    plt.arrow(0, 0, v2[0], v2[1], color='cyan', width=2, head_width=8, label='V2 (Bázis)')

plt.title("Bázisvektorok kiválasztása a hisztogramról")
plt.legend()
plt.show()


def find_best_seed_point(points, v1, v2):
    best_p0 = None
    max_score = -1
    
    # Minden pontot megvizsgálunk, mint lehetséges kezdőpontot
    for p_candidate in points:
        score = 0
        # Kiszámoljuk az összes többi pont relatív helyzetét ehhez a ponthoz képest
        rel_points = points - p_candidate
        
        # Átváltjuk a relatív koordinátákat a v1, v2 bázisrendszerbe
        # Ehhez egy 2x2-es mátrix inverzét használjuk
        basis_matrix = np.array([v1, v2]).T
        try:
            # (dx, dy) = a*v1 + b*v2 -> [a, b] kiszámítása
            coords_in_basis = np.linalg.solve(basis_matrix, rel_points.T).T
            
            # Megnézzük, hány pont esik közel egész számú koordinátákhoz (rácspontokhoz)
            # A hiba az egész számoktól való eltérés
            errors = np.abs(coords_in_basis - np.round(coords_in_basis))
            
            # Akkor jó egy pont, ha a hiba kicsi (pl. < 0.15 egység)
            inliers = np.all(errors < 0.15, axis=1)
            score = np.sum(inliers)
            
            if score > max_score:
                max_score = score
                best_p0 = p_candidate
        except np.linalg.LinAlgError:
            continue
            
    return best_p0

# 1. Kezdőpont megkeresése
p0 = find_best_seed_point(points, v1, v2)

# 2. Egy 9x9-es elméleti rács generálása a kezdőpont köré
# (Egy sakktáblának 9x9 metszéspontja van)
grid_indices = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5))
grid_flat = np.stack(grid_indices, axis=-1).reshape(-1, 2)
theoretical_grid = p0 + grid_flat[:, 0, None] * v1 + grid_flat[:, 1, None] * v2

# --- Megjelenítés ---
plt.figure(figsize=(10, 10))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.scatter(points[:, 0], points[:, 1], c='green', s=10, label='Detektált pontok', alpha=0.5)
plt.scatter(p0[0], p0[1], c='yellow', s=100, marker='*', label='Kezdőpont (P0)', edgecolors='black')
plt.scatter(theoretical_grid[:, 0], theoretical_grid[:, 1], c='red', s=30, marker='x', label='Rekonstruált rács')

plt.title("Rács-rekonstrukció: Elméleti rács illesztése")
plt.legend()
plt.show()


# 1. Párosítsuk az elméleti rácspontokat a valódi detektált pontokkal
image_pts = [] # A képen talált valódi zöld pontok
grid_pts = []  # A hozzájuk tartozó logikai koordináták (pl. 0,0, 0,1...)

for i, t_pt in enumerate(theoretical_grid):
    # Megkeressük a legközelebbi detektált zöld pontot
    dists = np.linalg.norm(points - t_pt, axis=1)
    min_idx = np.argmin(dists)
    
    # Csak akkor fogadjuk el, ha elég közel van (pl. 20 pixelen belül)
    if dists[min_idx] < 25:
        image_pts.append(points[min_idx])
        # A grid_flat-ben tároltuk a logikai (i, j) koordinátákat
        grid_pts.append(grid_flat[i])

image_pts = np.array(image_pts)
grid_pts = np.array(grid_pts)

print(f"Sikeresen párosítva: {len(image_pts)} pont.")

# 2. Homográfia számítása
# A logikai pontokat beszorozzuk a kívánt mezőmérettel (pl. 100 pixel)
dest_size = 800
# Elmozdítjuk a koordinátákat, hogy ne legyenek negatívak (0-tól 8-ig)
logical_offset_pts = (grid_pts + 4) * (dest_size / 8)

H_matrix, mask = cv.findHomography(image_pts, logical_offset_pts, cv.RANSAC, 5.0)

# 3. Kép transzformálása (Warping)
warped_img = cv.warpPerspective(img, H_matrix, (dest_size, dest_size))

# Megjelenítés
plt.figure(figsize=(8,8))
plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
plt.title("Kivasalt, perspektíva-mentes sakktábla")
plt.show()