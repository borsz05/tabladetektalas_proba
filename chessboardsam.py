import cv2
import numpy as np
import glob

# --- CONFIG ---
video_path = 'test_video_1.mp4'
yolo_mask_path = 'best.pt'  # YOLO model súlya
BOARD_SIZE = (8, 8)  # Sakktábla mérete

# --- HELPER FUNKCIÓK ---
def get_saddle_points(gray):
    """Sobel deriváltakból saddle pontok detektálása."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
    gyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
    gxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)
    saddle = -gxx * gyy + gxy**2
    # Non-maximum suppression
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(saddle, kernel)
    saddle[saddle < dilated] = 0
    pts = np.argwhere(saddle > 0)
    return pts

def warp_perspective_corners(corners, target_size=(800, 800)):
    """Negy corner-ból homográfia számítása és kép warp."""
    dst = np.array([[0,0], [target_size[0],0], [target_size[0], target_size[1]], [0, target_size[1]]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return M

def extract_grid(M, size=(8,8)):
    """Homográfiával generálunk rendezett grid pontokat."""
    grid = []
    step_x = 1.0 / size[0]
    step_y = 1.0 / size[1]
    for i in range(size[1]+1):
        for j in range(size[0]+1):
            pt = np.array([j*step_x*800, i*step_y*800, 1])
            pt_warp = M @ pt
            pt_warp /= pt_warp[2]
            grid.append(pt_warp[:2])
    return np.array(grid)

# --- VIDEÓ FOLYAMAT ---
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO maszkolás (itt csak placeholder, helyettesítsd a saját detektálóddal)
    # mask = run_yolo(frame) -> 0/1 maszk
    # Jelen példa: sima gray threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Maszk alkalmazása
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Saddle pontok
    saddle_pts = get_saddle_points(masked_gray)
    if len(saddle_pts) < 4:
        print(f"Frame {frame_idx}: kevés pont detektálva")
        frame_idx += 1
        continue

    # Corner pontok kiválasztása (legtávolabbi 4 pont)
    pts = saddle_pts
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1).flatten()
    corners = np.array([
        pts[np.argmin(sum_pts)],  # top-left
        pts[np.argmin(diff_pts)], # top-right
        pts[np.argmax(sum_pts)],  # bottom-right
        pts[np.argmax(diff_pts)], # bottom-left
    ], dtype=np.float32)

    # Homográfia és grid
    M = warp_perspective_corners(corners)
    grid = extract_grid(M, size=BOARD_SIZE)

    # Vizualizáció
    for g in grid:
        cv2.circle(frame, tuple(g.astype(int)), 3, (0,0,255), -1)
    cv2.imshow('grid', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
