# main_debug_visual.py
import cv2
import numpy as np
from saddlepoints import extract_saddle_points
from Brutesac import brute_expand
from ultralytics import YOLO

# YOLO modell betöltése
model = YOLO("best_nano.pt")

def process_yolo_mask_and_crop(img):
    # YOLO predikció
    results = model.predict(img, imgsz=640, device='cpu', verbose=False)

    mask = None
    for r in results:
        if hasattr(r, "masks") and r.masks is not None:
            mask = r.masks.data[0].cpu().numpy()
            break

    if mask is None:
        raise RuntimeError("❌ Nincs szegmentált maszk a képen")

    # Resize a kép méretéhez
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 0).astype(np.uint8)

    ys, xs = np.where(mask_bin)
    if len(xs) == 0:
        raise RuntimeError("❌ Üres maszk – nincs sakktábla")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Debug: bounding box a képen
    debug_img = img.copy()
    cv2.rectangle(debug_img, (x0, y0), (x1, y1), (255,0,0), 2)
    imshow_fit("Bounding box debug", debug_img)
    cv2.waitKey(500)

    # Maszkon kívüli részek fekete
    masked_img = np.zeros_like(img)
    masked_img[y0:y1+1, x0:x1+1] = img[y0:y1+1, x0:x1+1] * mask_bin[y0:y1+1, x0:x1+1][:, :, None]
    imshow_fit("Maszkolt kép debug", masked_img)
    cv2.waitKey(500)

    roi = masked_img[y0:y1+1, x0:x1+1]
    roi_mask = mask_bin[y0:y1+1, x0:x1+1]

    return roi, roi_mask, (x0, y0), masked_img


def imshow_fit(winname, img, max_w=1400, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(winname, img)

def main(img_path):
    img = cv2.imread(img_path)

    try:
        roi, roi_mask, (ox, oy), masked_full = process_yolo_mask_and_crop(img)
    except RuntimeError as e:
        print(e)
        return

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Saddle pontok keresése
    saddle_pts = extract_saddle_points(gray, roi_mask)
    print(f"Talált saddle pontok: {len(saddle_pts)}")

    # Debug: saddle pontok megjelenítése
    debug_overlay = masked_full.copy()
    for p in saddle_pts:
        cv2.circle(debug_overlay, (p[0]+ox, p[1]+oy), 3, (0,255,255), -1)
    imshow_fit("Saddle pontok debug", debug_overlay)
    cv2.waitKey(500)

    # Kezdeti H a bounding box alapján
    ys, xs = np.where(roi_mask)
    quad = np.array([
        [xs.min(), ys.min()],
        [xs.max(), ys.min()],
        [xs.max(), ys.max()],
        [xs.min(), ys.max()]
    ], dtype=np.float32)
    ideal_quad = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    H_init = cv2.getPerspectiveTransform(ideal_quad, quad)

    # Brute expand
    result = brute_expand(saddle_pts, H_init)

    if result is None:
        print("❌ Nem sikerült rácsot építeni")
        imshow_fit("Rács hiba debug", debug_overlay)
        cv2.waitKey(0)
        return

    H, ideal, snapped, good = result
    print(f"Jó pontok száma: {np.sum(good)}")

    # Debug: snapped pontok és grid overlay
    debug_overlay2 = img.copy()
    for p in snapped:
        cv2.circle(debug_overlay2, (int(p[0]+ox), int(p[1]+oy)), 3, (0,0,255), -1)
    for p in snapped[good]:
        cv2.circle(debug_overlay2, (int(p[0]+ox), int(p[1]+oy)), 6, (0,255,0), 1)

    # Grid vonalak overlay
    xs_grid = snapped[:,0] + ox
    ys_grid = snapped[:,1] + oy
    for x in xs_grid:
        cv2.line(debug_overlay2, (int(x), 0), (int(x), img.shape[0]), (255,0,0), 1)
    for y in ys_grid:
        cv2.line(debug_overlay2, (0, int(y)), (img.shape[1], int(y)), (255,0,0), 1)

    imshow_fit("Rács és pontok debug", debug_overlay2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✅ Folyamat befejezve")

if __name__ == "__main__":
    main("../tanito_kepek/80.jpg")
