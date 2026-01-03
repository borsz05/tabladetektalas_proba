import os
import cv2
import numpy as np

from detectchessboards import (
    findChessboard,
    generateNewBestFit,
    getBestLines,
    extract_squares_from_warp,
)

# -----------------------------
# Beállítások
# -----------------------------
VIDEO_IN = "../test_video_1.mp4"
OUT_DIR  = "data_occupancy"   # train/val/empty|occupied

DETECT_EVERY_N = 10           # csak ezekből mentünk (stabilabb)
MAX_SAVED_TRAIN = 6000        # osztályonkénti limit
MAX_SAVED_VAL   = 1200

# kontextus és crop
INNER_PAD_RATIO = 0.06        # kisebb padding, több jel
CONTEXT = 0.50                # +50% bbox növelés minden irányban (chesscog stílus) :contentReference[oaicite:3]{index=3}
IMG_SIZE = 100                # chesscog-ben gyakori az 100×100 :contentReference[oaicite:4]{index=4}

TRAIN_RATIO = 0.85

# -----------------------------
# ORIENTÁCIÓ (BELŐVÉS)
# -----------------------------
# 0,1,2,3 = 0°,90°,180°,270° forgatás a rácson
ORIENT = 1
FLIP_X = False
FLIP_Y = False


def ensure_dirs():
    for split in ["train", "val"]:
        for cls in ["empty", "occupied"]:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)


def rc_transform(r, c):
    rr, cc = r, c
    if ORIENT == 0:
        rr, cc = rr, cc
    elif ORIENT == 1:
        rr, cc = cc, 7 - rr
    elif ORIENT == 2:
        rr, cc = 7 - rr, 7 - cc
    elif ORIENT == 3:
        rr, cc = 7 - cc, rr
    else:
        raise ValueError("ORIENT must be 0,1,2,3")

    if FLIP_X:
        cc = 7 - cc
    if FLIP_Y:
        rr = 7 - rr
    return rr, cc


def occupancy_from_initial_position(r, c):
    """
    Kezdőállás occupancy:
    - a két 'saját' szélső 2 sor foglalt (rank1-2, rank7-8)
    - középső 4 sor üres
    """
    rr, cc = rc_transform(r, c)

    # konvenció: rr=7 -> rank1, rr=6 -> rank2, rr=1 -> rank7, rr=0 -> rank8
    if rr in [6, 7, 0, 1]:
        return "occupied"
    return "empty"


def expand_bbox(bbox, w, h, context=0.5):
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    dx = int(round(bw * context))
    dy = int(round(bh * context))
    X0 = max(0, x0 - dx)
    Y0 = max(0, y0 - dy)
    X1 = min(w, x1 + dx)
    Y1 = min(h, y1 + dy)
    return (X0, Y0, X1, Y1)


def to_rgb_and_resize(roi_gray, size=100):
    if roi_gray.ndim == 2:
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    else:
        roi_rgb = roi_gray
    roi_rgb = cv2.resize(roi_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return roi_rgb


def save_sample(img, split, cls, idx):
    path = os.path.join(OUT_DIR, split, cls, f"{cls}_{idx:06d}.png")
    cv2.imwrite(path, img)


def make_orientation_debug_image(img_warp, bbox_warp):
    """
    Debug: a warp képre ráírjuk, hogy a mi logikánk szerint
    mely mezők occupied/empty a kezdőállásban.
    """
    dbg = cv2.cvtColor(img_warp, cv2.COLOR_GRAY2BGR)
    for r in range(8):
        for c in range(8):
            x0, y0, x1, y1 = bbox_warp[r][c]
            cx = int((x0 + x1) * 0.5)
            cy = int((y0 + y1) * 0.5)
            lab = occupancy_from_initial_position(r, c)
            txt = "O" if lab == "occupied" else "E"
            cv2.putText(dbg, txt, (cx - 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return dbg


def main():
    ensure_dirs()

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót:", VIDEO_IN)
        return

    saved_train = {"empty": 0, "occupied": 0}
    saved_val   = {"empty": 0, "occupied": 0}

    frame_idx = 0
    debug_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx % DETECT_EVERY_N != 0:
            frame_idx += 1
            continue

        M, ideal_grid, grid_next, grid_good, spts = findChessboard(gray.copy())
        if M is None:
            frame_idx += 1
            continue

        M = generateNewBestFit((ideal_grid + 8) * 32, grid_next, grid_good)

        img_warp = cv2.warpPerspective(
            gray, M, (17 * 32, 17 * 32),
            flags=cv2.WARP_INVERSE_MAP
        )

        best_x, best_y = getBestLines(img_warp)
        squares, bbox_warp, centers_warp, _ = extract_squares_from_warp(
            img_warp, best_x, best_y, inner_pad_ratio=INNER_PAD_RATIO
        )

        Hh, Ww = img_warp.shape[:2]

        if not debug_saved:
            dbg = make_orientation_debug_image(img_warp, bbox_warp)
            cv2.imwrite(os.path.join(OUT_DIR, "orientation_debug.png"), dbg)
            debug_saved = True
            print("Mentve:", os.path.join(OUT_DIR, "orientation_debug.png"))
            print("Ha az O/E minta nem egyezik a valós kezdőállással, állíts ORIENT/FLIP-et.")

        for r in range(8):
            for c in range(8):
                cls = occupancy_from_initial_position(r, c)

                split = "train" if np.random.rand() < TRAIN_RATIO else "val"

                if split == "train" and saved_train[cls] >= MAX_SAVED_TRAIN:
                    continue
                if split == "val" and saved_val[cls] >= MAX_SAVED_VAL:
                    continue

                x0, y0, x1, y1 = bbox_warp[r][c]
                X0, Y0, X1, Y1 = expand_bbox((x0, y0, x1, y1), Ww, Hh, context=CONTEXT)
                roi = img_warp[Y0:Y1, X0:X1].copy()

                roi100 = to_rgb_and_resize(roi, size=IMG_SIZE)

                if split == "train":
                    save_sample(roi100, "train", cls, saved_train[cls])
                    saved_train[cls] += 1
                else:
                    save_sample(roi100, "val", cls, saved_val[cls])
                    saved_val[cls] += 1

        if frame_idx % (DETECT_EVERY_N * 10) == 0:
            print(f"[{frame_idx}] saved train={saved_train} val={saved_val}")

        frame_idx += 1

        done_train = all(saved_train[k] >= MAX_SAVED_TRAIN for k in saved_train)
        done_val   = all(saved_val[k]   >= MAX_SAVED_VAL   for k in saved_val)
        if done_train and done_val:
            print("Elég adat megvan, stop.")
            break

    cap.release()
    print("Kész. Mentett train:", saved_train, "val:", saved_val)
    print("Dataset mappa:", OUT_DIR)


if __name__ == "__main__":
    main()
