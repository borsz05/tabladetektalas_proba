import cv2
import numpy as np
import time

# A SAJÁT FÜGGVÉNYEIDET HASZNÁLJUK
# feltételezzük, hogy ezek importálhatók
from detectchessboards import (
    loadImage,
    findChessboard,
    generateNewBestFit,
    getBestLines,
    getUnwarpedPoints,
    getBoardOutline,
    extract_squares_from_warp,
    centers_to_image
)
from square_classify import classify_square_3class
from occupancy_model import OccupancyModel

# -----------------------------
# PARAMÉTEREK
# -----------------------------
VIDEO_IN  = "../test_video_1.mp4"
VIDEO_OUT = "chess_detected_with_pieces.mp4"

DETECT_EVERY_N = 20
MAX_TRACK_POINTS = 200

def crop_with_context(img_warp, bbox, context=0.50):
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    dx = int(round(bw * context))
    dy = int(round(bh * context))

    X0 = max(0, x0 - dx)
    Y0 = max(0, y0 - dy)
    X1 = min(img_warp.shape[1], x1 + dx)
    Y1 = min(img_warp.shape[0], y1 + dy)

    return img_warp[Y0:Y1, X0:X1].copy()

# -----------------------------
def main():

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a videót")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.2f} fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

    frame_idx = 0

    prev_gray = None
    tracked_pts = None
    M_last = None
    board_outline_last = None
    centers_warp_last = None   # 64 center warp térben (8x8 lista)
    centers_img_last = None    # 64 center image térben (8x8 lista) - ezt rajzolod
    tracked_pts_prev = None    # előző frame rácspontok (Nx1x2)
    centers_img_prev = None
    tracked_pts_ref = None
    centers_img_ref = None
    labels_last = None
    confs_last = None


    occ_model = OccupancyModel("occupancy_resnet18_best.pt")

    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        draw_frame = frame.copy()

        need_detect = (
            frame_idx % DETECT_EVERY_N == 0
            or tracked_pts is None
            or len(tracked_pts) < 20
        )

        # ----------------------------------
        # DETEKTÁLÁS
        # ----------------------------------
        if need_detect:
            print(f"[{frame_idx}] Detektálás...")
            # float32-re nincs szükség, findChessboard igényli a uint8-at
            centers_img_prev = centers_img_last   # <-- EZT IDE

            img_uint8 = gray.copy()
            M, ideal_grid, grid_next, grid_good, spts = findChessboard(img_uint8)

            if M is not None:
                M = generateNewBestFit(
                    (ideal_grid + 8) * 32,
                    grid_next,
                    grid_good
                )

                img_warp = cv2.warpPerspective(
                    img_uint8, M, (17 * 32, 17 * 32),
                    flags=cv2.WARP_INVERSE_MAP
                )

                best_x, best_y = getBestLines(img_warp)
                tracked_pts = getUnwarpedPoints(best_x, best_y, M)
                board_outline_last = getBoardOutline(best_x, best_y, M)
                M_last = M

                CONTEXT = 0.50  # ugyanaz, mint exportnál (ha ott 0.50 volt)

                squares, bbox_warp, centers_warp = extract_squares_from_warp(img_warp, best_x, best_y)
                labels = [[None]*8 for _ in range(8)]
                confs  = [[0.0]*8 for _ in range(8)]

                for r in range(8):
                    for c in range(8):
                        roi = crop_with_context(img_warp, bbox_warp[r][c], context=CONTEXT)
                        lab, conf = occ_model.predict_square(roi)
                        labels[r][c] = lab
                        confs[r][c]  = conf

                labels_last = labels
                confs_last = confs


                labels = [[None]*8 for _ in range(8)]
                debugs = [[None]*8 for _ in range(8)]

                for r in range(8):
                    for c in range(8):
                        lab, dbg = classify_square_3class(squares[r][c], occ_thresh=0.06)
                        labels[r][c] = lab
                        debugs[r][c] = dbg

                centers_warp_last = centers_warp
                # ÚJ detektált centerek külön változóba
                centers_img_det = centers_to_image(centers_warp, M_last)

                labels_last = labels
                
                # --- Ugrás-szűrés (gate): ha túl nagy a különbség, dobd el a detektálást ---
                if centers_img_det is not None and centers_img_last is not None:
                    old = np.array([centers_img_last[r][c] for r in range(8) for c in range(8)], dtype=np.float32)
                    new = np.array([centers_img_det[r][c]  for r in range(8) for c in range(8)], dtype=np.float32)
                    mean_dist = float(np.mean(np.linalg.norm(new - old, axis=1)))

                    # Küszöb: hangolható. 30-50 px jó indulás.
                    if mean_dist > 40.0:
                        # túl nagy ugrás → NE fogadd el
                        centers_img_det = None

                # Ha átment a gate-en, akkor jöhet a blending és felülírás
                if centers_img_det is not None:
                    # --- Blendelés, hogy ne ugráljon detektáláskor ---
                    alpha = 0.7
                    if centers_img_last is not None:
                        blended = [[None]*8 for _ in range(8)]
                        for r in range(8):
                            for c in range(8):
                                x0, y0 = centers_img_last[r][c]   # régi
                                x1, y1 = centers_img_det[r][c]    # új detektált
                                blended[r][c] = (alpha*x1 + (1-alpha)*x0,
                                                alpha*y1 + (1-alpha)*y0)
                        centers_img_last = blended
                    else:
                        centers_img_last = centers_img_det


                tracked_pts = tracked_pts.reshape(-1, 1, 2).astype(np.float32)
                tracked_pts_prev = tracked_pts.copy()
                tracked_pts_ref = tracked_pts.copy()      # <-- REF
                centers_img_ref = centers_img_last        # <-- REF (a detektált/blendelt)
                prev_gray = gray.copy()
            else:
                tracked_pts = None
                M_last = None
                centers_img_last = None


        # ----------------------------------
        # TRACKING (OPTICAL FLOW)
        # ----------------------------------
        else:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, tracked_pts, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            good_new = next_pts[status == 1]
            good_old = tracked_pts[status == 1]

            # frissítsük a tracked pontokat
            tracked_pts = good_new.reshape(-1, 1, 2).astype(np.float32)

            # H-t ref->current-ből számoljuk
            if tracked_pts_ref is not None and centers_img_ref is not None:
                # Válassz inlier pontokat a ref-ből és az aktuálisból ugyanazon indexekkel
                # Ehhez egyszerűen a teljes tracked_pts_ref és tracked_pts használható,
                # mert ugyanannyi pont van (ha nem, akkor sync kell).
                H, inliers = cv2.findHomography(tracked_pts_ref, tracked_pts, cv2.RANSAC, 3.0)

                if H is not None:
                    flat = np.array([centers_img_ref[r][c] for r in range(8) for c in range(8)], np.float32).reshape(-1,1,2)
                    flat2 = cv2.perspectiveTransform(flat, H).reshape(-1,2)

                    new_centers = [[None]*8 for _ in range(8)]
                    k=0
                    for r in range(8):
                        for c in range(8):
                            new_centers[r][c] = (float(flat2[k,0]), float(flat2[k,1]))
                            k+=1
                    centers_img_last = new_centers

            prev_gray = gray.copy()

        # ----------------------------------
        # RAJZOLÁS
        # ----------------------------------
        if tracked_pts is not None:
            for p in tracked_pts:
                x, y = p.ravel()
                cv2.circle(draw_frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        if centers_img_last is not None:
            for r in range(8):
                for c in range(8):
                    x, y = centers_img_last[r][c]
                    cv2.circle(draw_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        if centers_img_last is not None and labels_last is not None:
            for r in range(8):
                for c in range(8):
                    x, y = centers_img_last[r][c]
                    lab = labels_last[r][c]

                    if lab == "occupied":
                        color = (0, 0, 255)   # piros
                        txt = "O"
                    else:
                        color = (0, 255, 0)   # zöld
                        txt = "E"

                    cv2.circle(draw_frame, (int(x), int(y)), 6, color, -1)

                    # opcionális: confidence kiírás
                    if confs_last is not None:
                        conf = confs_last[r][c]
                        cv2.putText(draw_frame, f"{txt}:{conf:.2f}", (int(x)+6, int(y)-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(draw_frame, txt, (int(x)+6, int(y)-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        out.write(draw_frame)
        frame_idx += 1

    cap.release()
    out.release()

    t1 = time.time()
    print(f"Kész. Feldolgozási idő: {t1 - t0:.2f} s")
    print(f"Kimenet: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
