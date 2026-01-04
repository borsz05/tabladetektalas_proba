import cv2
import numpy as np
import time

from detectchessboards import (
    findChessboard,
    generateNewBestFit,
    getBestLines,
    getUnwarpedPoints,
    getBoardOutline,
    extract_squares_from_warp,
    centers_to_image
)

from occupancy_model import OccupancyModel

# -----------------------------
# PARAMÉTEREK
# -----------------------------
VIDEO_IN  = "../test_video_1.mp4"
VIDEO_OUT = "chess_detected_occupancy.mp4"

DETECT_EVERY_N = 20  # ennyi frame-enként új detekt
MIN_TRACK_POINTS = 20

# Fontos: egyezzen az export/training logikával!
INNER_PAD_RATIO = 0.06   # ajánlott: kisebb pad -> több info a bbox-ban
CONTEXT = 0.50           # ugyanaz, mint exportnál (bbox +50%)


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

    occ_model = OccupancyModel("occupancy_resnet18_best.pt")

    frame_idx = 0
    prev_gray = None
    tracked_pts = None

    # Stabil “ref” pontok: legutóbbi detektáláskor rögzített rácspontok
    tracked_pts_ref = None

    # Centerek
    centers_img_last = None     # 8x8 középpontok aktuális képen
    centers_img_ref = None      # 8x8 középpontok a ref frame-en (detektáláskor)

    # Label + conf a legutóbbi detektálásból
    labels_last = None          # 8x8: "empty"/"occupied"
    confs_last  = None          # 8x8: float

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
            or len(tracked_pts) < MIN_TRACK_POINTS
        )

        # ----------------------------------
        # DETEKTÁLÁS (lassú ág)
        # ----------------------------------
        if need_detect:
            print(f"[{frame_idx}] Detektálás...")

            M, ideal_grid, grid_next, grid_good, spts = findChessboard(gray.copy())

            if M is not None:
                # Finomítjuk a M-et, hogy stabil warp legyen
                M = generateNewBestFit((ideal_grid + 8) * 32, grid_next, grid_good)

                img_warp = cv2.warpPerspective(
                    gray, M, (17 * 32, 17 * 32),
                    flags=cv2.WARP_INVERSE_MAP
                )

                best_x, best_y = getBestLines(img_warp)

                # Tracking pontok (rácsvonal metszések) képtérben
                pts = getUnwarpedPoints(best_x, best_y, M)  # Nx2
                tracked_pts = pts.reshape(-1, 1, 2).astype(np.float32)
                tracked_pts_ref = tracked_pts.copy()

                prev_gray = gray.copy()

                # Mező bbox-ok warp térben + centerek warp térben
                squares, bbox_warp, centers_warp, _ = extract_squares_from_warp(
                    img_warp, best_x, best_y, inner_pad_ratio=INNER_PAD_RATIO
                )

                # Centerek képtérben
                centers_img_det = centers_to_image(centers_warp, M)

                # Label occupancy CNN-nel: bbox+context crop-on
                labels = [[None]*8 for _ in range(8)]
                confs  = [[0.0]*8 for _ in range(8)]

                for r in range(8):
                    for c in range(8):
                        roi = crop_with_context(img_warp, bbox_warp[r][c], context=CONTEXT)
                        lab, conf = occ_model.predict_square(roi)
                        labels[r][c] = lab
                        confs[r][c]  = conf

                # Frissítjük a “ref”-et és “last”-ot
                centers_img_last = centers_img_det
                centers_img_ref  = centers_img_det
                labels_last = labels
                confs_last  = confs

            else:
                # fail -> mindent nullázunk
                tracked_pts = None
                tracked_pts_ref = None
                centers_img_last = None
                centers_img_ref = None
                labels_last = None
                confs_last = None

        # ----------------------------------
        # TRACKING (gyors ág)
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

            tracked_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
            prev_gray = gray.copy()

            # Centerek mozgatása homográfiával (ref -> current)
            if tracked_pts_ref is not None and centers_img_ref is not None and len(good_old) >= 8:
                H, inliers = cv2.findHomography(tracked_pts_ref, tracked_pts, cv2.RANSAC, 3.0)
                if H is not None:
                    flat = np.array(
                        [centers_img_ref[r][c] for r in range(8) for c in range(8)],
                        dtype=np.float32
                    ).reshape(-1, 1, 2)
                    flat2 = cv2.perspectiveTransform(flat, H).reshape(-1, 2)

                    new_centers = [[None]*8 for _ in range(8)]
                    k = 0
                    for r in range(8):
                        for c in range(8):
                            new_centers[r][c] = (float(flat2[k, 0]), float(flat2[k, 1]))
                            k += 1
                    centers_img_last = new_centers

        # ----------------------------------
        # RAJZOLÁS
        # ----------------------------------
        # rácspontok (piros)
        #if tracked_pts is not None:
        #    for p in tracked_pts:
        #        x, y = p.ravel()
        #        cv2.circle(draw_frame, (int(x), int(y)), 2, (0, 0, 255), -1)

        # mezőközéppont + occupancy
        if centers_img_last is not None and labels_last is not None:
            for r in range(8):
                for c in range(8):
                    x, y = centers_img_last[r][c]
                    lab = labels_last[r][c]
                    conf = confs_last[r][c] if confs_last is not None else None

                    if lab == "occupied":
                        color = (0, 0, 255)   # piros
                        txt = "O"
                    else:
                        color = (0, 255, 0)   # zöld
                        txt = "E"

                    cv2.circle(draw_frame, (int(x), int(y)), 6, color, -1)

                    #if conf is not None:
                    #   cv2.putText(draw_frame, f"{txt}:{conf:.2f}",
                    #              (int(x)+6, int(y)-6),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    #            (255, 255, 255), 2, cv2.LINE_AA)

        out.write(draw_frame)
        frame_idx += 1

    cap.release()
    out.release()

    t1 = time.time()
    print(f"Kész. Feldolgozási idő: {t1 - t0:.2f} s")
    print(f"Kimenet: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
