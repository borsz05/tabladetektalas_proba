import cv2
import numpy as np

def _center_mask(h, w, rx=0.30, ry=0.28, y_bias=0.18):
    """
    Elliptikus maszk a mező közepére, enyhén lejjebb tolva (talp-zóna felé).
    rx, ry: sugár arányok a szélesség/magasság szerint
    y_bias: pozitív -> lejjebb tolja a maszk közepét
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(w * 0.5)
    cy = int(h * (0.5 + y_bias))
    axes = (max(1, int(w * rx)), max(1, int(h * ry)))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask

def classify_square_3class(roi_gray,
                           occ_thresh=0.06,
                           white_black_thresh=None):
    """
    roi_gray: egy mező ROI (szürke, uint8)
    Returns: label, debug_dict
      label ∈ {"empty", "white", "black"}
    """

    if roi_gray is None or roi_gray.size == 0:
        return "empty", {"reason": "empty_roi"}

    g = roi_gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    h, w = g.shape[:2]

    # 1) Talp-zóna: alsó 60% + elliptikus középmaszk metszete
    bottom = np.zeros((h, w), np.uint8)
    y0 = int(h * 0.40)
    bottom[y0:, :] = 255

    ell = _center_mask(h, w, rx=0.38, ry=0.45, y_bias=0.12)
    zone = cv2.bitwise_and(bottom, ell)  # ez lesz a "megbízható" régió

    # 2) Foreground becslés: élek + textúra (morph gradient)
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # adaptív küszöb a grad-on -> "kiemelkedő" rész
    # (tapasztalat: robust fényváltozásra)
    _, fg = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # csak a zónában számítson
    fg = cv2.bitwise_and(fg, zone)

    # tisztítás
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=1)

    fg_ratio = float(np.count_nonzero(fg)) / max(1, np.count_nonzero(zone))

    # 3) Üres/tele döntés
    if fg_ratio < occ_thresh:
        return "empty", {"fg_ratio": fg_ratio}

    # 4) Fehér/fekete döntés: foreground pixelek intenzitása
    fg_pixels = g[fg > 0]
    if fg_pixels.size < 20:
        return "empty", {"fg_ratio": fg_ratio, "reason": "too_few_fg"}

    mean_int = float(np.mean(fg_pixels))
    med_int = float(np.median(fg_pixels))

    # ha nincs megadva küszöb, csináljunk egy egyszerűt a mező egészéhez viszonyítva
    # (ez nem tökéletes, de meglepően működő baseline)
    if white_black_thresh is None:
        # a mező "zóna" átlagához képest
        zone_pixels = g[zone > 0]
        base = float(np.mean(zone_pixels)) if zone_pixels.size else 128.0
        # ha a foreground világosabb, inkább fehér
        white_black_thresh = base + 5.0

    label = "white" if med_int > white_black_thresh else "black"
    return label, {
        "fg_ratio": fg_ratio,
        "mean_int": mean_int,
        "med_int": med_int,
        "wb_thresh": white_black_thresh
    }
