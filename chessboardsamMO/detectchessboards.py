# -*- coding: utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import scipy.ndimage

np.set_printoptions(suppress=True, linewidth=200)
plt.rcParams['image.cmap'] = 'jet'


# =========================
# Saddle
# =========================

def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)
    return gxx * gyy - gxy ** 2


def nonmax_sup(img, win=10):
    h, w = img.shape
    img_sup = np.zeros_like(img, dtype=np.float64)

    for i, j in np.argwhere(img):
        ta = max(0, i - win)
        tb = min(h, i + win + 1)
        tc = max(0, j - win)
        td = min(w, j + win + 1)

        cell = img[ta:tb, tc:td]
        if img[i, j] == cell.max():
            img_sup[i, j] = img[i, j]

    return img_sup


def pruneSaddle(s):
    thresh = 128
    while np.sum(s > 0) > 10000:
        thresh *= 2
        s[s < thresh] = 0


def getMinSaddleDist(saddle_pts, pt):
    best_dist = None
    best_pt = pt

    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt) ** 2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt

    return best_pt, np.sqrt(best_dist)


# =========================
# Contours
# =========================

def simplifyContours(contours):
    for i in range(len(contours)):
        contours[i] = cv2.approxPolyDP(
            contours[i],
            0.04 * cv2.arcLength(contours[i], True),
            True
        )


def getAngle(a, b, c):
    k = (a * a + b * b - c * c) / (2 * a * b)
    k = np.clip(k, -1.0, 1.0)
    return np.degrees(np.arccos(k))


def is_square(cnt, eps=3.0):
    center = cnt.sum(axis=0) / 4.0

    dd = [
        np.linalg.norm(cnt[i] - cnt[(i + 1) % 4])
        for i in range(4)
    ]

    xa = np.linalg.norm(cnt[0] - cnt[2])
    xb = np.linalg.norm(cnt[1] - cnt[3])
    xratio = min(xa, xb) / max(xa, xb)

    angles = np.array([
        getAngle(dd[i - 1], dd[i], xb if i % 2 == 0 else xa)
        for i in range(4)
    ])

    good_angles = np.all((angles > 40) & (angles < 140))
    return good_angles and xratio > 0.5


def pruneContours(contours, hierarchy, saddle):
    new_contours = []
    new_hierarchy = []

    for cnt, h in zip(contours, hierarchy):
        if h[2] != -1:
            continue
        if len(cnt) != 4:
            continue
        if cv2.contourArea(cnt) < 64:
            continue
        if not is_square(cnt.squeeze()):
            continue

        cnt = updateCorners(cnt, saddle)
        if len(cnt) == 4:
            new_contours.append(cnt)
            new_hierarchy.append(h)

    if not new_contours:
        return np.array([]), np.array([])

    areas = np.array([cv2.contourArea(c) for c in new_contours])
    med = np.median(areas)
    mask = (areas >= med * 0.25) & (areas <= med * 2.0)

    return np.array(new_contours, dtype=object)[mask], np.array(new_hierarchy)[mask]


def getContours(img, edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    # MODERN: OpenCV 4 API
    contours, hierarchy = cv2.findContours(
        grad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = list(contours)  # MODERN: tuple → list
    simplifyContours(contours)

    return np.array(contours, dtype=object), hierarchy[0]


# =========================
# Corners
# =========================

def updateCorners(contour, saddle):
    ws = 4
    new_contour = contour.copy()

    for i in range(4):
        x, y = contour[i, 0]
        window = saddle[
            max(0, y - ws): y + ws + 1,
            max(0, x - ws): x + ws + 1
        ]

        if window.size == 0:
            return []

        dy, dx = np.unravel_index(window.argmax(), window.shape)
        if window[dy, dx] > 0:
            new_contour[i, 0] = [x + dx - ws, y + dy - ws]
        else:
            return []

    return new_contour


# =========================
# Grid / Homography
# =========================

def getIdentityGrid(N):
    a = np.arange(N)
    aa, bb = np.meshgrid(a, a)
    return np.vstack([aa.flatten(), bb.flatten()]).T

def getChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))

    quadB = getIdentityGrid(4) - 1
    quadB_pad = np.pad(quadB, ((0,0),(0,1)), constant_values=1)

    C = (M @ quadB_pad.T).T
    C[:, :2] /= C[:, 2:3]

    return C

def findGoodPoints(grid, spts, max_px_dist=5):
    new_grid = grid.copy()
    chosen_spts = set()
    N = len(new_grid)
    grid_good = np.zeros(N, dtype=bool)

    def hash_pt(pt):
        return f"{int(pt[0])}_{int(pt[1])}"

    for pt_i in range(N):
        pt2, d = getMinSaddleDist(spts, new_grid[pt_i, :2])
        if hash_pt(pt2) in chosen_spts:
            d = max_px_dist
        else:
            chosen_spts.add(hash_pt(pt2))

        if d < max_px_dist:
            new_grid[pt_i, :2] = pt2
            grid_good[pt_i] = True

    return new_grid, grid_good

def getInitChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    return makeChessGrid(M, 1)

def makeChessGrid(M, N):
    ideal_grid = getIdentityGrid(2 + 2 * N) - N
    ideal_grid_pad = np.pad(
        ideal_grid,
        ((0, 0), (0, 1)),
        constant_values=1
    )

    grid = (M @ ideal_grid_pad.T).T
    grid[:, :2] /= grid[:, 2:3]
    grid = grid[:, :2]

    return grid, ideal_grid, M

def generateNewBestFit(grid_ideal, grid, grid_good):
    a = grid_ideal[grid_good].astype(np.float32)
    b = grid[grid_good].astype(np.float32)
    M, _ = cv2.findHomography(a, b, cv2.RANSAC)
    return M

def getGrads(img):
    img = cv2.blur(img, (5,5))
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    grad_mag = gx * gx + gy * gy
    grad_phase = np.arctan2(gy, gx)
    grad_phase_masked = grad_phase.copy()

    threshold = 2 * np.mean(grad_mag)
    grad_phase_masked[grad_mag < threshold] = np.nan

    return grad_mag, grad_phase_masked, grad_phase, gx, gy

def getBestLines(img_warped):
    grad_mag, _, _, gx, gy = getGrads(img_warped)

    gx_pos = np.clip(gx, 0, None)
    gx_neg = np.clip(-gx, 0, None)
    score_x = np.sum(gx_pos, axis=0) * np.sum(gx_neg, axis=0)

    gy_pos = np.clip(gy, 0, None)
    gy_neg = np.clip(-gy, 0, None)
    score_y = np.sum(gy_pos, axis=1) * np.sum(gy_neg, axis=1)

    a = np.array([(offset + np.arange(7) + 1) * 32 for offset in np.arange(1, 9)])
    scores_x = np.array([np.sum(score_x[pts]) for pts in a])
    scores_y = np.array([np.sum(score_y[pts]) for pts in a])

    return a[scores_x.argmax()], a[scores_y.argmax()]


# =========================
# Image
# =========================

def loadImage(filepath):
    img_orig = Image.open(filepath).convert('L')
    w, h = img_orig.size
    scale = min(500 / w, 500 / h)
    img = img_orig.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.array(img)

def findChessboard(img, min_pts_needed=15, max_pts_needed=25):
    blur_img = cv2.blur(img, (3,3))
    saddle = -getSaddle(blur_img)
    saddle[saddle < 0] = 0
    pruneSaddle(saddle)

    s2 = nonmax_sup(saddle)
    s2[s2 < 100000] = 0
    spts = np.argwhere(s2)

    edges = cv2.Canny(img, 20, 250)
    contours_all, hierarchy = getContours(img, edges)
    contours, hierarchy = pruneContours(contours_all, hierarchy, saddle)

    curr_num_good = 0
    curr_grid_next = curr_grid_good = curr_M = None

    for cnt in contours:
        cnt = cnt.squeeze()
        grid_curr, ideal_grid, M = getInitChessGrid(cnt)

        for grid_i in range(7):
            grid_curr, ideal_grid, _ = makeChessGrid(M, grid_i + 1)
            grid_next, grid_good = findGoodPoints(grid_curr, spts)
            num_good = np.sum(grid_good)

            if num_good < 4:
                M = None
                break

            M = generateNewBestFit(ideal_grid, grid_next, grid_good)
            if M is None or abs(M[0,0] / M[1,1]) > 15:
                M = None
                break

        if M is not None and num_good > curr_num_good:
            curr_num_good = num_good
            curr_grid_next = grid_next
            curr_grid_good = grid_good
            curr_M = M

        if num_good > max_pts_needed:
            break

    if curr_num_good > min_pts_needed:
        final_ideal_grid = getIdentityGrid(16) - 7
        return curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts

    return None, None, None, None, None

def getUnwarpedPoints(best_lines_x, best_lines_y, M):
    x, y = np.meshgrid(best_lines_x, best_lines_y)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = xy[None, :, :]

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0]

def getBoardOutline(best_lines_x, best_lines_y, M):
    d = best_lines_x[1] - best_lines_x[0]
    ax = [best_lines_x[0] - d, best_lines_x[-1] + d]
    ay = [best_lines_y[0] - d, best_lines_y[-1] + d]

    x, y = np.meshgrid(ax, ay)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = xy[[0,1,3,2,0]]
    xy = xy[None, :, :]

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0]

def processSingle(filename='input/img_10.png'):
    img = loadImage(filename)
    M, ideal_grid, grid_next, grid_good, spts = findChessboard(img)
    print(M)

    # View
    if M is not None:
        M = generateNewBestFit((ideal_grid + 8) * 32, grid_next, grid_good)
        print(M)

        img_warp = cv2.warpPerspective(
            img, M, (17 * 32, 17 * 32),
            flags=cv2.WARP_INVERSE_MAP
        )

        best_lines_x, best_lines_y = getBestLines(img_warp)
        xy_unwarp = getUnwarpedPoints(best_lines_x, best_lines_y, M)

        plt.figure(figsize=(20, 20))

        plt.subplot(212)
        plt.imshow(img_warp, cmap='Greys_r')
        [plt.axvline(line, color='red', lw=2) for line in best_lines_x]
        [plt.axhline(line, color='green', lw=2) for line in best_lines_y]

        plt.subplot(211)
        axs = plt.axis()
        plt.imshow(img, cmap='Greys_r')
        axs = plt.axis()

        plt.plot(spts[:, 1], spts[:, 0], 'o')

        grid_np = np.asarray(grid_next)
        plt.plot(grid_np[:, 0], grid_np[:, 1], 'rs')
        plt.plot(
            grid_np[grid_good, 0],
            grid_np[grid_good, 1],
            'rs',
            markersize=12
        )

        plt.plot(xy_unwarp[:, 0], xy_unwarp[:, 1], 'go', markersize=15)
        plt.axis(axs)

        plt.savefig('result_single.png', bbox_inches='tight')
        plt.show()


def _to_int(x):
    return int(round(float(x)))

def build_9_lines(best_lines):
    """
    best_lines: 7 belső vonal (np.array)
    visszaad: 9 határvonal (külsőkkel együtt)
    """
    best_lines = np.array(best_lines, dtype=np.float32).reshape(-1)
    if len(best_lines) != 7:
        raise ValueError(f"best_lines must have length 7, got {len(best_lines)}")

    d = best_lines[1] - best_lines[0]
    lines9 = np.concatenate([[best_lines[0] - d], best_lines, [best_lines[-1] + d]])
    return lines9, float(d)

def extract_squares_from_warp(img_warp, best_x, best_y, inner_pad_ratio=0.12):
    """
    img_warp: warpolt (felülnézeti) szürke kép, pl. (17*32, 17*32)
    best_x, best_y: getBestLines(img_warp) kimenete (7 belső vonal x/y irányban)
    inner_pad_ratio: a mező széléből levágott arány (0.10-0.15 jó tipikusan)

    Returns:
      squares: 8x8 lista ROI képekkel (numpy array)
      bbox_warp: 8x8 lista (x0,y0,x1,y1) warp koordinátában (belső paddel együtt)
      centers_warp: 8x8 lista (cx,cy) warp koordinátában (a teljes mező közepére)
      lines: (x_lines9, y_lines9) a 9-9 határvonal
    """
    h, w = img_warp.shape[:2]

    x_lines, dx = build_9_lines(best_x)
    y_lines, dy = build_9_lines(best_y)

    squares = [[None for _ in range(8)] for _ in range(8)]
    bbox_warp = [[None for _ in range(8)] for _ in range(8)]
    centers_warp = [[None for _ in range(8)] for _ in range(8)]

    for r in range(8):
        for c in range(8):
            x0f, x1f = x_lines[c], x_lines[c + 1]
            y0f, y1f = y_lines[r], y_lines[r + 1]

            # biztosítsuk a sorrendet
            x0f, x1f = (x0f, x1f) if x0f <= x1f else (x1f, x0f)
            y0f, y1f = (y0f, y1f) if y0f <= y1f else (y1f, y0f)

            # belső padding, hogy a rácsvonal ne zavarjon
            padx = inner_pad_ratio * (x1f - x0f)
            pady = inner_pad_ratio * (y1f - y0f)

            x0 = _to_int(x0f + padx)
            x1 = _to_int(x1f - padx)
            y0 = _to_int(y0f + pady)
            y1 = _to_int(y1f - pady)

            # clamp
            x0 = max(0, min(w - 1, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h, y1))

            # ha túl kicsi lett, fallback: padding nélkül
            if x1 - x0 < 4 or y1 - y0 < 4:
                x0 = max(0, min(w - 1, _to_int(x0f)))
                x1 = max(0, min(w, _to_int(x1f)))
                y0 = max(0, min(h - 1, _to_int(y0f)))
                y1 = max(0, min(h, _to_int(y1f)))

            roi = img_warp[y0:y1, x0:x1].copy()

            squares[r][c] = roi
            bbox_warp[r][c] = (x0, y0, x1, y1)

            # középpont: a TELJES mező közepe (nem a paddelt ROI közepe)
            cx = 0.5 * (x0f + x1f)
            cy = 0.5 * (y0f + y1f)
            centers_warp[r][c] = (float(cx), float(cy))

    return squares, bbox_warp, centers_warp, (x_lines, y_lines)

def warp_points_to_image(pts_warp, M):
    """
    pts_warp: Nx2 pontok a warp térben
    M: az a homográfia, amit a warpPerspective-ben használsz WARP_INVERSE_MAP-pel.
       Nálad: img_warp = warpPerspective(img, M, ..., flags=WARP_INVERSE_MAP)
       Ez azt jelenti, hogy M az "warp->image" transzformáció is használható perspectiveTransformmal.

    Returns: Nx2 pontok az eredeti képen.
    """
    pts = np.asarray(pts_warp, dtype=np.float32).reshape(-1, 1, 2)
    pts_img = cv2.perspectiveTransform(pts, M)
    return pts_img.reshape(-1, 2)

def centers_to_image(centers_warp_8x8, M):
    """
    centers_warp_8x8: 8x8 lista (cx,cy) warp térben
    Returns:
      centers_img_8x8: 8x8 lista (x,y) eredeti képen
    """
    flat = []
    for r in range(8):
        for c in range(8):
            flat.append(centers_warp_8x8[r][c])
    flat = np.array(flat, dtype=np.float32)
    flat_img = warp_points_to_image(flat, M)

    centers_img = [[None for _ in range(8)] for _ in range(8)]
    k = 0
    for r in range(8):
        for c in range(8):
            centers_img[r][c] = (float(flat_img[k, 0]), float(flat_img[k, 1]))
            k += 1
    return centers_img

# =========================
# Main
# =========================

def main():
    filenames = glob.glob('../tanito_kepek/7.jpg')
    filenames = sorted(filenames)
    print("Files: %s" % filenames)

    fig = plt.figure(figsize=(20, 20))
    n = len(filenames)

    if n == 0:
        print("No files found.")
        return

    col = 4
    row = n // col
    if n % col != 0:
        row += 1

    for i in range(n):
        filename = filenames[i]
        print("Processing %d/%d : %s" % (i + 1, n, filename))

        img = loadImage(filename)
        M, ideal_grid, grid_next, grid_good, spts = findChessboard(img)

        if M is not None:
            M = generateNewBestFit((ideal_grid + 8) * 32, grid_next, grid_good)

            img_warp = cv2.warpPerspective(
                img, M, (17 * 32, 17 * 32),
                flags=cv2.WARP_INVERSE_MAP
            )

            best_lines_x, best_lines_y = getBestLines(img_warp)
            xy_unwarp = getUnwarpedPoints(best_lines_x, best_lines_y, M)
            board_outline_unwarp = getBoardOutline(best_lines_x, best_lines_y, M)

            fig.add_subplot(row, col, i + 1)

            axs = plt.axis()
            plt.imshow(img, cmap='Greys_r')
            axs = plt.axis()

            plt.plot(xy_unwarp[:, 0], xy_unwarp[:, 1], 'r.')
            plt.plot(
                board_outline_unwarp[:, 0],
                board_outline_unwarp[:, 1],
                'ro-',
                markersize=5,
                linewidth=3
            )

            plt.axis(axs)
            plt.title("%s :  N matches=%d" % (filename, np.sum(grid_good)))
            plt.axis('off')

            print("    N good pts %d" % np.sum(grid_good))

        else:
            fig.add_subplot(row, col, i + 1)
            plt.imshow(img, cmap='Greys_r')
            plt.title("%s : Fail" % filename)
            plt.axis('off')
            print("    Fail")

    plt.savefig('result.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("Start")
    main()