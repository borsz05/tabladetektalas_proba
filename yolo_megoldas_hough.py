import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. MODELL ---
model = YOLO("best.pt")

def get_ordered_corners(points):
    pts = np.array(points, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def line_intersection(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    A = np.array([
        [x2-x1, x3-x4],
        [y2-y1, y3-y4]
    ])
    B = np.array([
        x3-x1,
        y3-y1
    ])

    if abs(np.linalg.det(A)) < 1e-6:
        return None

    t, _ = np.linalg.solve(A, B)
    return np.array([x1 + t*(x2-x1), y1 + t*(y2-y1)])

# --- 2. VIDEÓ ---
cap = cv2.VideoCapture("test_video_1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_size = 600
alpha = 0.2
prev_corners = None

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_original = cv2.VideoWriter("output_original2.mp4", fourcc, fps, (w, h))
out_warped = cv2.VideoWriter("output_warped2.mp4", fourcc, fps, (output_size, output_size))

# --- 3. LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, verbose=False)[0]

    if results.masks is not None:
        mask = results.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=150,
            maxLineGap=20
        )

        if lines is not None:
            horiz = []
            vert = []

            for l in lines[:,0]:
                x1,y1,x2,y2 = l
                angle = abs(np.arctan2(y2-y1, x2-x1))
                if angle < np.pi/6:
                    horiz.append(l)
                elif angle > np.pi/3:
                    vert.append(l)

            if len(horiz) >= 2 and len(vert) >= 2:
                top = min(horiz, key=lambda l: min(l[1], l[3]))
                bottom = max(horiz, key=lambda l: max(l[1], l[3]))
                left = min(vert, key=lambda l: min(l[0], l[2]))
                right = max(vert, key=lambda l: max(l[0], l[2]))

                corners = [
                    line_intersection(top, left),
                    line_intersection(top, right),
                    line_intersection(bottom, right),
                    line_intersection(bottom, left)
                ]

                if all(c is not None for c in corners):
                    ordered = get_ordered_corners(np.array(corners))

                    # --- SMOOTHING ---
                    if prev_corners is None:
                        smooth = ordered
                    else:
                        smooth = alpha * ordered + (1 - alpha) * prev_corners
                    prev_corners = smooth

                    dst = np.array([
                        [0, 0],
                        [output_size-1, 0],
                        [output_size-1, output_size-1],
                        [0, output_size-1]
                    ], dtype=np.float32)

                    M = cv2.getPerspectiveTransform(smooth.astype(np.float32), dst)
                    warped = cv2.warpPerspective(frame, M, (output_size, output_size))
                    out_warped.write(warped)

                # debug
                for l in [top,bottom,left,right]:
                    x1,y1,x2,y2 = l
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    out_original.write(frame)

cap.release()
out_original.release()
out_warped.release()
