import cv2 as cv
import numpy as np


# Segédfüggvény a pontok sorbarendezéséhez (bal-fent, jobb-fent, jobb-lent, bal-lent)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Bal-fent: legkisebb összeg
    rect[2] = pts[np.argmax(s)] # Jobb-lent: legnagyobb összeg
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Jobb-fent: legkisebb különbség
    rect[3] = pts[np.argmax(diff)] # Bal-lent: legnagyobb különbség
    return rect

img = cv.imread('../sakktabla2.jpg')



gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gaussblur = cv.GaussianBlur(gray, (5, 5), 0)
# A blokkméret (11) és a konstans (2) legyen kísérletezve, néha a THRESH_BINARY_INV jobb
adaptivethreshold = cv.adaptiveThreshold(gaussblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

contours, _ = cv.findContours(adaptivethreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

ideally_square = None
for cnt in contours:
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
    
    if len(approx) == 4: # Megvan a négyszög
        ideally_square = approx
        break

if ideally_square is not None:
    # Pontok kinyerése és sorbarendezése
    pts_src = ideally_square.reshape(4, 2).astype(np.float32)
    pts_src = order_points(pts_src)
    
    # Cél koordináták (fontos a sorrend!)
    pts_dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
    
    M = cv.getPerspectiveTransform(pts_src, pts_dst)
    warped_image = cv.warpPerspective(img, M, (800, 800),flags=cv.INTER_LANCZOS4)
    
    cv.drawContours(img, [ideally_square], -1, (0, 255, 0), 3)
    cv.imshow('Warped Chessboard', warped_image)
else:
    print("Nem találtam megfelelő négyszöget.")

cv.imshow('Detected Chessboard', img)
cv.waitKey(0)
cv.destroyAllWindows()