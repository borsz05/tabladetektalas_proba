import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
#Enter the number of inside corners in x
nx = 9
#Enter the number of inside corners in y
ny = 6
chess_board_image = mpimg.imread("../sakktabla.jpg")
# Convert to grayscale
gray = cv2.cvtColor(chess_board_image, cv2.COLOR_RGB2GRAY)
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(chess_board_image, (nx, ny), None)
# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(chess_board_image, (nx, ny), corners, ret)
    cv2.imwrite("result.jpg", chess_board_image)
    cv2.imshow('img', chess_board_image)