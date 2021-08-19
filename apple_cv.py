from PIL import ImageGrab
import cv2 as cv
import numpy as np

def onChange(pos):
    pass

cv.namedWindow("Trackbar Windows", None)
cv.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
cv.setTrackbarPos("threshold", "Trackbar Windows", 180)

while True:
    image_origin = cv.cvtColor(np.array(ImageGrab.grab()), cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(image_origin, cv.COLOR_BGR2GRAY)
    image_resize = cv.resize(img_gray, dsize=(0, 0), fx=0.8, fy=0.8)

    img_blur = cv.bilateralFilter(image_resize, 9, 75, 75)
    thresh = cv.getTrackbarPos("threshold", "Trackbar Windows")
    _, binary = cv.threshold(image_resize, thresh, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.imshow("Trackbar Windows", binary)
    cv.imshow("capture_window", image_resize)
    key = cv.waitKey(5)
    if key == 27:
        break

cv.destroyWindow("capture_window")