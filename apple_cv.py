from PIL import ImageGrab
import cv2 as cv
import numpy as np

while True:
    image_origin = cv.cvtColor(np.array(ImageGrab.grab()), cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(image_origin, cv.COLOR_BGR2GRAY)
    image_resize = cv.resize(img_gray, dsize=(0, 0), fx=0.8, fy=0.8)
    cv.imshow("capture_window", image_resize)
    key = cv.waitKey(5)
    if key == 27:
        break

cv.destroyWindow("capture_window")