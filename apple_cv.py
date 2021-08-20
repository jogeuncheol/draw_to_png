from PIL import ImageGrab
import cv2 as cv
import numpy as np


def onChange(pos):
    pass

def find_min(coord):
    return (min)

def find_max(coord):
    return (max)

def find_min_max_coordinate(find_coord, axis):
    min, max, coord = 0, 0, 0
    if axis == 'y':
        coord = 1
    min = find_coord[0][coord]
    max = find_coord[0][coord] + find_coord[0][coord + 2]
    for i in range(len(find_coord)):
        f_min = find_coord[i][coord]
        f_max = find_coord[i][coord] + find_coord[i][coord + 2]
        if (min > f_min):
            min = f_min
        if (max < f_max):
            max = f_max
    print(axis, "min, max : ", min, max)
    return (min, max)

# cv.namedWindow("Trackbar Windows", None)
# cv.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
# cv.setTrackbarPos("threshold", "Trackbar Windows", 180)

image_origin = cv.imread("E:/workspace/tes_img/draw_to_image/test_img2.jpg", cv.IMREAD_COLOR)

# while True:
    # image_origin = cv.cvtColor(np.array(ImageGrab.grab()), cv.COLOR_BGR2RGB)

image_resize = cv.resize(image_origin, dsize=(0, 0), fx=1, fy=1)
img_gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)
img_blur = cv.bilateralFilter(img_gray, 9, 75, 75)
# thresh = cv.getTrackbarPos("threshold", "Trackbar Windows")
_, binary = cv.threshold(img_gray, 125, 255, cv.THRESH_BINARY_INV)
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

find_coord = []
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(image_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
    find_coord.append([x, y, w, h])

resolution = []
resolution.append(find_min_max_coordinate(find_coord, 'x'))
resolution.append(find_min_max_coordinate(find_coord, 'y'))
print(resolution)
new_image = np.zeros((resolution[1][1] - [1][0], resolution[0][1] - resolution[0][0]), dtype=np.uint8) # np.zeros((height, width), np.uint8)
cv.imshow("new_image", new_image)

while True:
    coordinate = 4
    if (coordinate == 4):
        break

# cv.drawContours(image_resize, contours, -1, (0, 0, 255), 5)

cv.imshow("Trackbar Windows", binary)
cv.imshow("capture_window", image_resize)

key = cv.waitKey()
cv.destroyWindow()