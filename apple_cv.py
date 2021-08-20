from PIL import ImageGrab
import cv2 as cv
import numpy as np

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

image_origin = cv.imread("E:/workspace/tes_img/draw_to_image/test_img6.jpg", cv.IMREAD_COLOR)
print(image_origin.shape)

# while True:
    # image_origin = cv.cvtColor(np.array(ImageGrab.grab()), cv.COLOR_BGR2RGB)
new_image1 = np.zeros((image_origin.shape[0], image_origin.shape[1], 4), dtype=np.uint8) # np.zeros((height, width), np.uint8)
image_resize = cv.resize(image_origin, dsize=(0, 0), fx=1, fy=1)
img_gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)
img_blur = cv.bilateralFilter(img_gray, 9, 75, 75)
# thresh = cv.getTrackbarPos("threshold", "Trackbar Windows")
_, binary = cv.threshold(img_gray, 180, 255, cv.THRESH_BINARY_INV)
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(new_image1, contours, -1, (255, 255, 255), 3)
cv.imshow('draw_contours_new_img', new_image1)

img_gray = cv.cvtColor(new_image1, cv.COLOR_BGR2GRAY)
cv.imshow('gray', img_gray)
_, binary2 = cv.threshold(img_gray, 180, 255, cv.THRESH_BINARY)
cv.imshow('binary2', binary2)
contours2, hierarchy = cv.findContours(binary2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

find_coord = []
for cnt in contours2:
    x, y, w, h = cv.boundingRect(cnt)
    # cv.rectangle(new_image1, (x, y), (x + w, y + h), (0, 255, 0), 1) # <-- draw rectangle
    find_coord.append([x, y, w, h])

# cv.imshow('draw_contours_new_img2', new_image2)
# 잘라낼 이미지의 해상도를 구하는 부분.
# find_min_max_coordinate(find_coord, 'x or y')
# 1. find_coord : 찾은 직사각형의 x, y, w, h 값의 배열들
# 2. x or y : 최소값과 최대값을 구할 축 x-min, x-max, y-min, y-max
resolution = []
resolution.append(find_min_max_coordinate(find_coord, 'x'))
resolution.append(find_min_max_coordinate(find_coord, 'y'))
print(resolution)
width = resolution[0][1] - resolution[0][0]
height = resolution[1][1] - resolution[1][0]

cp_img = new_image1[resolution[1][0]:resolution[1][1], resolution[0][0]:resolution[0][1]].copy()
cp_img = 255 - cp_img
cv.imshow("cp_img", cp_img)
cv.imshow("Trackbar Windows", binary)
cv.imshow("capture_window", image_resize)
cv.imwrite("E:/workspace/tes_img/draw_to_image/test_cut_image.png", cp_img)

key = cv.waitKey()
cv.destroyWindow()