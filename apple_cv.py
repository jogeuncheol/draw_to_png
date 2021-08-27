import sys
import potrace
import cv2 as cv
import numpy as np
# from PIL import ImageGrab


def find_min_max_coordinate(find_coord_list, axis):
    res_min, res_max, coord = 0, 0, 0
    if axis == 'y':
        coord = 1
    res_min = find_coord_list[0][coord]
    res_max = find_coord_list[0][coord] + find_coord_list[0][coord + 2]
    for i in range(len(find_coord_list)):
        f_min = find_coord_list[i][coord]
        f_max = find_coord_list[i][coord] + find_coord_list[i][coord + 2]
        if res_min > f_min:
            res_min = f_min
        if res_max < f_max:
            res_max = f_max
    print(axis, "min, max : ", res_min, res_max)
    return res_min, res_max

# cv.namedWindow("Trackbar Windows", None)
# cv.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
# cv.setTrackbarPos("threshold", "Trackbar Windows", 180)


image_origin = cv.imread("test_img_c.jpg", cv.IMREAD_UNCHANGED)
if image_origin is None:
    print("image read fail")
    sys.exit(1)
print(image_origin.shape)

# image_origin = cv.cvtColor(np.array(ImageGrab.grab()), cv.COLOR_BGR2RGB)
# np.zeros((height, width), np.uint8)
new_image1 = np.zeros((image_origin.shape[0], image_origin.shape[1], 4), dtype=np.uint8)
image_resize = cv.resize(image_origin, dsize=(0, 0), fx=0.3, fy=0.3)
img_gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)
#img_gray = cv.add(img_gray, 50)
img_blur = cv.GaussianBlur(img_gray, (0, 0), 1)
cv.imshow("GaussianBlur image", img_blur)
# img_blur = cv.bilateralFilter(img_gray, 10, 50, 50)
# thresh = cv.getTrackbarPos("threshold", "Trackbar Windows")
_, binary = cv.threshold(img_blur, 100, 255, cv.THRESH_BINARY_INV)
cv.imshow('binary', binary)
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(new_image1, contours, -1, (255, 255, 255), 3)
cv.imshow("drawContours image", new_image1)

img_gray = cv.cvtColor(new_image1, cv.COLOR_BGR2GRAY)
_, binary2 = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY)
cv.imshow("binary2", binary2)
contours2, hierarchy2 = cv.findContours(binary2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

find_coord = []
for cnt in contours2:
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(image_resize, (x, y), (x + w, y + h), (0, 255, 0), 1) # <-- draw rectangle
    find_coord.append([x, y, w, h])
cv.imshow('new_image : rectangle', new_image1)
print(find_coord)
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
cv.imshow("capture_window", image_resize)
cv.imwrite("test_cut_image.bmp", cp_img)
print(cp_img)

# Make a numpy array with a rectangle in the middle
data = np.zeros((32, 32), np.uint8)
data[1:31, 1:31] = 1
print(data)
data2 = cp_img
print(data2)

# Create a bitmap from the array
bmp = potrace.Bitmap(data)

# Trace the bitmap to a path
path = bmp.trace()

# Iterate over path curves
for curve in path:
    print("start_point =", curve.start_point)
    for segment in curve:
        print(segment)
        end_point_x, end_point_y = segment.end_point
        if segment.is_corner:
            c_x, c_y = segment.c
            print('c_x, c_y', c_x, c_y)
        else:
            c1_x, c1_y = segment.c1
            c2_x, c2_y = segment.c2
            print('c1_x c1_y', c1_x, c1_y)

key = cv.waitKey()
cv.destroyAllWindows()
