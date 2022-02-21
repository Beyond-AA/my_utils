import os
import numpy as np
import cv2

file = r"E:\imgs"


def find_rect_contour_point(img):
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lens = []
    for contour in contours:
        lens.append(len(contour))
    try:
        min_rect = cv2.minAreaRect(contours[lens.index(max(lens))])
    except:
        pass
    rect_points = cv2.boxPoints(min_rect)
    rect_points = np.int0(rect_points)
    area = []
    for i in rect_points:
        area.append(i[0]*i[1])
    top_left = [rect_points[area.index(min(area))][0], rect_points[area.index(min(area))][1]]
    bottom_right = [rect_points[area.index(max(area))][0], rect_points[area.index(max(area))][1]]
    rect_points_ = []
    for i in rect_points:
        c = list(i)
        rect_points_.append(c)
    flask_point = [top_left, bottom_right]
    a = []
    for i in rect_points_:
        if i not in flask_point:
            a.append(i)
    y_sorted = sorted(a, key=lambda tup: tup[1])
    new_point = [top_left, y_sorted[0], y_sorted[1], bottom_right]
    return new_point


def affine(point_lis, img):
    pts1 = np.float32(point_lis)
    pts2 = np.float32([[0, 0], [1024, 0], [0, 1024], [1024, 1024]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (1024, 1024))
    return dst

"""
for i in os.listdir(file):
    file1 = os.path.join(file, i)
    for j in os.listdir(file1):
        file2 = os.path.join(file1, j)
        for k in os.listdir(file2):
            img_path = os.path.join(file2, k)
            img = cv2.imread(img_path, 0)
            # pts1 = np.float32(find_rect_contour_point(img))
            # pts2 = np.float32([[0, 0], [1024, 0], [0, 1024], [1024, 1024]])
            # M = cv2.getPerspectiveTransform(pts1, pts2)
            # dst = cv2.warpPerspective(img, M, (1024, 1024))
            dst = affine(find_rect_contour_point(img), img)
            cv2.imwrite("E:/all_imgs/{}".format(k), dst)
            # cv2.imshow("img", dst)
            # cv2.waitKey(1)
# cv2.destroyAllWindows()
"""

for i in os.listdir(file):
    file1 = os.path.join(file, i)
    for j in os.listdir(file1):
        file2 = os.path.join(file1, j)
        for k in os.listdir(file2):
            img_path = os.path.join(file2, k)
            img = cv2.imread(img_path, 0)
            # pts1 = np.float32(find_rect_contour_point(img))
            # pts2 = np.float32([[0, 0], [1024, 0], [0, 1024], [1024, 1024]])
            # M = cv2.getPerspectiveTransform(pts1, pts2)
            # dst = cv2.warpPerspective(img, M, (1024, 1024))
            dst = affine(find_rect_contour_point(img), img)
            print(j, k)
            cv2.imwrite("E:/hours/{0}/{1}".format(i, k), dst)

