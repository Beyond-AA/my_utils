import cv2
import os
import numpy as np

img_file = r"G:\hours\3"
# img1 = cv2.imread(r"E:\all_imgs\4R1-C1_2286948888_132_0T0_A.png", 0)
# img2 = cv2.imread(r"E:\all_imgs\4R1-C1_2286949112_144_0T0_A.png", 0)

def add_img(img1, img2):
    imgadd = cv2.addWeighted(img1, 1, img2, 1, 0)
    return imgadd


#dst = cv2.addWeighted(img1, 1, img2, 1, 0)
# ret, mask_dst = cv2.threshold(dst, 60, 255, cv2.THRESH_BINARY)
# dst_bg = cv2.bitwise_and(dst, mask_dst, mask=mask_dst)

def grey_scale(image):
    #img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = image
    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    #print('A = %d,B = %d' % (A, B))
    # output = np.uint8(255 / (B - A) * (img_gray - A) + 0.5)
    output = 255 / (B - A) * (img_gray - A)
    return output

dst = np.zeros((1024, 1024)).tolist()

def s():
    for i in os.listdir(img_file):
        try:
            global dst
            img_path = os.path.join(img_file, i)
            img = cv2.imread(img_path, 0)
            img = grey_scale(img)
            #dst = grey_scale(dst)
            dst = np.add(dst, img).tolist()
            print(dst)
            # cv2.imshow("dst", dst)
            # cv2.waitKey(1000)
        except:
            pass
s()
dst = np.array(dst)
rows, cols = dst.shape
dst_gray = dst.reshape((cols * rows,)).tolist()
max_gray = max(dst_gray)
min_gray = min(dst_gray)
print(max_gray, min_gray)
dst_img = np.uint8(255 / (max_gray - min_gray) * (dst - min_gray))
cv2.imwrite(r"G:\3.png", dst_img)
#cv2.destroyAllWindows()
# for i in range(len(os.listdir(img_file))):
