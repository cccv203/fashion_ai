import cv2
import os

import numpy as np
def on_mouse(event, x, y, flags, params):
    img, points = params['img'], params['points']
    if event == cv2.EVENT_FLAG_LBUTTON:
        points.append((x, y))
    if event == cv2.EVENT_FLAG_RBUTTON:
        points.pop()
    temp = img.copy()
    if len(points) > 2:
        cv2.fillPoly(temp, [np.array(points)], (0, 0, 255))
    for i in range(len(points)):
        cv2.circle(temp, points[i], 1, (0, 0, 255))
    cv2.circle(temp, (x, y), 1, (0, 255, 0))
    cv2.imshow('img', temp)

def label_img(img, label_name):
    c = 'x'
    tiny = np.zeros(img.shape)
    while c != 'n':
        cv2.namedWindow('img', 0)
        temp = img.copy()
        points = []
        cv2.setMouseCallback('img', on_mouse, {'img': temp, 'points': points})
        cv2.imshow('img', img)
        c = chr(cv2.waitKey(0))
        if c == 's':
            if len(points) > 0:
                print('save')
                cv2.fillPoly(img, [np.array(points)], (0, 0, 255))
                cv2.fillPoly(tiny, [np.array(points)], (255, 255, 255))
    cv2.imwrite(label_name, tiny)
    return


if __name__ == '__main__':
    img_dir = 'F:/fashionAI_key_points_test_a_20180227/test/Images/skirt/'
    save_dir = 'F:/fashionAI_key_points_test_a_20180227/test/seg/skirt/'

    img_list = os.listdir(img_dir)
    for name in img_list:
        src = cv2.imread(img_dir+name, 1)
        label_img(src, save_dir+name)
