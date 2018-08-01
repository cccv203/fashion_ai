import  cv2
import  numpy as np
# joints:  关键点坐标
# visible:  -1 不存在  0 遮挡  1 可见
# bbox  [xmin,ymin, xmax,ymax]

def draw_joints(src,joints=None, visible=None,withText=True, bbox=None,wait=0):
    img = src.copy()
    i = 0
    if  joints is not None:
      for coord in joints:
        if visible[i] !=-1:
            keypt = (int(coord[0]), int(coord[1]))
            text_loc = (keypt[0]+5, keypt[1]+7)
            if visible[i]==0:
                cv2.circle(img, keypt, 3, (0,0,255), -1)
                if withText:
                    cv2.putText(img, str(i), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            if visible[i]==1:
                cv2.circle(img, keypt, 3, (0,255,0), -1)
                if withText:
                    cv2.putText(img, str(i), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        i += 1
    if bbox is not None:
        cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,255),3,16 )
    cv2.namedWindow('img',0)
    cv2.imshow('img', img)
    cv2.waitKey(wait)

def draw_hm(src,hm):
    img=src.copy()
    hm_gui = hm.copy()
    mask = cv2.resize(hm_gui,(img.shape[1],img.shape[0]))
    img[:, :, 2] = np.where(mask > 0.1, 255, img[:, :, 2])
    img[:, :, 1] = np.where(mask > 0.1, 0, img[:, :, 1])
    img[:, :, 0] = np.where(mask > 0.1, 0, img[:, :, 0])
    cv2.imshow('img', img)
    cv2.namedWindow('hm', 0)
    cv2.imshow('hm', hm_gui)
    cv2.waitKey(0)
