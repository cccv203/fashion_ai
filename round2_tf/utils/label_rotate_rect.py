### 标注旋转矩形
import  cv2
import  numpy as np
blouse_inx_list = [0,1,2,3,4,5,6, 9,10,11,12,13,14]  ## 13 pts
outwear_inx_list = [0,1,3,4,5,6,7,8,9,10,11,12,13,14] ## 14 pts
### Together with blouse and outwear
blouse_outwear_inx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  ##15pts
skirt_inx_list = [15,16,17,18] ## 4 pts
dress_inx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,17,18] ## 15 pts
trousers_inx_list = [15,16,19,20,21,22,23]  ## 7 pts



def on_mouse(event, x, y, flags, param):
    cv2.circle(param['tmp'], (x, y), 15,(0, 255, 0), 5, 16)
    pt1  = (param['pt1'][0],param['pt1'][1])
    pt2 = ( param['pt2'][0],param['pt2'][1])
    cv2.rectangle(param['tmp'],tuple(param['pt1']),tuple(param['pt2']), (0, 255, 0), 3, 16)
    if not param['mouseDown']:
        dis1 = (x-pt1[0])*(x-pt1[0]) + (y-pt1[1])*(y-pt1[1])
        dis2 = (x-pt2[0])*(x-pt2[0]) + (y-pt2[1])*(y-pt2[1])
        if dis1<dis2:
            param['spotInx'] =0
            cv2.circle(param['tmp'], pt1, 5, (0, 0, 255), 3, 16)
        else:
            param['spotInx'] = 1
            cv2.circle(param['tmp'], pt2, 5, (0, 0, 255), 3, 16)
    if param['mouseDown'] and event==cv2.EVENT_MOUSEMOVE:
        if param['spotInx']==0:
            param['pt1'][0] = param['spotInitP'][0] + (x- param['baseP'][0])
            param['pt1'][1] = param['spotInitP'][1] + (y - param['baseP'][1])
            cv2.circle(param['tmp'], tuple(param['pt1']), 5, (0, 0, 255), 3, 16)
        else:
            param['pt2'][0] = param['spotInitP'][0] + (x - param['baseP'][0])
            param['pt2'][1] = param['spotInitP'][1] + (y - param['baseP'][1])
            cv2.circle(param['tmp'], tuple(param['pt2']), 5, (0, 0, 255), 3, 16)
    if event==cv2.EVENT_LBUTTONDOWN:
        if not param['mouseDown']:
               param['baseP'] = [x,y]
        param['mouseDown'] = True
        param['spotInitP'] = param['pt1'].copy() if param['spotInx'] == 0 else  param['pt2'].copy()
    if event==cv2.EVENT_LBUTTONUP:
        param['mouseDown']=False
    cv2.imshow('img', param['tmp'])
    param['tmp'] = param['src'].copy()


RED = (0, 0, 255)
def show_prections(img, ptss):
    i = 0
    for coord in ptss:
        i += 1
        keypt = (int(coord[0]), int(coord[1]))
        text_loc = (keypt[0]+5, keypt[1]+7)
        cv2.circle(img, keypt, 3, RED, -1)
        cv2.putText(img, str(i), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, RED, 1, cv2.LINE_AA)
    cv2.namedWindow('img',0)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def label_one_img(img, ptss):
    # for pts in ptss:
    #    cv2.circle(img, (pts[0],pts[1]), 3, (0, 255, 0), 5, 16)
    mouse_params={
       'src':img,
        'tmp':img.copy(),
        'ptss':ptss,
        'spotInx':0,
        'mouseDown':False,
        'baseP':None,
        'spotInitP':None
    }

    cv2.namedWindow("img", 0)
    ##cv2.setMouseCallback("img", on_mouse,mouse_params)
    cv2.imshow("img", mouse_params['tmp'])
    cv2.waitKey(0)



def get_self_catogory_joints(joints,catogory_list):
    catogory = catogory_list[0]
    for i in range(1, len(catogory_list)):
        catogory += '_' + catogory_list[i]

    if catogory == 'blouse':
        return joints[blouse_inx_list]
    if catogory == 'skirt':
        return joints[skirt_inx_list]
    if catogory == 'outwear':
        return joints[outwear_inx_list]
    if catogory == 'dress':
        return joints[dress_inx_list]
    if catogory == 'trousers':
        return joints[trousers_inx_list]
    if catogory == 'blouse_outwear' or catogory == 'outwear_blouse':
        return joints[blouse_outwear_inx_list]


def build_data_dict(filename,catgory_list):
    input_file = open(filename, 'r')
    data_dict = {}
    i = 0
    for line in input_file:
        if len(line) < 10:
            continue
        if i == 0:
            i = 1
            continue
        line = line.strip()
        line = line.split(',')
        name = line[0]
        type = line[1]
        if type not in catgory_list:
            continue
        def fn(x):
            c = x.split('_')
            return list(map(int, c[:]))
        joints = list(map(fn, line[2:]))
        joints = np.reshape(joints, (-1, 3))
        joints = get_self_catogory_joints(joints, catgory_list)
        valid_joints = joints[np.where(joints[:, 2] != -1)][:, 0:2]
        b = cv2.boundingRect(np.array(valid_joints))
        bb = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        data_dict[name] = {'joints': joints[:, 0:2], 'visible': joints[:, 2], 'bb': bb, 'type': type}
    input_file.close()
    return data_dict

if __name__=='__main__':
    img_dir= "F:\\fashionAI_key_points_train_20180227\\train\\"
    catgory_list =['skirt']
    gt_file = '../csv/part_train.csv'

    gt_data = build_data_dict(gt_file,catgory_list=catgory_list)

    for name in gt_data.keys():
        img = cv2.imread(img_dir+name)
        ptss = gt_data[name]['joints']

        show_prections(img, ptss)
