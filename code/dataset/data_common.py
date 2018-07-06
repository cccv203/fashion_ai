import  numpy as np
import cv2


joints_list=[
'neckline_left', ## 脖圈左 0
'neckline_right',## 脖圈右 1
'center_front', ## 脖圈中间 2
'shoulder_left',## 肩膀左 3
'shoulder_right',##肩膀右 4
'armpit_left',##腋窝左 5
'armpit_right',##腋窝右 6
'waistline_left',## 腰身左 7
'waistline_right',## 腰身右 8

 ####  衣袖  ####
'cuff_left_in',## 9
'cuff_left_out',## 10
'cuff_right_in',## 11
'cuff_right_out',## 12
############################

'top_hem_left', # 13 上身褶边左
'top_hem_right',# 14 上身褶边右

'waistband_left',# 15 裤子或者裙子的裤腰那
'waistband_right',# 16

'hemline_left', # 17 dress和skirt 下部褶边
'hemline_right',#18

'crotch',  ##  19 裆部
### 裤腿
'bottom_left_in', #20
'bottom_left_out',#21
'bottom_right_in',#22
'bottom_right_out'#23
]

data_map={

### 标准数据
'standard':{
    'blouse':{'catgory_list':['blouse'],
              'inx_list':[0,1,2,3,4,5,6, 9,10,11,12,13,14]}, ## 13
    'dress':{'catgory_list':['dress'],
             'inx_list': [0,1,2,3,4,5,6,7,8,9,10,11,12,17,18]}, ## 15 pts
    'outwear':{'catgory_list':['outwear'],
               'inx_list':[0,1,3,4,5,6,7,8,9,10,11,12,13,14]},## 14
    'skirt':{'catgory_list':['skirt'],
             'inx_list':[15,16,17,18]}, ## 4 pts
    'trousers': {'catgory_list':['trousers'],
                 'inx_list':[15,16,19,20,21,22,23]} ## 7 pts
},
### 组合数据
'mix':{
    ###  将blouse  dress outwear  进行组合,但是不考虑袖子
    'mix1':{'catgory_list':['blouse','dress','outwear'],
            'inx_list':[0,1,2,3,4,5,6,7,8, 13,14,17,18]} ,
    ###  将trousers(裤头)和dress （下褶边） 的数据并入skirt中   0417 : 去掉 dress
    'mix2':{'catgory_list':['skirt','trousers'],'inx_list':[15,16,17,18,19,20,21,22,23] },
     ###  将blouse  dress outwear  所有点进行组合
    'mix3':{'catgory_list':['blouse','dress','outwear'],
            'inx_list':[0,1,2,3,4,5,6,7,8,9,10,11,12, 13,14,17,18],
            } ,
    'mix_b_o':{'catgory_list':['blouse','outwear'],
            'inx_list':[0,1,2,3,4,5,6,7,8,9,10,11,12, 13,14]} ,
    'mix5':{'catgory_list':['blouse','dress','outwear','skirt','trousers'],
            'inx_list':[0,1,2,3,4,5,6,7,8,9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23],
            'symmetry':[(0, 1), (3, 4), (5, 6), (7, 8), (9, 11), (10, 12), (13, 14), (15, 16), (17,18),(20,22),(21,23)]} ,

}
}


def get_train_data_dict_from_csv(csv_file,except_list,catgory_list, inx_list,symmetry=None):
    print(catgory_list)
    input_file = open(csv_file, 'r')
    data_dict = {}
    i = 0
    pts_num=np.zeros(len(inx_list))
    for line in input_file:
        if i == 0: ### remove  header
            i += 1
            continue
        # if i>10:
        #     break
        line = line.strip()
        line = line.split(',')
        name = line[0]
        type = line[1]
        if type not in catgory_list:
            continue
        if name in except_list:
            continue
        def fn(x):
            c = x.split('_')
            return list(map(int, c[:]))
        joints = list(map(fn, line[2:]))
        joints = np.reshape(joints, (-1, 3))
        joints = joints[inx_list]
        valid_joints = joints[np.where(joints[:, 2] != -1)][:, 0:2]
        b = cv2.boundingRect(np.array(valid_joints))
        strict_bb = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        ### 考虑到混合模型  不存在的点设置为遮挡较为合适
        for (i, p) in enumerate(joints):
            if p[2]!=-1:
                pts_num[i]+=1
            if p[0] == -1:
                joints[i] = [2e3,2e3, -1]
        data_dict[name] = {'joints': joints[:, 0:2], 'visible': joints[:, 2],'sbb':strict_bb, 'type': type}

        if symmetry is not None:
            joints_flip = joints.copy()
            #print(joints_flip[7])
            for (q, w) in symmetry:
                joints_flip[q], joints_flip[w] = joints[w], joints[q]
            #print(joints_flip[7])
            data_dict[name + '_flip'] = {'joints': joints_flip[:, 0:2], 'visible': joints_flip[:, 2], 'type': type}


    #print(pts_num)
    input_file.close()
    return data_dict
'''
type   'standard'  'mix'  'critical'
which  具体type里的哪个
'''
def get_train_data_dict(csv_file, type, which,except_list, flip=True):
    catgory_list = data_map[type][which]['catgory_list']
    inx_list = data_map[type][which]['inx_list']
    joints_num = len(inx_list)
    symmetry=None
    if flip:
        symmetry=data_map[type][which]['symmetry']
    return  get_train_data_dict_from_csv(csv_file,except_list,catgory_list,inx_list,symmetry), joints_num

def build_bbox_dict(bboxAnnoFile):
    input_file = open(bboxAnnoFile, 'r')
    data_dict = {}
    for line in input_file:
        line = line.strip()
        line = line.split(',')
        name = line[0]
        b0 = list(map(float, line[1:5]))
        b0 = [min(int(b0[0]), int(b0[2])),
                  min(int(b0[1]), int(b0[3])),
                  max(int(b0[0]), int(b0[2])),
                  max(int(b0[1]), int(b0[3]))]
        data_dict[name] = b0
    input_file.close()
    return data_dict

def get_err_list(file):
    f=open(file)
    names=[]
    for line in f:
        line = line.split(',')
        name = line[0].strip()
        names.append(name)
    return names
