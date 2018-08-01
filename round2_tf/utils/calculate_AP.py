import numpy as np
import cv2
GT_FILE = '../csv/local_test.csv'  ### 真值文件
PRED_FILE = '../pred_result/local_pred.csv'### 预测的文件


def pts_avg_distance(pts_gt,pts_pre,catgory):
    dis=0
    n=0
    for i,p in enumerate(pts_gt):
        if p[2]==1:
            n=n+1
            d =np.sqrt((p[0]-pts_pre[i][0])*(p[0]-pts_pre[i][0]) + (p[1]-pts_pre[i][1])*(p[1]-pts_pre[i][1]) )
            dis = dis+d
    norm =0
    if catgory=='dress'  or catgory=='outwear' or catgory=='blouse':
        norm = np.sqrt(np.square(pts_gt[5][0]-pts_gt[6][0])  + np.square(pts_gt[5][1]-pts_gt[6][1])  )
        if np.isnan(norm):
            print(pts_gt[5])
            print(pts_gt[6])
    else:
        norm = np.sqrt(np.square(pts_gt[15][0] - pts_gt[16][0]) + np.square(pts_gt[15][1] - pts_gt[16][1]))
        if np.isnan(norm):
            print(pts_gt[15])
            print(pts_gt[16])
    if norm==0:
       norm = 256
    if n==0:
        ##print('n==0')
        return  0
    try:
       d = (dis / n) / norm
    except:
        print('except: '+norm)
        exit(0)
    return d

def build_data_dict(filename):
    input_file = open(filename, 'r')
    data_dict={}
    i= 0
    for line in input_file:
        if len(line)<10:
            continue
        if i==0:
            i=1
            continue

        line = line.strip()
        line = line.split(',')
        name = line[0]
        type = line[1]
        def fn(x):
            c = x.split('_')
            return list(map(int,c[:]))
        joints = list(map(fn, line[2:]))
        joints = np.reshape(joints, (-1, 3))
        data_dict[name] = {'joints': joints,  'type': type}
    input_file.close()
    return data_dict

if __name__=='__main__':
    gt_data = build_data_dict(GT_FILE)
    pred_data = build_data_dict(PRED_FILE)
    filter_cat = ['blouse','dress','outwear','skirt','trousers']
    n=0
    dis=0
    for (name,v) in gt_data.items():
        if v['type'] not in filter_cat:
            continue
        d  = pts_avg_distance(v['joints'], pred_data[name]['joints'],v['type'])
        dis=dis+d
        n=n+1
    dis = dis/n
    print('Total  Score: ', dis*100)

    for cat in filter_cat:
        n = 0
        dis = 0
        for (name, v) in gt_data.items():
            if v['type'] !=cat:
                continue
            d = pts_avg_distance(v['joints'], pred_data[name]['joints'], v['type'])
            dis = dis + d
            n = n + 1
        dis = dis / n
        print(cat+ ' Score: ', dis * 100)

    ###
    # filter_cat = ['trousers']
    # loss_dict = {}
    # n = 0
    # dis = 0
    # for (name, v) in gt_data.items():
    #     if v['type'] not in filter_cat:
    #         continue
    #     d = pts_avg_distance(v['joints'], pred_data[name]['joints'], v['type'])
    #     loss_dict[name] = d
    #     dis = dis + d
    #     n = n + 1
    # dis = dis / n
    #
    # dict = sorted(loss_dict.items(), key=lambda d: d[1], reverse=True)
    # N = 100
    # i = 0
    # img_dir = "F:\\fashionAI_key_points_train_20180227\\train\\"
    # for (name, v) in dict:
    #     i = i + 1
    #     if i > N:
    #         break
    #     src = cv2.imread(img_dir + name)
    #     pts1 = pred_data[name]['joints']
    #     for p in pts1:
    #         cv2.circle(src, (p[0], p[1]), 2, (0, 255, 0), 2, 16)
    #     pts2 = gt_data[name]['joints']
    #     for i, p in enumerate(pts2):
    #         if p[2] == 1:
    #             cv2.circle(src, (p[0], p[1]), 2, (0, 0, 255), 2, 16)
    #             cv2.line(src, (p[0], p[1]), (pts1[i][0], pts1[i][1]), (0, 0, 255), 1, 16)
    #
    #     print(name + ' loss: ', v)
    #     cv2.namedWindow('src', 0)
    #     cv2.imshow('src', src)
    #     cv2.waitKey(0)