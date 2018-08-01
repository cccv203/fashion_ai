import  numpy as np

blouse_inx_list = [0,1,2,3,4,5,6, 9,10,11,12,13,14]  ## 13 pts
dress_inx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,17,18] ## 15 pts
outwear_inx_list = [0,1,3,4,5,6,7,8,9,10,11,12,13,14] ## 14 pts
skirt_inx_list = [15,16,17,18] ## 4 pts
trousers_inx_list = [15,16,19,20,21,22,23]  ## 7 pts




# 从所有24点中滤除某类的点
def get_self_catogory_joints(joints, catogory):
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


def parse_csv_list(filename_list):
    data_dict = {}
    for filename in filename_list:
        input_file = open(filename, 'r')
        i = 0
        for line in input_file:
            if i == 0:
                i = 1
                continue
            line = line.strip()
            line = line.split(',')
            name = line[0]
            type = line[1]
            def fn(x):
                c = x.split('_')
                return list(map(int, c[:]))
            joints = list(map(fn, line[2:]))
            joints = np.reshape(joints, (-1, 3))
            joints = get_self_catogory_joints(joints, type)
            data_dict[name] = {'joints': joints[:, 0:2], 'type': type}
        input_file.close()
    return data_dict

def to_one_submit_str(img_name, catgory, predPtss, inx_list):
    strDstList = []
    N = 24
    for i in range(N):
        strDstList.append('-1_-1_-1')
    i = 0
    for pts in predPtss:
        one = str(int(pts[0])) + '_' + str(int(pts[1])) + '_1'
        strDstList[inx_list[i]] = one
        i += 1
    ret = img_name + ',' + catgory
    for item in strDstList:
        ret += ','
        ret += item
    return ret



def save_data_to_final_csv(data_dict,filename):
    f = open(filename, 'w')
    img_list = list(data_dict.keys())
    header = 'image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out'
    f.write(header + '\n')
    for name in img_list:
        catgory = data_dict[name]['type']
        inx_list=None
        if catgory=='blouse':
            inx_list = blouse_inx_list
        if catgory=='dress':
            inx_list = dress_inx_list
        if catgory == 'outwear':
            inx_list = outwear_inx_list
        if catgory=='skirt':
            inx_list = skirt_inx_list
        if catgory=='trousers':
            inx_list= trousers_inx_list
        content = to_one_submit_str(name, catgory, data_dict[name]['joints'], inx_list)
        f.write(content + '\n')
    f.close()

pred_file_list = ['../pred_result/0331/blouse_pred.csv',
                  '../pred_result/0331/dress_pred.csv',
                  '../pred_result/0331/outwear_pred.csv',
                  '../pred_result/0331/skirt_pred.csv',
                  '../pred_result/0331/trousers_pred.csv']
if __name__ =='__main__':
    save_name='../pred_result/0331/0331_pred.csv'
    data_dict = parse_csv_list(pred_file_list)
    save_data_to_final_csv(data_dict,save_name)

