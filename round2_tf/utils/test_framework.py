import  numpy as np
import  cv2
import imgaug as ia
from imgaug import augmenters as iaa
blouse_inx_list = [0,1,2,3,4,5,6, 9,10,11,12,13,14]  ## 13 pts
skirt_inx_list = [15,16,17,18] ## 4 pts
outwear_inx_list = [0,1,3,4,5,6,7,8,9,10,11,12,13,14] ## 14 pts
dress_inx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,17,18] ## 15 pts
trousers_inx_list = [15,16,19,20,21,22,23]  ## 7 pts

### Together with blouse and outwear
blouse_outwear_inx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  ##15pts


class TestData():
    '''
    img_dir: 图片所在根目录
    csv_file: 阿里提供的test.csv (只含有类别信息)  或者是我们自己的test.csv  包含关键点真值
    bbox_file: 检测器产生的bbox 文件
    '''
    def __init__(self,img_dir,csv_file,catgory,img_size=(384,256),rr=0.05, bbox_file=None, has_gt_joints=True):
        self.rr=rr
        self.catgory = catgory
        self.img_size = img_size
        self.img_dir = img_dir
        self.bbox_file = bbox_file
        self.has_gt_joints  = has_gt_joints
        self.csv_file = csv_file
        if self.bbox_file is not  None:  ## 如果提供了bbox标注则使用标注的
            print('Is online test.csv')
            self.bbox_data_dict = self.build_bbox_dict(bbox_file)
            self.img_list= self.get_online_test_imglist(csv_file,catgory)
            print('images: ', len(self.img_list))
            self.is_local_test = False
        elif has_gt_joints:
            print('Is local test.csv')
            self.is_local_test = True
            self.data_dict,self.bbox_data_dict = self._build_data_dict(csv_file,catogory=catgory)
            self.img_list   = list(self.data_dict.keys())
            print('images: ', len(self.img_list))

        self.crop_new_x_y={}
        self.pred_dict={}
        self.final_pred_dict={}
        self.in_box_image_size_dict={}
    def get_online_test_imglist(self,cvs_file,catgory):
        file_list=[]
        input_file = open(cvs_file, 'r')
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
            if type != catgory:
                continue
            file_list.append(name)
        input_file.close()
        return file_list

    def to_one_submit_str(self,img_name, catgory, predPtss, inx_list):
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

    def pts_avg_distance(self,pts_gt, pts_pre, catgory):
        dis = 0
        n = 0
        for i, p in enumerate(pts_gt):
            if p[2] == 1:
                n = n + 1
                d = np.sqrt(
                    (p[0] - pts_pre[i][0]) * (p[0] - pts_pre[i][0]) + (p[1] - pts_pre[i][1]) * (p[1] - pts_pre[i][1]))
                dis = dis + d
        norm = 0
        if catgory == 'dress' or catgory == 'outwear' or catgory == 'blouse':
            norm = np.sqrt(np.square(pts_gt[5][0] - pts_gt[6][0]) + np.square(pts_gt[5][1] - pts_gt[6][1]))
            if np.isnan(norm):
                print(pts_gt[5])
                print(pts_gt[6])
        else:
            norm = np.sqrt(np.square(pts_gt[15][0] - pts_gt[16][0]) + np.square(pts_gt[15][1] - pts_gt[16][1]))
            if np.isnan(norm):
                print(pts_gt[15])
                print(pts_gt[16])
        if norm == 0:
            norm = 256
        if n==0:
            return 0
        try:
            d = (dis / n) / norm
        except:
            print('except: ' + norm)
            return 0
        return d
    def calculate_AP(self,GT_FILE,PRED_FILE):
        gt_data= self.build_data_dict_for_ap(GT_FILE)
        pred_data = self.build_data_dict_for_ap(PRED_FILE)
        filter_cat = [self.catgory]
        n = 0
        dis = 0
        for (name, v) in gt_data.items():
            if v['type'] not in filter_cat:
                continue
            d = self.pts_avg_distance(v['joints'], pred_data[name]['joints'], v['type'])
            dis = dis + d
            n = n + 1
        dis = dis / n
        print('err: ', dis * 100)

    def show_top_n_err_samples(self,GT_FILE,PRED_FILE,topN=100):
        gt_data = self.build_data_dict_for_ap(GT_FILE)
        pred_data = self.build_data_dict_for_ap(PRED_FILE)
        filter_cat = [self.catgory]
        loss_dict = {}
        n = 0
        dis = 0
        for (name, v) in gt_data.items():
            if v['type'] not in filter_cat:
                continue
            d = self.pts_avg_distance(v['joints'], pred_data[name]['joints'], v['type'])
            loss_dict[name] = d
            dis = dis + d
            n = n + 1
        dis = dis / n

        dict = sorted(loss_dict.items(), key=lambda d: d[1], reverse=True)
        N = topN
        i = 0
        ff=open('hard.csv','w')
        for (name, v) in dict:
            ff.write(name+'\n')
            i = i + 1
            if i > N:
                break
            src, preprocessed = self.img_crop_resize(name)
            pts1 = pred_data[name]['joints']
            for p in pts1:
                cv2.circle(src, (p[0], p[1]), 2, (0, 255, 0), 2, 16)
            pts2 = gt_data[name]['joints']
            for i, p in enumerate(pts2):
                if p[2] == 1:
                    cv2.circle(src, (p[0], p[1]), 2, (0, 0, 255), 2, 16)
                    cv2.line(src, (p[0], p[1]), (pts1[i][0], pts1[i][1]), (0, 0, 255), 1, 16)

            pred_in_crop = self.pred_dict[name]
            for p in pred_in_crop:
                    cv2.circle(preprocessed, (int(p[1]),int(p[0])), 2, (0, 255, 0), 2, 16)
        ff.close()
            # print(name + ' loss: ', v)
            # cv2.namedWindow('pre', 0)
            # cv2.imshow('pre', preprocessed)
            # cv2.namedWindow('src', 0)
            # cv2.imshow('src', src)
            # cv2.waitKey(0)
    def save_seg_bbox_result(self,filename,catgory):
        f = open(filename, 'a')

        for name in self.img_list:
            content = name
            for p in self.final_pred_dict[name]:
                content = content + ',' + str(p[0]) +','+str(p[1])
            f.write(content + '\n')
        f.close()

    def write_result_to_csv(self,filename, model_catgory_list,save_catgory):

        catgory =  model_catgory_list[0]
        for i in range(1, len( model_catgory_list)):
            catgory += '_' +  model_catgory_list[i]
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
        if catgory=='blouse_outwear' or catgory=='outwear_blouse':
            inx_list= blouse_outwear_inx_list
        f = open(filename, 'w')
        header = 'image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out'
        f.write(header+'\n')
        for name in self.img_list:
            content = self.to_one_submit_str(name,save_catgory, self.final_pred_dict[name],inx_list)
            f.write(content+'\n')
        f.close()



    def img_crop_resize(self,name):
        src=  cv2.imread(self.img_dir+name)
        if self.has_gt_joints:
            expandL = self.rr * np.min([src.shape[0], src.shape[1]])
            box = [0,0,0,0]
            bb = self.bbox_data_dict[name]
            bb[0] = min(bb[0],bb[2])
            bb[1] = min(bb[1], bb[3])
            bb[2] = max(bb[0], bb[2])
            bb[3] = max(bb[1], bb[3])

            box[0] =int( bb[0] - expandL if bb[0] - expandL > 0 else 0)
            box[1] =int( bb[1] - expandL if bb[1] - expandL > 0 else 0)
            box[2] =int( bb[2] + expandL if bb[2] + expandL < src.shape[1] else src.shape[1]-1)
            box[3] =int( bb[3] + expandL if bb[3] + expandL <src.shape[0] else src.shape[0]-1)
            self.bbox_data_dict[name] = box.copy()

        b = self.bbox_data_dict[name]
        image = src[b[1]:b[3],b[0]:b[2]]


        pts=[]
        pts.append(ia.Keypoint(x=0, y=0))
        keypoints = ia.KeypointsOnImage(pts, shape=image.shape)
        r = self.img_size[0] / self.img_size[1]
        pad = max(image.shape[0], image.shape[1]) * min(r, 1 / r) - min(image.shape[0], image.shape[1])
        pad = int(pad / 2)
        if pad < 0:
            self.in_box_image_size_dict[name] = image.shape
            pad = 0
        if image.shape[0] > image.shape[1]:
            self.in_box_image_size_dict[name] = (image.shape[0],image.shape[1]+2*pad)
            px = (0, pad, 0, pad)
        else:
            self.in_box_image_size_dict[name] = (image.shape[0]+2*pad, image.shape[1] )
            px = (pad, 0, pad, 0)

        seq = iaa.Sequential([
                iaa.CropAndPad(px=px),
                iaa.Scale({'height': self.img_size[0], 'width': self.img_size[1]})
        ])

        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
        # x = int(keypoints_aug.keypoints[0].x)
        # y = int(keypoints_aug.keypoints[0].y)
        x = pad
        y = 0
        if image.shape[0] < image.shape[1]:
            x = 0
            y = pad

        self.crop_new_x_y[name] = [x,y]
        return  src, image_aug


    def preprocess_img(self,name,color_mode):
        #print('preprocess file: ',name)

        src, img_crop_resize= self.img_crop_resize(name)
        if color_mode == 'RGB':
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        if color_mode == 'RGB':
            img_crop_resize = cv2.cvtColor(img_crop_resize, cv2.COLOR_BGR2RGB)
        return src,img_crop_resize



    #  检测preprocessed  图之后 登记检测到的点
    def reg_pred(self,name, pred, isXFisrt =False):
        self.isXFirst = isXFisrt
        self.pred_dict[name] = pred
        raw_size = self.in_box_image_size_dict[name]
        b = self.bbox_data_dict[name]
        c = self.crop_new_x_y[name]
        final_pred=[]
        for p in pred:
            y = p[0]
            x = p[1]
            if isXFisrt:
                y=p[1]
                x=p[0]
            x = int(np.round(x * raw_size[1] / self.img_size[1] +b[0] - c[0]))
            y = int(np.round(y * raw_size[0] / self.img_size[0] +b[1] - c[1]))
            final_pred.append([x,y])
        self.final_pred_dict[name] = final_pred
    def draw_final_pred(self,src, name):
        final_pred = self.final_pred_dict[name]
        for p in final_pred:
            cv2.circle(src, tuple(p), 3, (0, 255, 0), 3, 16)




    def generator(self,color_mode='BGR'):
        for name in self.img_list:
            src, img_crop_resize =self.preprocess_img(name,color_mode)
            yield src,img_crop_resize,name

    def build_data_dict_for_ap(self,filename):
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

            def fn(x):
                c = x.split('_')
                return list(map(int, c[:]))

            joints = list(map(fn, line[2:]))
            joints = np.reshape(joints, (-1, 3))
            data_dict[name] = {'joints': joints, 'type': type}
        input_file.close()
        return data_dict

    def _build_data_dict(self,filename,catogory):
        input_file = open(filename, 'r')
        data_dict = {}
        bbox_dict={}
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
            if type!=catogory:
                continue
            def fn(x):
                c = x.split('_')
                return list(map(int, c[:]))
            joints = list(map(fn, line[2:]))
            joints_all = np.reshape(joints, (-1, 3))
            joints = self._get_self_catogory_joints(joints_all,type)
            valid_joints = joints[np.where(joints[:,2]!=-1)][:,0:2]
            b = cv2.boundingRect(np.array(valid_joints))
            bb = [b[0],b[1],b[0]+b[2],b[1]+b[3]]
            data_dict[name] = {'joints': joints[:,0:2],'joints_3': joints_all, 'visible':joints[:,2],'type': type}
            bbox_dict[name] = bb
        input_file.close()
        return data_dict,bbox_dict

    ## 从所有24点中滤除某类的点
    def _get_self_catogory_joints(self, joints, catogory):
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

    def build_bbox_dict(self,bboxAnnoFile):
        input_file = open(bboxAnnoFile, 'r')
        data_dict = {}
        for line in input_file:
            line = line.strip()
            line = line.split(',')
            name = line[0]
            b0 = list(map(float, line[1:5]))
            b0 = [int(b0[0]), int(b0[1]), int(b0[2]), int(b0[3])]
            data_dict[name] = b0
        input_file.close()
        return data_dict


if __name__ =='__main__':
    img_dir = "F:\\fashionAI_key_points_train_20180227\\train\\"
    csv_file = './train_test_split/total_train.csv'
    data = TestData(img_dir,csv_file,'outwear')
    data_iter =  data.generator()
    for src, preprocessed,name in data_iter:
        cv2.namedWindow('src',0)
        cv2.imshow('src',src)

        cv2.circle(preprocessed,tuple(data.crop_new_x_y[name]), 2,(0,0,255),2,8)
        cv2.namedWindow('preprocessed', 0)
        cv2.imshow('preprocessed', preprocessed)
        cv2.waitKey(0)
