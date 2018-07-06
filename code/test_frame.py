import  numpy as np
import  cv2
import imgaug as ia
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from dataset.data_common import *

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

class TestData(Dataset):
    '''
    img_dir: 图片所在根目录
    csv_file: 阿里提供的test.csv (只含有类别信息)
    bbox_file: 检测器产生的bbox 文件
    '''
    def __init__(self,img_dir,csv_file,catgory_list,inx_list, bbox_file=None,img_size=(300,300),expandS=0.0):
        self.expandS = expandS
        self.inx_list=inx_list
        self.catgory_list = catgory_list
        self.img_size = img_size
        self.img_dir = img_dir
        self.bbox_file = bbox_file
        self.csv_file = csv_file

        self.img_list,self.type_dict= self.get_test_imglist_and_type(csv_file,catgory_list)
        if bbox_file is not None:
           self.bbox_data_dict = self.build_bbox_dict(bbox_file)
        print('images: ', len(self.img_list))
        self.crop_new_x_y={}
        self.pred_dict={}
        self.final_pred_dict={}
        self.in_box_image_size_dict={}
    def get_test_imglist_and_type(self,cvs_file,catgory_list):
        file_list=[]
        type_dict={}
        input_file = open(cvs_file, 'r')
        i = 0
        for line in input_file: ## remove header
            if i == 0:
                i = 1
                continue
            line = line.strip()
            line = line.split(',')
            name = line[0]
            type = line[1]
            if type not in catgory_list:
                continue
            file_list.append(name)
            type_dict[name] = type
        input_file.close()
        return file_list,type_dict

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



    def write_result_to_csv(self,filename):
        f = open(filename, 'w')
        header = 'image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out'
        f.write(header+'\n')
        for name in self.img_list:
            content = self.to_one_submit_str(name,self.type_dict[name], self.final_pred_dict[name],self.inx_list)
            f.write(content+'\n')
        f.close()


    def img_crop_resize(self,name):
        src=  cv2.imread(self.img_dir+name)
        b = [0,0,src.shape[1]-0, src.shape[0]-1]
        if self.bbox_file is not  None:
            try:
                bb = self.bbox_data_dict[name]
            except:
                print('no bbox')
                bb = [0, 0, src.shape[1] - 0, src.shape[0] - 1]
            b0 = [0, 0, 0, 0]
            b0[0] = np.min([bb[0], bb[2]])
            b0[1] = np.min([bb[1], bb[3]])
            b0[2] = np.max([bb[0], bb[2]])
            b0[3] = np.max([bb[1], bb[3]])
            expandL = self.expandS * np.min([b0[2] - b0[0], b0[3] - b0[1]])
            b1 = [0, 0, 0, 0]
            b1[0] = int(b0[0] - expandL if b0[0] - expandL > 0 else 0)
            b1[1] = int(b0[1] - expandL if b0[1] - expandL > 0 else 0)
            b1[2] = int(b0[2] + expandL if b0[2] + expandL < src.shape[1] else src.shape[1] - 1)
            b1[3] = int(b0[3] + expandL if b0[3] + expandL < src.shape[0] else src.shape[0] - 1)
            self.bbox_data_dict[name] = b1.copy()
            b = self.bbox_data_dict[name]

        image = src[b[1]:b[3],b[0]:b[2]]
        # image = image.astype(np.float32)
        #
        # _R_MEAN = 123.68
        # _G_MEAN = 116.78
        # _B_MEAN = 103.94
        # image[:, :, 0] = image [:, :, 0] - _R_MEAN
        # image[:, :, 1] = image [:, :, 1] - _G_MEAN
        # image[:, :, 2] = image [:, :, 2] - _B_MEAN

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
                iaa.CropAndPad(px=px,pad_cval=114),
                iaa.Scale({'height': self.img_size[0], 'width': self.img_size[1]})
        ])
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        x = pad
        y = 0
        if image.shape[0] < image.shape[1]:
            x = 0
            y = pad
        self.crop_new_x_y[name] = [x,y]
        return  src, image_aug

    def preprocess_img(self,name,color_mode):
        src, img_crop_resize= self.img_crop_resize(name)
        if color_mode == 'RGB':
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        if color_mode == 'RGB':
            img_crop_resize = cv2.cvtColor(img_crop_resize, cv2.COLOR_BGR2RGB)
        return src,img_crop_resize

    #  检测preprocessed  图之后 登记检测到的点
    def reg_pred(self,src,name, pred, isXFisrt =False):
        self.isXFirst = isXFisrt
        self.pred_dict[name] = pred
        raw_size = self.in_box_image_size_dict[name]
        b = [0, 0, src.shape[1] - 0, src.shape[0] - 1]
        if self.bbox_file is not None:
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
        b = self.bbox_data_dict[name]
        cv2.rectangle(src,(b[0],b[1]),(b[2],b[3]),(0,0,255),3,16)
        for i,p in enumerate(final_pred):
            cv2.circle(src, tuple(p), 2, (0, 0, 255),2, 16)
            text_loc = (p[0] + 5, p[1] + 7)
            cv2.circle(src, tuple(p), 3, (0, 0, 255), -1)
            cv2.putText(src, str(self.inx_list[i]), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        name = self.img_list[item]
        src, img_crop_resize = self.preprocess_img(name, 'RGB')

        img = img_crop_resize.astype(np.float32)
        img[:, :, 0] = img[:, :, 0] - _R_MEAN
        img[:, :, 1] = img[:, :, 1] - _G_MEAN
        img[:, :, 2] = img[:, :, 2] - _B_MEAN

        img = img.transpose(2,0,1)


        img = torch.from_numpy(img).float()

        return img,src,name

    # def generator(self,color_mode='RGB'):
    #     for name in self.img_list:
    #         src, img_crop_resize =self.preprocess_img(name,color_mode)
    #         yield src,img_crop_resize,name


    def build_bbox_dict(self,bboxAnnoFile):
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

