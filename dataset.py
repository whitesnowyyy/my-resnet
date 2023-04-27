from symbol import argument
import common
import augment
import albumentations as A
# from albumentations.pytorch import ToTensorV2,ToTensor

import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F #??
import torchvision.transforms as T

def bbox_detection_augment(outwidth,outheight,mean,std):
    return  A.Compose(
        [augment.RandomAffine(outwidth, outheight,scale_limit=(0.8,3.0)),
        A.HorizontalFlip(),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=27),
            A.RandomContrast(limit=0.8),
            A.JpegCompression(quality_lower=5, quality_upper=100),
        ]),
        # A.OneOf([
        #     A.ISONoise(),
        #     # A.IAAAdditiveGaussianNoise(),
        #     A.IAASharpen(),
        # ]),
        # A.OneOf([
        #     A.Cutout(num_holes=32, max_h_size=24, max_w_size=24, p=0.5),
        #     A.RandomRain(p=0.2),
        #     A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
        #     A.IAAPerspective(p=0.5)
        # ]),
        A.OneOf([
            A.Blur(blur_limit=9),
            A.MotionBlur(p=1, blur_limit=7),
            A.GaussianBlur(blur_limit=21),
            A.GlassBlur(),
            A.ToGray(),
            A.RandomGamma(gamma_limit=(0, 120), p=0.5),
        ]),
        #ToTensorV2()
    ], bbox_params=A.BboxParams("pascal_voc"))#没有 label_fields=["labels"

class Dataset:
    def __init__(self,width,height,annotation,root):
        mean = [0.4914,0.4822,0.4465]
        std = [0.2023,0.1994,0.2010]
        self.width = width
        self.height = height
        self.transform = bbox_detection_augment(width,height,mean,std)
        self.annotation = annotation
        self.root = root
        self.convert_to_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=mean,std=std)
            ]
        )
        assert os.path.exists(annotation),f'{annotation} not exists'
        self.images = common.load_widerface_annotation(annotation)#加载了标注信息


    def __getitem__(self,index):
        item = self.images[index]
        image_path = f'{self.root}/{item.file}'
        image = cv2.imread(image_path)

        bboxes = [box.location +(False,) for box in item.bboxes if box.width >=10 and box.height>=10]#??

        if image is None:
            print(f'empty image:{image_path}')

            image = (np.random.normal(size=(self.height,self.width,3))*255).astype(np.uint8)#也一定是uint8
            bboxes = []
        try:
            trans_out = self.transform(image=image,bboxes = bboxes)
        except Exception as e:
            print(e,image_path,index)
            raise e

        stride = 4
        heatmap_height = self.height//stride
        heatmap_width = self.width // stride
        point_heatmap = np.zeros((1,heatmap_height,heatmap_width),dtype=np.float32)
        coord_heatmap = np.zeros((4,heatmap_height,heatmap_width),dtype=np.float32)
        mask_heatmap = np.zeros((4,heatmap_height,heatmap_width),dtype=np.bool)

        
        for x,y,r,b,invalid in trans_out['bboxes']:
            cx,cy = (x+r)*0.5,(y+b)*0.5
            box_width,box_height =(r-x+1)/stride,(b-y+1)/stride
            cell_x,cell_y = int(cx/stride +0.5),int(cy/stride +0.5)
            cell_x = max(0,min(cell_x,heatmap_width-1))
            cell_y = max(0,min(cell_y,heatmap_height-1))
            common.draw_gauss(point_heatmap[0],cell_x,cell_y,(box_width,box_height))
            if not invalid:
                coord_heatmap[:,cell_y,cell_x] = x,y,r,b
                mask_heatmap[:cell_y,cell_x] = True
        return self.convert_to_tensor(trans_out['image']),trans_out['image'],torch.as_tensor(point_heatmap),torch.as_tensor(coord_heatmap),torch.as_tensor(mask_heatmap)



    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = Dataset(800,800,'label-1.txt','image')
    print(len(dataset))
    image ,raw_image,point ,coord,mask = dataset[6857]
    print(point.shape)

