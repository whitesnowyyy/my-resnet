import random
import cv2
import numpy as np

from albumentations.core.transforms_interface import DualTransform

class RandomAffine(DualTransform):

    # @property
    # def targets(self):
    #     return {
    #         "image": self.apply,
    #         "bboxes": self.apply_to_bboxes,
    #         "keypoints": self.apply_to_keypoints,
    #     }
    
    # def to_tuple(self, value, add_value=1):
    #     if isinstance(value, tuple):
    #         return value
    #     else:
    #         return (add_value-value, add_value+value)
    
    def __init__(self, width, 
                     height,
                    angle_limit=(-45, +45), scale_limit=(0.8, 1.2), 
                     offset_limit=(0.4, 0.6), rotate_angle_threshold=30,
                min_size = 10,always_apply=False, p=1.0):
        super(RandomAffine, self).__init__(always_apply=always_apply, p=p)
        self.width = width
        self.height = height
        self.rotate_angle_threshold = rotate_angle_threshold
        # self.scale_limit = scale_limit
        # self.offset_limit = offset_limit
        # self.angle_limit = angle_limit
        # self.min_size = min_size
        self.scale_limit = self.to_tuple(scale_limit, 1)
        self.offset_limit = self.to_tuple(offset_limit, 0.5)#???
        self.angle_limit = self.to_tuple(angle_limit, 0)
        self.min_size = self.to_tuple(min_size,None)

    def apply(self, img, M, **params):
        return cv2.warpAffine(img, M, (self.width, self.height))

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        height, width = image.shape[:2]
        
        scale = 1
        angle = 0
        cx = 0.5 * width
        cy = 0.5 * height
        
        angle = random.uniform(*self.angle_limit)
        scale = random.uniform(*self.scale_limit)
        cx = random.uniform(*self.offset_limit) * width
        cy = random.uniform(*self.offset_limit) * height
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] -= cx - self.width * 0.5
        M[1, 2] -= cy - self.height * 0.5
        return {"M": M, "scale": scale, "angle": angle,'image_width':width,"image_height":height}
    
    def apply_to_bboxes(self, bboxes, M, scale, angle, image_width,image_height,**params):
        
        if len(bboxes) == 0:
            return []
        #两个中括号，1*4
        np_image_size = np.array([[image_width,image_height,image_width,image_height]])
        #一个中括号，（4，）
        np_output_size = np.array([self.width,self.height,self.width,self.height])

        # 获取每个box的中心位置, nx2
        tail = np.array([item[4:] for item in bboxes])
        npbboxes = np.array([item[:4] for item in bboxes])*np_image_size #恢复到原始的框
        np_bboxes_center = np.array([[(x + r) * 0.5, (y + b) * 0.5] for x, y, r, b in npbboxes])
        
        # 将nx2转换为2xn
        np_bboxes_center_t_2row = np_bboxes_center.T
        
        # 增加维度，变换为3xn，第三个行是全1
        one_row = np.ones((1, np_bboxes_center_t_2row.shape[-1]))
        np_bboxes_center_coordinate = np.vstack([np_bboxes_center_t_2row, one_row])
        
        # 变换
        project = M @ np_bboxes_center_coordinate
        
        # 转换为list
        list_project = project.T.tolist()
        
        # scale乘以0.5，是return时cx - scale * width的转换。真值为cx - scale * 0.5 * width。此时合并scale * 0.5
        half_scale = scale * 0.5
        
        result = np.array([[
            cx - (r - x + 1) * half_scale, 
            cy - (b - y + 1) * half_scale, 
            cx + (r - x + 1) * half_scale, 
            cy + (b - y + 1) * half_scale
        ] for (x, y, r, b), (cx, cy) in zip(npbboxes, list_project)])
        
        # 限制框不能超出范围
        x, y, r, b = result[:, 0], result[:, 1], result[:, 2], result[:, 3]
        x[...] = x.clip(min=0, max=self.width-1)
        y[...] = y.clip(min=0, max=self.height-1)
        r[...] = r.clip(min=0, max=self.width-1)
        b[...] = b.clip(min=0, max=self.height-1)
        w = (r - x + 1).clip(min=0)
        h = (b - y + 1).clip(min=0)

        cond = (w >= 10) & (h >= 10)#限制图片大小，min_size
        
        # 对于tail0，认为是是否存在旋转，即目标是否大于30度
        if abs(angle) > 30 and len(tail) > 0:
            tail[cond, 0] = True#创造布尔索引，筛选
        return [list(coord/np_output_size) + list(tail_item) for coord, tail_item in zip(result[cond], tail[cond])]
        #除以输出的尺寸
    def apply_to_keypoints(self, keypoint, **params):
        raise NotImplementedError
    @property
    def targets(self):
        return{
            "image":self.apply,
            "bboxes":self.apply_to_bboxes
        }

    
    def to_tuple(self,value,add_value=1):
        if isinstance(value,tuple):
            return value
        elif add_value is not None:
            return(add_value-value,add_value+value)
        else:
            return (value,value)

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return "height", "width", "scale_limit", "angle_limit", "offset_limit","rotate_angele_threshold"
    
# trans = A.Compose([
#     RandomPerspective(800, 800),
#     A.ShiftScaleRotate(rotate_limit=20),
#     A.HorizontalFlip(),
# ], bbox_params=A.BboxParams("pascal_voc", label_fields=["labels"]))

# out = trans(image=cv_image, bboxes=[item.location + [False] for item in image.bboxes], labels=["face" for item in image.bboxes])
# show(out["image"], out["bboxes"])
if __name__ == "__main__":
    m = RandomAffine(800,800)
    print(m)