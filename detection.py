import resnet
import torch.nn as nn
import torch.nn.functional as F

def upmodule(in_feature, out_feature, scale=2):
    # Upsample + Conv + BN
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="nearest"),
        nn.Conv2d(in_feature, out_feature, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_feature)
    )

def projection_module(in_feature, out_feature):
    # Conv + BN
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(out_feature)
    )

def point_head(in_feature,out_feature):
    point = nn.Conv2d(in_feature,out_feature,kernel_size=1,padding=0,stride=1)
    point.weight.data.normal_(std=0.01)
    point.bias.data.fill_(-2.94443897916664403)
    return point
def coord_head(in_feature,out_feature):
    coord = nn.Conv2d(in_feature,out_feature,kernel_size=1,padding=0,stride=1)
    coord.weight.data.normal_(std=0.01)
    coord.bias.data.fill_(0.0)
    return coord

class Detection(nn.Module):
    def __init__(self):
        super().__init__()
        
        model = resnet.resnet18()
        self.layer0 = model.layer0  # 4     c=64
        self.layer1 = model.layer1  # 4     c=64
        self.layer2 = model.layer2  # 8     c=128
        self.layer3 = model.layer3  # 16    c=256
        self.layer4 = model.layer4  # 32    c=512
        
        self.conv_proj = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        wide = 64

        self.conv_32=nn.Sequential(
            nn.Conv2d(256,wide,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(wide),
            nn.ReLU(),
        )

        self.u4 = upmodule(512, wide)
        self.p3 = projection_module(256, wide)
        
        self.u3 = upmodule(wide, wide)
        self.p2 = projection_module(128, wide)
        
        self.u2 = upmodule(wide, wide)
        self.p1 = projection_module(64, wide)

        self.point_4 = point_head(wide,1)
        self.coordinate_4 = coord_head(wide,4)

        self.point_8 = point_head(wide,1)
        self.coordinate_8 = coord_head(wide,4)

        self.point_16 = point_head(wide,1)
        self.coordinate_16 = coord_head(wide,4)

        self.point_32 = point_head(wide,1)
        self.coordinate_32 = coord_head(wide,4)

        

        
        # self.point = nn.Conv2d(wide, 1, kernel_size=1, padding=0, stride=1)
        # self.point.weight.data.normal_(std=0.001)
        # self.point.bias.data.fill_(-2.9444389791664403)
        
        # self.coordinate = nn.Conv2d(wide, 4, kernel_size=1, padding=0, stride=1)
        # self.coordinate.weight.data.normal_(std=0.001)
        # self.coordinate.bias.data.fill_(0)
        
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x_proj = self.conv_proj(x4)
        x_32 = self.conv_32(x_proj)
        
        u4 = self.u4(x4)
        p3 = self.p3(x3)
        o4 = F.relu(u4 + p3)  # 16倍
        
        u3 = self.u3(o4)
        p2 = self.p2(x2)
        o3 = F.relu(u3 + p2)  # 8倍
        
        u2 = self.u2(o3)
        p1 = self.p1(x1)
        o2 = F.relu(u2 + p1) 
        
        point_32 = self.point_32(x_32)
        coord_32 = self.coordinate_32(x_32)

        point_16 = self.point_16(o4)
        coord_16 = self.coordinate_32(o4)

        point_8 = self.point_8(o3)
        coord_8 = self.coordinate_8(o3)

        point_4 = self.point_4(o2)
        coord_4 = self.coordinate_4(o2)

        point_32 = self.point_32(x_32)
        coord_32 = self.coordinate_32(x_32)

        
        # return point_4,coord_4,point_8,coord_8,point_16,coord_16,point_32,coord_32,#???????????
        #  # 4倍
        # point = self.point(o2)
        # coordinate = self.coordinate(o2)
        return point_4,coord_4