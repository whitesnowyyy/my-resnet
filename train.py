
import torch
import torchvision.transforms as T
import torchvision.datasets.cifar as cifar
import torch.nn as nn
import torch.optim
import resnet
from torch.utils.data import DataLoader
import utils
import argparse
import json


import detection
import dataset
import losses
import os
import torch.onnx
import torch.nn.functional as F
import cv2


class Config:
    def __init__(self):
        self.experiment_name = "default"
        self.model_name = "resnet50"
        self.batch_size = 32
        self.gpu = 0
        self.weight = ''
        self.height = 800
        self.width = 800
        self.about = ""
        # self.base_directory = 'worksapce'
        

    def get_path(self, path):
        return f"workspace/{self.experiment_name}/{path}"

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)


def eval_model(model, val_dataloader, device):
    model.eval()
    correct = 0
    for batch_index, (images, targets) in enumerate(val_dataloader):
        images = images.to(device)
        targets = targets.to(device)
        feature = model(images)
        predict = torch.argmax(feature, dim=1)
        correct += (targets == predict).sum().item()
    accuracy = (correct / len(val_dataloader.dataset))
    return accuracy

def train():
    utils.setup_seed(31)
    utils.mkdirs(config.get_path("models"))
    utils.copy_code_to('.',config.get_path('code'))#??
    # device = config.gpu
    device = 'cpu'


    train_set = dataset.Dataset(config.width,config.height,'label-1.txt','images')##??
    train_dataloader = DataLoader(train_set,batch_size=config.batch_size,num_workers=16,shuffle=True,pin_memory=True)

    # train_transform = T.Compose([
    #     T.RandomCrop(32, padding=4, fill=(128, 128, 128)),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])

    # val_transform = T.Compose([
    #     T.ToTensor(),
    #     T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])

    # train_set = cifar.CIFAR10("/data-rbd/wish/four_lesson/dataset", train=True, transform=train_transform)
    # val_set = cifar.CIFAR10("/data-rbd/wish/four_lesson/dataset", train=False, transform=val_transform)

    # train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=8, shuffle=True)
    # val_dataloader = DataLoader(val_set, batch_size=128, num_workers=8, shuffle=False)



    # resnet_module_all_variables = vars(resnet)
    # if config.model_name not in resnet_module_all_variables:
    #     logger.error(f"Can not found model: {config.model_name}")
    #     return

    # model_funciton = resnet_module_all_variables[config.model_name]
    # model = model_funciton(10, small_input=True)
    model = detection.Detection()

    if config.weight !='':
        if os.path.exists(config.weight):
            print(f'Load weight:{config.weight}')
            check_point = torch.load(config.weight)#??
            model.load_state_dict(check_point)
        else:
            logger.error(f'weight not exist:{config.weight}')
    model.to(device)


    glr = 2e-3
    optimizer = torch.optim.Adam(model.parameters(), glr)
    coordinate_loss_function = losses.GIoULoss()
    point_loss_function = losses.focal_loss
    stride = 4
    heatmap_width,heatmap_height = config.width//stride,config.height//stride
    gy,gx = torch.meshgrid(torch.arange(heatmap_height),torch.arange(heatmap_width))
    cell_grid = torch.stack((gx,gy,gx,gy),dim=0).unsqueeze(0).to(device)
    # 每个epoch需要迭代多少次
    num_iter_per_epoch = len(train_dataloader)

    epochs = 10
    lr_schedule = {
        1: 1e-3,
        100: 1e-4,
        130:1e-5
    }

    for epoch in range(epochs):

        # 学习率修改
        if epoch in lr_schedule:

            glr = lr_schedule[epoch]
            for group in optimizer.param_groups:
                group["lr"] = glr

        # mean_loss = 0
        num_iter =0
        model.train()
        for batch_index, (images, raw_images,point_targets,coord_targets,mask_targets) in enumerate(train_dataloader):
            images = images.to(device)
            point_targets =point_targets.to(device)
            coord_targets = coord_targets.to(device)
            mask_targets = mask_targets.to(device)

            point_predict,coord_predict = model(images)
            point_logits = point_predict.sigmoid()

            coord_predict = ((coord_predict+cell_grid)*stride).permute(0,2,3,1)
            coord_targets = coord_targets.permute(0,2,3,1)
            mask_targets = mask_targets.permute(0,2,3,1)
            coord_restore = coord_predict[mask_targets].view(-1,4)
            coord_select = coord_targets[mask_targets].view(-1,4)
            coordinate_loss = coordinate_loss_function(coord_restore,coord_select)
            point_loss = point_loss_function(point_logits,point_targets)
            loss = point_loss +coordinate_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_iter += 1

            if num_iter %100 ==0:
                current_epoch = epoch +batch_index/num_iter_per_epoch
                logger.info(f'iter:{num_iter},epoch:{current_epoch:.2f},loss:{loss.item():.3f},Point:{point_logits_seclect}')

                show_batch = -1
                point_logits_seclect = point_logits[show_batch].detach().unsqueeze(0)
                point_pooded = F.max_pool2d(point_logits_seclect,kernel_size=(3,3),stride=1,padding=1)
                ys,xs = torch.where((point_logits_seclect[0,0]== point_pooded[0,0])&(point_logits_seclect[0,0]>0.3))
                raw_image_select = raw_images[show_batch].numpy()
                for y,x in zip(ys,xs):
                    px,py,pr,pb = coord_predict[show_batch,y,x,:].detach().cpu().long()
                    cv2.rectangle(raw_image_select,(px,py),(pr,pb),(0,255,0),2)


        #     mean_loss += loss.item()

        # mean_loss /= num_iter_per_epoch
        # accuracy = eval_model(model, val_dataloader,  device)

        if epoch % 5 == 0:
            save_path = config.get_path(f"models/{epoch:03d}.pth")
            logger.info(f"Save model to {save_path}")
            torch.save(model.state_dict(), save_path)

        logger.info(f"Epoch: {epoch} / {epochs}, Loss: {loss:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--weight", type=str, help="preModel", default="")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    parser.add_argument("--gpu", type=str, help="GPU id", default="cuda:0")
    parser.add_argument('--about',type=str,help='说明',default='')
    args = parser.parse_args()
    
    config = Config()
    # config.model_name = args.model
    config.experiment_name = args.name
    config.batch_size = args.batch_size
    config.gpu = args.gpu
    config.weight = args.weight
    config.height = 800
    config.width = 800
    config.about = args.about
    logger = utils.getLogger(config.get_path("/logs/log.log"))
    logger.info("Startup, config is:")
    logger.info(config)
    train()