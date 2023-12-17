import os
from os.path import join
import random
from glob import glob
from PIL import Image
from datetime import datetime
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import monai
from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from skimage import transform
device='cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'
torch.cuda.empty_cache()
# model_path='./res/finetune/best_model.pth'
model_path='./model/sam_vit_h_4b8939.pth'
# model_path='./res/finetune/best1.pth'
# sam_model=torch.load(model_path)['model']
# torch.save(models,'./res/finetune/models_best.pth')
sam_model=sam_model_registry['vit_h'](checkpoint=model_path).to(device)
predictor=SamPredictor(sam_model)

num2color={
    0:[0,0,0],
    1:[0, 153, 255],
    2:[102, 255, 153],
    3:[0, 204, 153],
    4:[255, 255, 102],
    5:[255, 255, 204],
    6:[255, 153, 0],
    7:[255, 102, 255],
    8:[102, 0, 51],
    9:[255, 204, 255],
    10:[255, 0, 102]
}
num2label={
    0:'background',
    1:'skin',
    2:'left eyebrow',
    3:'right eyebrow',
    4:'left eye',
    5:'right eye',
    6:'nose',
    7:'upper lip',
    8:'inner mouth',
    9:'lower lip',
    10:'hair'
}
class LaPaDataset(Dataset):
    def __init__(self,data_path,label_path):
        self.data_path=data_path
        self.label_path=label_path

        self.data=sorted(glob(os.path.join(self.data_path,'*.jpg')))
        self.label=sorted(glob(os.path.join(self.label_path,'*.png')))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_path=self.data[index]
        label_path=self.label[index]
        img=Image.open(data_path)
        img=img.resize((1024,1024),Image.BICUBIC)
        img=np.array(img)
        
        label=Image.open(label_path)
        label=np.array(label)
        label_ids=np.unique(label)[1:]
        if label_ids[0]==None:
            return (None,None,None,None)
        i=random.choice(label_ids.tolist())
        gt_mask=np.uint8(label==i)
        gt_mask=np.uint8(transform.resize(
            gt_mask,
            (1024,1024),
            order=0,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        ))
        gt_mask[gt_mask>0.3]=1
        gt_mask[gt_mask!=1]=0
        img=(img-img.min())/(img.max()-img.min())
        img=img.transpose((2,0,1))

            # if (gt_mask.any()==i)==False:
            #     center_list.append(None)
            #     bbox_list.append(None)
            #     continue
        y_idx,x_idx=np.where(gt_mask==1)
        y_min,y_max=np.min(y_idx),np.max(y_idx)
        x_min,x_max=np.min(x_idx),np.max(x_idx)
        dt_mask=cv2.distanceTransform(gt_mask[y_min:y_max+1,x_min:x_max+1],cv2.DIST_L2,3)
        local_coords=np.unravel_index(np.argmax(dt_mask,),dt_mask.shape)
        center_point=np.expand_dims(
            np.array([local_coords[1],local_coords[0]])+np.array([x_min,y_min]),axis=0)
            
        img=torch.tensor(img).float()
        label_list=torch.tensor(gt_mask).long()
        center_list=torch.tensor(center_point).long()
        bbox_list=torch.tensor([x_min,y_min,x_max,y_max]).long()

        return (img,label_list,center_list,bbox_list)
    
train_lapa_set=LaPaDataset(data_path='./LaPa/train/images',label_path='./LaPa/train/labels')
train_lapa_loader=DataLoader(train_lapa_set,batch_size=4,shuffle=True)

class finetuneSAM(nn.Module):
    def __init__(
            self,
            image_encoder:ImageEncoderViT,
            mask_decoder:MaskDecoder,
            prompt_encoder:PromptEncoder,

    ):
        super().__init__()
        self.image_encoder=image_encoder
        self.mask_decoder=mask_decoder
        self.prompt_encoder=prompt_encoder

        for param in self.image_encoder.parameters():
            param.requires_grad=False
        for param in self.prompt_encoder.parameters():
            param.requires_grad=False

    def forward(self,image:torch.Tensor,prompt:torch.Tensor,type:str):
        with torch.no_grad():
            image_embedding=self.image_encoder(image)
            if type=="single":
                label=torch.ones(size=prompt.shape[:-1],dtype=torch.long,device=prompt.device)
                sparse_embeddings,dense_embeddings=self.prompt_encoder(
                    points=(prompt,label),boxes=None,masks=None,
                )
            else:
                sparse_embeddings,dense_embeddings=self.prompt_encoder(
                    points=None,boxes=prompt,masks=None,
                )
        low_res_masks,_=self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks=F.interpolate(low_res_masks,size=(image.shape[2],image.shape[3]),mode='bilinear',
                                      align_corners=False,)
        return ori_res_masks
    
def train(model,dataloader,output_path='./res/finetune'
          ,data_path='./LaPa/train/images',label_path='./LaPa/train/labels'):
    os.makedirs(output_path, exist_ok=True)
    sam_model = model
    finetune_model = finetuneSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    finetune_model.train()

    optimizer=torch.optim.Adam(finetune_model.mask_decoder.parameters(),lr=1e-4,weight_decay=1e-4)
    seg_loss=monai.losses.DiceLoss(sigmoid=True,squared_pred=True,reduction="mean")
    ce_loss=nn.BCEWithLogitsLoss(reduction='mean')

    num_epochs=10
    losses=[]
    best_loss=1e10

    for epoch in range(0,num_epochs):
        epoch_loss=0
        for step,(img,label_list,center_list,box_list) in enumerate(tqdm(dataloader)):
            if img==None:
                continue
            optimizer.zero_grad()
            loss1=0
            img=img.to(device)
            box=box_list.to(device)
            center_point=center_list.to(device)
            label=label_list[:,None,:].long().to(device)
            
            finetune_pred=finetune_model(img,center_point,'single')
            loss=seg_loss(finetune_pred,label)+ce_loss(finetune_pred,label.float())
            # loss=ce_loss(finetune_pred,label.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss+=loss.item()
            loss1+=loss.item()
            
            finetune_pred=finetune_model(img,box,'box')
            loss=seg_loss(finetune_pred,label)+ce_loss(finetune_pred,label.float())
            # loss=ce_loss(finetune_pred,label.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss+=loss.item()
            loss1+=loss.item()
            print(loss1)
            if (step+1)%250==0:
                epoch_loss/=250
                losses.append(epoch_loss)
                
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                    torch.save(finetune_model.state_dict(),join(output_path,'best1.pth'))
                epoch_loss=0
                print('batch loss:{:.4f}'.format(loss1))
                torch.save(finetune_model.state_dict(),join(output_path,'latest1.pth'))
                plt.plot(losses)
                plt.title("Dice + Cross Entropy Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.show()
                plt.savefig(join(output_path,f'finetune_loss_train4.png'))
                plt.close()


class ValidSet(Dataset):
    def __init__(self,data_path,label_path):
        self.data_path=data_path
        self.label_path=label_path

        self.data=sorted(glob(os.path.join(self.data_path,'*.jpg')))
        self.label=sorted(glob(os.path.join(self.label_path,'*.png')))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_path=self.data[index]
        label_path=self.label[index]
        img=Image.open(data_path)
        img=np.array(img)
        label=Image.open(label_path)
        label=np.array(label)
        label_id=np.unique(label)[1:]
        print(label_id)
        id=random.choice(label_id.tolist())
        gt_mask=np.uint8(label==id)
        # print(np.unique(gt_mask))
        y_idx,x_idx=np.where(gt_mask==1)
        y_min,y_max=np.min(y_idx),np.max(y_idx)
        x_min,x_max=np.min(x_idx),np.max(x_idx)
        dt_mask=cv2.distanceTransform(gt_mask[y_min:y_max+1,x_min:x_max+1],cv2.DIST_L2,3)
        local_coords=np.unravel_index(np.argmax(dt_mask,),dt_mask.shape)
        center_point=np.expand_dims(
            np.array([local_coords[1],local_coords[0]])+np.array([x_min,y_min]),axis=0)
            
        img=np.array(img)
        label_list=gt_mask
        center_list=center_point
        bbox_list=np.array([x_min,y_min,x_max,y_max])

        return (img,label_list,center_list,bbox_list,id)


def eval(model,img_path,label_path,origin=False):
    model.eval()
    model=SamPredictor(model)
    # model=model.to(device)
    validset=ValidSet(data_path=img_path,label_path=label_path)
    seg_list=[]
    gt_list=[]
    id_list=[]
    with torch.no_grad():
        for (img,label,center,box,id) in tqdm(validset):
            img=img.astype(np.uint8)
            model.set_image(img)
            seg,_,_=model.predict(point_coords=center,point_labels=np.array([1]),box=box,
                          multimask_output=False)
            seg_list.append(seg[0])
            gt_list.append(label)
            id_list.append(id)
            
    gt_dic={i:0 for i in range(11)}
    pred_dic={i:0 for i in range(11)}
    and_dic={i:0 for i in range(11)}
    for i in range(len(gt_list)):
        pred=seg_list[i]
        gt=gt_list[i]
        and_img=np.logical_and(pred,gt)
        id=id_list[i]
        pred_num=len(np.where(pred>0)[0])
        gt_num=len(np.where(gt>0)[0])
        and_num=len(np.where(and_img>0)[0])
        gt_dic[id]+=gt_num
        pred_dic[id]+=pred_num
        and_dic[id]+=and_num
        dice=0
        total=0
        for i in range(1,11):
            print(gt_dic[i])
            if gt_dic[i]==0:
                continue
            else:
                total+=1
                dice+=2*and_dic[i]/(pred_dic[i]+gt_dic[i])
        mdice=dice/total
        flag=os.path.exists('./res')
        if not flag:
            os.makedirs('./res')
        x_label=['skin','left eyebow','right eyebow','left eye','right eye','nose','upper lip',
                 'inner mouth','lower lip','hair']
        y_label=[2*and_dic[i]/(pred_dic[i]+gt_dic[i]) for i in range(1,11)]
        plt.bar(x=x_label,height=y_label)
        plt.xticks(rotation=45)
        for i in range(len(x_label)):
            plt.text(float(i),y_label[i],'%.2f'%y_label[i],ha='center',fontsize=8)
        if origin:
            plt.title(f'origin-{mdice}')
            plt.savefig(f'./res/origin.png')
        else:
            plt.title(f'train-{mdice}')
            plt.savefig(f'./res/train.png')
        
        return mdice
        
        
if __name__=='__main__':
    torch.cuda.empty_cache()
    train(sam_model,dataloader=train_lapa_loader)
    # eval(model=sam_model,img_path='./LaPa/val/images',label_path='./LaPa/val/labels',origin=False)