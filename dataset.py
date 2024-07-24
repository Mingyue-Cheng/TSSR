
import torch
import json,pickle
import pandas as pd
import numpy as np
import torch.utils.data as Data
from torchvision import transforms

from PIL import Image 

class IMGDataset(Data.Dataset):
    def __init__(self, args,mode):
        
        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values   # user id seq
        self.mode = mode # train valid test
        self.num_user = self.data.shape[0]
        self.num_item = self.data.max()+1
        self.max_len = args.max_len
        self.img_path = args.content_path 

        if args.load_proprecessing:
            self.ilist = pickle.load(open(self.img_path + '/Image_all.pkl','rb'))
            return
        #get all images
        
        # image transform
        self.images_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # you should prepare a list of image names ordered by id
        f = open(self.img_path + '/id2img.json') 
        self.id2name = json.load(f) 
        self.id2name = [self.img_path + '/Images/'+i+'.jpg' for i in self.id2name]
        f.close()
        # get image with mul threading
        with torch.no_grad():
            self.ilist = []
            self.imgs = {}
            x = 11000 # nums to process for each single thread 
            import threading
            if mode == 'train' :
                T = []
                for n in range(0,self.num_item,x):
                    stop = n + x if n + x < self.num_item else self.num_item
                    th = threading.Thread(target = self.get_all,args = (n,stop))
                    T.append(th)
                    th.start()
                for t in T:
                    t.join()
                for n in range(0, self.num_item, x):
                    self.ilist = self.ilist + self.imgs[n]
                # save the image after transform if your storage is enough
                if args.save_proprecessing:
                    pickle.dump(self.ilist,open(self.img_path+'/Image_all.pkl','wb'))



    def get_all(self,n,stop):
        from tqdm import tqdm
        imgs = []
        
        for i in tqdm(range(n,stop)):
            try:
                if i == 0:
                    img = torch.zeros(3,224,224)
                else:
                    path = self.id2name[i]
                    img = Image.open(path).convert("RGB")
                    img = self.images_transforms(img)
                imgs.append(img.unsqueeze(0).numpy())
            except:
                imgs.append('missing')
        self.imgs[n] = imgs

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):

        if self.mode == 'train':

            seq = self.data[index, -self.max_len - 3:-3].tolist()
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq
            img_seq = []
            for i in seq:
                img_seq.append(torch.tensor(self.ilist[i])) 
            img_seq = torch.cat(img_seq,dim=0)
            idlist = self.data[index, -self.max_len - 2:-2].tolist()
            padding_len = self.max_len - len(idlist)
            idlist = [0] * padding_len + idlist

            return img_seq,torch.LongTensor(idlist),torch.LongTensor(seq)

        elif self.mode == 'test':
            seq = self.data[index, -self.max_len - 1:-1].tolist()
            labels = [self.data[index,-1].tolist()]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq
            img_seq = []
            for i in seq:
                img_seq.append(torch.tensor(self.ilist[i]))
            img_seq = torch.cat(img_seq,dim=0)
            
            return img_seq,torch.LongTensor(labels),torch.LongTensor(seq)

        elif self.mode == 'valid':
            seq = self.data[index, -self.max_len - 2:-2].tolist()
            labels = [self.data[index,-2].tolist()]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq
            img_seq = []
            for i in seq:
                img_seq.append(torch.tensor(self.ilist[i]))
            img_seq = torch.cat(img_seq,dim=0)

            return img_seq,torch.LongTensor(labels),torch.LongTensor(seq)




class TextDataset(Data.Dataset):
    def __init__(self, args,mode):

        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.mode = mode
        self.num_user = self.data.shape[0]
        self.num_item = self.data.max()+1
        self.max_len = args.max_len

        # load news preprocessed data by BERT tokenizer
        self.text = pickle.load(open(args.content_path+'/text.pkl','rb')) # item_id -> text_word seq
        self.mask = pickle.load(open(args.content_path+'/mask.pkl','rb')) # item_id -> padding_mask seq


    def __len__(self):
        return self.num_user

    def __getitem__(self, index):

        if self.mode == 'train':

            seq = self.data[index, -self.max_len - 3:-3].tolist()

            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq
            img_seq = []
            mask = []
            for i in seq:

                img_seq.append(torch.LongTensor(self.text[i]).unsqueeze(0)) 
                m = torch.LongTensor(self.mask[i]).unsqueeze(0)
                mask.append(m) 

            img_seq = torch.cat(img_seq,dim=0)
            mask = torch.cat(mask,dim=0)

            
            idlist = self.data[index, -self.max_len - 2:-2].tolist()
            padding_len = self.max_len - len(idlist)
            idlist = [0] * padding_len + idlist

            return img_seq,mask,torch.LongTensor(idlist),torch.LongTensor(seq)

        elif self.mode == 'test':
            seq = self.data[index, -self.max_len - 1:-1].tolist()
            labels = [self.data[index,-1].tolist()]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            img_seq = []
            mask = []
            for i in seq:

                img_seq.append(torch.LongTensor(self.text[i]).unsqueeze(0)) 
                mask.append(torch.LongTensor(self.mask[i]).unsqueeze(0)) 

            img_seq = torch.cat(img_seq,dim=0)
            mask = torch.cat(mask,dim=0)

            
            return img_seq,mask,torch.LongTensor(labels),torch.LongTensor(seq)
        
        elif self.mode == 'valid':
            seq = self.data[index, -self.max_len - 2:-2].tolist()
            labels = [self.data[index,-2].tolist()]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            img_seq = []
            mask = []
            for i in seq:

                img_seq.append(torch.LongTensor(self.text[i]).unsqueeze(0)) 
                mask.append(torch.LongTensor(self.mask[i]).unsqueeze(0)) 

            img_seq = torch.cat(img_seq,dim=0)
            mask = torch.cat(mask,dim=0)

            return img_seq,mask,torch.LongTensor(labels),torch.LongTensor(seq)