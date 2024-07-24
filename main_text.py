
import torch
from torch import nn
import torch.utils.data as Data
from tqdm import tqdm
from TSSR import TSSR_text as TSSR
import os,pickle
from utils import eval_text as eval

device_ids = [0,1,2,3]

def main(args):
    model = TSSR(args)
    gpus = len(device_ids)
    model = torch.nn.DataParallel(model,device_ids=device_ids)
    model = model.cuda(device=device_ids[0])
    train_dataset = TextDataset(args,mode='train')
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size*gpus, shuffle=True,num_workers=32,pin_memory=True)

    valid_dataset = TextDataset(args, mode='valid')
    valid_loader = Data.DataLoader(valid_dataset, batch_size=args.test_batch_size*gpus,num_workers=32,pin_memory=True)

    print('dataset initial ends')
    

    CE_pre = nn.CrossEntropyLoss(ignore_index=0)
    CE_Co = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = best = 0
    ld1 = ld2 = 1
    for epoch in range(args.num_epoch):
        model.train()
        total1 = total2 = total3 = total4 = total5 = 0
        train_loader = tqdm(train_loader)

        for idx,(text,mask,label,seq) in enumerate(train_loader):
            step += 1
            text,mask,label,seq = text.cuda(device=device_ids[0]),mask.cuda(device=device_ids[0]),label.cuda(device=device_ids[0]),seq.cuda(device=device_ids[0])

            bc = seq.shape[0]
            out,user_id_embd,user_text_embd,item_id_embd,item_text_embd = model(seq,text,mask,test=False)
            user_text_pooling = torch.mean(user_text_embd,dim = 1)
            user_id_pooling = torch.mean(user_id_embd,dim = 1)
            score_user_text2id = torch.cosine_similarity(user_text_pooling.unsqueeze(1),user_id_pooling.unsqueeze(0),dim=-1)
            score_user_id2text = torch.cosine_similarity(user_id_pooling.unsqueeze(1),user_text_pooling.unsqueeze(0),dim=-1)
 
            score_item_text2id = torch.matmul(user_text_embd[:,:-1,:].transpose(0,1),item_id_embd[:,1:,:].transpose(0,1).transpose(-1,-2))
            score_item_id2text = torch.matmul(user_id_embd[:,:-1,:].transpose(0,1),item_text_embd[:,1:,:].transpose(0,1).transpose(-1,-2)) # L,B,B


            out = out.view(out.shape[0]*out.shape[1],-1)
            label = label.view(-1)
            loss_pre = CE_pre(out,label)

            # mask the padding item and sub user
            filt = (seq>0)[:,1:].transpose(0,1) # L * B
            mask = filt.unsqueeze(1) * filt.unsqueeze(2) # L * B * B
            score_item_text2id[mask==False] = -1e9
            score_item_id2text[mask==False] = -1e9
            filt = filt.reshape(-1)
            score_item_text2id = score_item_text2id.view(-1,score_item_text2id.shape[-1])
            score_item_id2text = score_item_id2text.view(-1,score_item_id2text.shape[-1])

            Co_user_label = torch.range(0,bc-1).long().to(score_user_text2id.device)
            loss_user_text2id = CE_Co(score_user_text2id/args.le_t,Co_user_label)
            loss_user_id2text = CE_Co(score_user_id2text/args.le_t,Co_user_label)
            Co_item_label = Co_user_label.unsqueeze(0).repeat(args.max_len-1,1).view(-1)
            
            loss_item_text2id = CE_Co(score_item_text2id[filt],Co_item_label[filt])
            loss_item_id2text = CE_Co(score_item_id2text[filt],Co_item_label[filt])
            loss = loss_pre + (loss_user_id2text + loss_user_text2id)*ld1 + (loss_item_id2text + loss_item_text2id)*ld2

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                total1 += loss_pre.cpu().item()
                total2 += loss_user_text2id.cpu().item() 
                total3 += loss_user_id2text.cpu().item()
                total4 += loss_item_text2id.cpu().item() 
                total5 += loss_item_id2text.cpu().item()
                
                if idx % args.print_step == 0:
                    f = open(args.save_path + 'loss.txt','a+')
                    print(total1 / (idx+1),total2/(idx+1),total3/(idx+1),total4/(idx+1),total5/(idx+1),file=f)
                    f.close()

            if step % args.eval_per_steps == 0 :
                model.eval()
                with torch.no_grad():
                    f = open(args.save_path + 'result.txt','a+')
                    m = eval(valid_loader,model,[100,50,20,10],args.num_item)
                    print(epoch,m,file = f)
                    if m['NDCG@10'] > best:
                        g = open(args.save_path +'model.pkl','wb')
                        pickle.dump(model,g)
                        best = m['NDCG@10']
                        g.close()
                    f.close()
                model.train()

if __name__ == '__main__':

    from args import args
    from dataset import TextDataset
    main(args)
