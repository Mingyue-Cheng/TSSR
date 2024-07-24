import torch
from torch import nn
from tqdm import tqdm
def recalls_and_ndcgs_for_ks(scores, labels, ks): 
        metrics = {}
        answer_count = labels.sum(1)
        labels_float = labels.float()
        rank = (-scores).argsort(dim=1)
        cut = rank
        for k in sorted(ks, reverse=True):
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)

            metrics['Recall@%d' % k] = \
                (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device),
                                         labels.sum(1).float())).mean().cpu().item()

            position = torch.arange(2, 2 + k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()
        return metrics

def eval(data_loader,model,metric_ks,num_item):

    model.eval()
    with torch.no_grad():
        metrics = {}
        for idx,batch in enumerate(tqdm(data_loader)):
            img,answer,seqs = batch
            img,answer,seqs = img.cuda(device=0),answer.cuda(device=0),seqs.cuda(device=0)
            x = model(seqs,img,test = True)
            answers = answer.tolist()
            seqs = seqs.tolist()
            x = x[:,-1,:]
            x = x.view(-1,x.shape[-1])
            answer = answer.view(-1)
            labels = answer.tolist()

            # filter out the history
            for i in range(len(answers)):
                row = []
                col = []
                seq = list(set(seqs[i] + answers[i]))
                seq.remove(answers[i][0])
                row += [i] * len(seq)
                col += seq
                x[row, col] = -1e9

            labels = torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.int64), num_classes=num_item+1)
            metrics_batch = recalls_and_ndcgs_for_ks(x,labels.to(x.device), metric_ks)
            for k, v in metrics_batch.items():
                if not metrics.__contains__(k):
                    metrics[k] = v
                else:
                    metrics[k] += v

        for k, v in metrics.items():
            metrics[k] = v/(idx+1)
        del(labels)
    model.train()
    return metrics 

def eval_text(data_loader,model,metric_ks,num_item):

    model.eval()
    with torch.no_grad():
        metrics = {}
        for idx,batch in enumerate(tqdm(data_loader)):
            img,mask,answer,seqs = batch
            img,mask,answer,seqs = img.cuda(device=0),mask.cuda(device=0),answer.cuda(device=0),seqs.cuda(device=0)
            x = model(seqs,img,mask,test = True)
            answers = answer.tolist()
            seqs = seqs.tolist()
            x = x[:,-1,:]
            x = x.view(-1,x.shape[-1])
            answer = answer.view(-1)
            labels = answer.tolist()

            # filter out the history
            for i in range(len(answers)):
                row = []
                col = []
                seq = list(set(seqs[i] + answers[i]))
                seq.remove(answers[i][0])
                row += [i] * len(seq)
                col += seq
                x[row, col] = -1e9

            labels = torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.int64), num_classes=num_item+1)
            metrics_batch = recalls_and_ndcgs_for_ks(x,labels.to(x.device), metric_ks)
            for k, v in metrics_batch.items():
                if not metrics.__contains__(k):
                    metrics[k] = v
                else:
                    metrics[k] += v

        for k, v in metrics.items():
            metrics[k] = v/(idx+1)

    model.train()
    return metrics 