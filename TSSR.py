import torch
from torch import nn
import numpy as np
from tsf import PositionalEmbedding,TransformerBlock,MultiHeadAttention,PointWiseFeedForward
import torchvision.models as models
from torch.nn.init import xavier_normal_, uniform_, constant_
import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class NewsEncoder(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.text_encoders = bert_model
        self.reduce_dim_linear = nn.Linear(768,args.d_model)



    def forward(self,inputs_ids,mask):
        """
        Args:
        Returns:
            (shape) batch_size, news_dim
        """
        text_vectors = self.text_encoders(inputs_ids,attention_mask=mask).pooler_output
        final_news_vector = self.reduce_dim_linear(text_vectors)
        return final_news_vector
    
class cross_attn(nn.Module):
    def __init__(self, args):
        super(cross_attn, self).__init__()

        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = args.d_ffn
        dropout = args.dropout
  
        self.c = MultiHeadAttention(attn_heads, d_model, dropout)

        self.a1 = nn.Parameter(torch.tensor(1e-8))
        self.a2 = nn.Parameter(torch.tensor(1e-8))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cross_ffn = PointWiseFeedForward(d_model, d_ffn, dropout)

    def forward(self,Q,KV,mask):
        x = self.norm1(Q + self.a1 * self.dropout1(self.c(Q,KV,KV,mask))) # cross attn + add + norm 
        x = self.norm2(x + self.a2 * self.dropout2(self.cross_ffn(x))) # ffn + add + norm
        return x

class TSSR_img(nn.Module):
    def __init__(self, args):
        super(TSSR_img, self).__init__()
        
        self.num_item = args.num_item + 1
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = args.d_ffn
        layers = args.bert_layers
        dropout = args.dropout
        self.max_len = args.max_len
        enable_res_parameter = args.enable_res_parameter
        self.dense_vis = nn.Linear(2048,d_model)
        self.token = nn.Embedding(self.num_item, d_model)
        self.position = PositionalEmbedding(self.max_len, d_model)


        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool))
        self.attention_mask.requires_grad = False
        
        self.TRMs_id = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])
        self.TRMs_content = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

        self.position = PositionalEmbedding(self.max_len, d_model)
        self.concat = nn.Linear(2*d_model,d_model)
        self.pred = nn.Linear(d_model,self.num_item)

        self.cross_vid =  nn.ModuleList([cross_attn(args) for _ in range(args.cross_layer)])
        self.cross_vcontent = nn.ModuleList([cross_attn(args) for _ in range(args.cross_layer)])

        self.apply(self._init_weights)


        model = models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(args.encoder_path))
        features = list(model.children())[:-1]
        self.content_model = torch.nn.Sequential(*features)



    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)
    
    def forward(self,x,content,test):

        # get mask
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask *= self.attention_mask.to(mask.device)
        mask.requires_grad = False

        # get image embd
        shape0 = content.shape[0]
        shape1 = content.shape[1]
        content = content.view(content.shape[0]*content.shape[1],content.shape[2],content.shape[3],content.shape[4])
        content_embd = self.content_model(content).view(shape0,shape1,-1)
        content_embd_ = self.dense_vis(content_embd)
        content_embd = content_embd_ + self.position(content_embd_)
        for TRM in self.TRMs_content:
            content_embd = TRM(content_embd, mask)

        # get id embd
        id_embd_ = self.token(x)
        id_embd = id_embd_ + self.position(x)
        for TRM in self.TRMs_id:
            id_embd = TRM(id_embd, mask)

        x = id_embd
        y = content_embd
        for cross in self.cross_vid:
            x = cross(x,content_embd,mask)
        for cross in self.cross_vcontent:
            y = cross(y,id_embd,mask)
        
        # gate 
        e = torch.cat([x,y],dim=-1)
        g = nn.Sigmoid()(self.concat(e))
        e = g * x + (1-g) * y

        # predict
        pre = self.pred(e)

        if test:
            return pre

        
        return pre,id_embd,content_embd,id_embd_,content_embd_

class TSSR_text(nn.Module):
    def __init__(self, args):
        super(TSSR_text, self).__init__()
        
        self.num_item = args.num_item + 1
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = args.d_ffn
        layers = args.bert_layers
        dropout = args.dropout
        self.max_len = args.max_len
        enable_res_parameter = args.enable_res_parameter
        self.dense_vis = nn.Linear(2048,d_model)
        self.token = nn.Embedding(self.num_item, d_model)
        self.position = PositionalEmbedding(self.max_len, d_model)


        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool))
        self.attention_mask.requires_grad = False
        
        self.TRMs_id = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])
        self.TRMs_content = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

        self.position = PositionalEmbedding(self.max_len, d_model)
        self.concat = nn.Linear(2*d_model,d_model)
        self.pred = nn.Linear(d_model,self.num_item)

        self.cross_vid =  nn.ModuleList([cross_attn(args) for _ in range(args.cross_layer)])
        self.cross_vcontent = nn.ModuleList([cross_attn(args) for _ in range(args.cross_layer)])

        self.apply(self._init_weights)



        model = AutoModel.from_pretrained(args.encoder_path)
        self.content_model = NewsEncoder(args,model)


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)
    
    def forward(self,x,content,text_mask,test):

        # get mask
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask *= self.attention_mask.to(mask.device)
        mask.requires_grad = False

        # get image embd
        shape0 = content.shape[0]
        shape1 = content.shape[1]
        content = content.view(content.shape[0]*content.shape[1],-1)
        text_mask = text_mask.view(shape0*shape1,-1)
        content_embd_ = self.content_model(content,text_mask).view(shape0,shape1,-1)
        #content_embd_ = self.dense_vis(content_embd)
        content_embd = content_embd_ + self.position(content_embd_)
        for TRM in self.TRMs_content:
            content_embd = TRM(content_embd, mask)

        # get id embd
        id_embd_ = self.token(x)
        id_embd = id_embd_ + self.position(x)
        for TRM in self.TRMs_id:
            id_embd = TRM(id_embd, mask)

        # cross attention
        x = id_embd
        y = content_embd
        for cross in self.cross_vid:
            x = cross(x,content_embd,mask)
        for cross in self.cross_vcontent:
            y = cross(y,id_embd,mask)
        
        # gate 
        e = torch.cat([x,y],dim=-1)
        g = nn.Sigmoid()(self.concat(e))
        e = g * x + (1-g) * y

        # predict
        pre = self.pred(e)

        if test:
            return pre

        
        return pre,id_embd,content_embd,id_embd_,content_embd_
