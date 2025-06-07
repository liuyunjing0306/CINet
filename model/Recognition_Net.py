import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np
from torch.autograd import Variable
from util import get_alphabet
from .Backbone import *
from .Text_decoder import Text_Decoder
from .FIM_Block import FIM
from .PreNorm_Attention_image import ImageAttention
from .PreNorm_Attention_text import TextAttention

torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)
# alphabet = get_alphabet("character")
alphabet = get_alphabet()


## 识别辅助分支为 backbone+FIM +2Self_Attention+TextDecoder
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        return embed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.proj(x)


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

        self.word_n_class = len(alphabet)
        self.embedding_word = Embeddings(512, self.word_n_class)
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)
        self.encoder = ResNet(num_in=3, block=BasicBlock, layers=[3, 4, 6, 3]).cuda()  # Backbone
        self.decoder = Text_Decoder()  # Text_decoder
        self.generator_word = Generator(1024, 2048)
        ## 添加
        self.fim = FIM().cuda()
        self.image_attention = ImageAttention(dim=1024, depth=1, heads=8, dim_head=128).cuda()
        self.text_attention = TextAttention(dim=1024, depth=1, heads=8, dim_head=128).cuda()
        ##

    def forward(self, image, text_length, text_input, conv_feature=None, test=False):

        if conv_feature is None:
            out1, out2, out3, out4, conv_feature = self.encoder(image)

        if text_length is None:
            return {
                'conv': conv_feature,
            }

        text_embedding = self.embedding_word(text_input)
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda()
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2)
        batch, seq_len, _ = text_input_with_pe.shape
        ## 添加
        inpaint_x1, recognition_x1, common_x = self.fim(conv_feature, conv_feature)
        inpaint_x2 = self.image_attention(inpaint_x1, common_x)
        recognition_x2 = self.text_attention(recognition_x1, common_x)
        ##
        text_input_with_pe, attention_map = self.decoder(text_input_with_pe,
                                                         recognition_x2)  ###  self.decoder(text_input_with_pe, conv_feature)
        word_decoder_result = self.generator_word(text_input_with_pe)

        if test:
            return {
                'pred': word_decoder_result,
                'map': attention_map,
                'conv': conv_feature,
                'inpaint_map': inpaint_x2,
                'encoder_out1': out1,
                'encoder_out2': out2,
                'encoder_out3': out3,
                'encoder_out4': out4,

            }

        else:
            total_length = torch.sum(text_length).data
            probs_res = torch.zeros(total_length, 2048).type_as(word_decoder_result.data)

            start = 0
            for index, length in enumerate(text_length):
                length = length.data
                probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
                start = start + length

            return {
                'pred': probs_res,
                'map': attention_map,
                'conv': conv_feature,
                'inpaint_map': inpaint_x2,
                'encoder_out1': out1,
                'encoder_out2': out2,
                'encoder_out3': out3,
                'encoder_out4': out4,
            }

