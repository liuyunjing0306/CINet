import torch.nn as nn
import torch
import math, copy, time
import pdb
import torch.nn.functional as F
from einops import rearrange
from .Image_decoder import Image_Decoder
from .PreNorm_Attention_cross_image import CrossAttentionTransformer
from .Recognition_Net import Transformer


## 修复分支为 Cross_Attention+Image_Decoder

def get_cuda(tensor, gpu):
    # if torch.cuda.is_available():
    #     tensor = tensor
    return tensor.to(torch.device('cuda:{}'.format(gpu)))


class StyleEmbeddings(nn.Module):
    def __init__(self, n_style, d_style):
        super(StyleEmbeddings, self).__init__()
        self.lut = nn.Embedding(n_style, d_style)

    def forward(self, x):
        return self.lut(x)


class InpaintingNet(nn.Module):
    def __init__(self, gpu):
        super(InpaintingNet, self).__init__()
        self.img_decoder = Image_Decoder()
        self.style_embed = StyleEmbeddings(100, 1024)
        self.cross_attention = CrossAttentionTransformer(dim=64, depth=1, heads=8, dim_head=128)
        self.sigmoid = nn.Sigmoid()
        self.gpu = gpu

    def forward(self, image_conv, text_conv):
        # 交叉注意力  其中image_conv [128,1024,8,8] text_conv [128,2048]
        latent = self.cross_attention(image_conv, text_conv)  # latent [128,1024,8,8]

        return latent

    def getLatent(self, x):

        latent_style = rearrange(x, 'b c h w -> b c (h w)')  # 将输入x处理成[128,1024,64]
        latent_style = self.sigmoid(latent_style)
        # memory = self.position_layer(memory)
        # latent_style = torch.sum(latent_style, dim=1)  # (batch_size, d_model) (128,64)
        # 修改
        latent_style = torch.mean(latent_style, dim=-1)  # (128,1024)
        return latent_style

    def getSim(self, latent_style, style=None):
        # latent_norm=torch.norm(latent, dim=-1) #batch, dim
        latent_clone = get_cuda(latent_style.clone(), self.gpu)
        if style is not None:
            style = style.unsqueeze(2)
            style = self.style_embed(style.long())
            pdb.set_trace()
            style = style.reshape(style.size(0), style.size(1), style.size(-1))
        else:
            # style = get_cuda(torch.tensor([[0], [1]]).long(), self.gpu)
            # style = torch.cat(latent_style.size(0) * [style])  # 128, 2, 1
            # style = style.reshape(latent_clone.size(0), -1, 1)
            style = get_cuda(torch.arange(10).view(1, -1, 1).expand(latent_clone.size(0), -1, 1).long(),
                             self.gpu)  # (128,10,1)
            style = self.style_embed(style)  # (128,10,1,1024)
            style = style.reshape(style.size(0), style.size(1), -1)  # (128,10,1024)

        dot = torch.bmm(style, latent_clone.unsqueeze(2))  # batch, style_num, 1
        dot = dot.reshape(dot.size(0), dot.size(1))  # batch,style_num
        return style, dot

    # 处理交叉注意力+风格输入到解码器
    def getOutput(self, out1, out2, out3, out4, latent_style):  # 1->4 从低层到高层,特征尺寸变小
        rec_image4, rec_image3, rec_image2, rec_image1 = self.img_decoder(out1, out2, out3, out4, latent_style)

        return {'rec_image4': rec_image4,
                'rec_image3': rec_image3,
                'rec_image2': rec_image2,
                'rec_image1': rec_image1,
                }



