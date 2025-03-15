from model.arcface_model import Backbone
from model.temporal_convolutional_model import TemporalConvNet
import math
import os
import torch
from torch import nn
from model.backbone import VisualBackbone, AudioBackbone
from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module
from model.transformer import MultimodalTransformerEncoder, IntraModalTransformerEncoder, InterModalTransformerEncoder

class MultimodalFusion(nn.Module):
    def __init__(self, visual_dim=256, audio_dim=128, hidden_dim=256, num_heads=4):
        super().__init__()
        
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.audio_fusion = nn.Sequential(
            nn.Linear(audio_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hiudden_dim,
            num_heads=nm_heads,
            batch_first=True
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.Sigmoid()
        )
        
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 2)
        )


    def forward(self, visual_feat, audio_feat1, audio_feat2):
        audio_concat1 = torch.cat([audio_feat1, audio_feat1], dim=-1)
        audio_fused1 = self.audio_fusion(audio_concat1)  
        audio_concat2 = torch.cat([audio_feat2, audio_feat2], dim=-1)
        audio_fused2 = self.audio_fusion(audio_concat2)  
        
        attn_out1, _ = self.cross_attn(
            query=visual_feat,
            key=audio_fused1,
            value=audio_fused1
        ) 
        attn_out2, _ = self.cross_attn(
            query=visual_feat,
            key=audio_fused2,
            value=audio_fused2
        )
        
        gate = self.gate_net(torch.cat([visual_feat, attn_out1, attn_out2], dim=-1))
        fused = gate * visual_feat + (1 - gate) * attn_out  
        
        return self.regressor(fused) 


class my_model(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None, attention=0,
                 output_dim=1, kernel_size=5, dropout=0.1, root_dir='', input_dim_other=[128, 39]):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.attention = attention
        self.dropout = dropout
        self.other_feature_dim = input_dim_other  # input dimension of other modalities

        def load_visual_backbone(self):
            resnet = VisualBackbone(mode='ir', use_pretrained=False)
            state_dict = torch.load(os.path.join(self.root_dir, "res50_ir_0.887.pth"), 
                                map_location='cpu')
            resnet.load_state_dict(state_dict)
            for param in resnet.parameters():
                param.requires_grad = False
            return resnet

        def load_audio_backbone(self):
            vggish = AudioBackbone()
            state_dict = torch.load(os.path.join(self.root_dir, "vggish.pth"),
                                map_location='cpu')
            vggish.backbone.load_state_dict(state_dict)
            for param in vggish.parameters():
                param.requires_grad = False
            return vggish

    def init(self, fold=None):
        path = os.path.join(self.root_dir, "res50_ir_0.887" + ".pth")
        path1 = os.path.join(self.root_dir, "vggish" + ".pth")

        spatial = self.load_visual_backbone()
        state_dict = torch.load(path, map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False
        
        spatial1 = self.load_audio_backbone()
        state_dict = torch.load(path1, map_location='cpu')
        spatial1.load_state_dict(state_dict)

        for param in spatial1.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=3, attention=self.attention,
            dropout=self.dropout)
        self.temporalf = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=5, attention=self.attention,
            dropout=self.dropout)
        self.temporalf1 = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=7, attention=self.attention,
            dropout=self.dropout)

        self.temporal1 = TemporalConvNet(
            num_inputs=self.other_feature_dim[0], num_channels=[64, 64, 64, 64], kernel_size=3,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal1v = TemporalConvNet(
            num_inputs=self.other_feature_dim[0], num_channels=[64, 64, 64, 64], kernel_size=5,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal1v1 = TemporalConvNet(
            num_inputs=self.other_feature_dim[0], num_channels=[64, 64, 64, 64], kernel_size=7,
            attention=self.attention,
            dropout=self.dropout)

        self.temporal2 = TemporalConvNet(
            num_inputs=self.other_feature_dim[1], num_channels=[64, 64, 64, 64], kernel_size=3,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal2m = TemporalConvNet(
            num_inputs=self.other_feature_dim[1], num_channels=[64, 64, 64, 64], kernel_size=5,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal2m1 = TemporalConvNet(
            num_inputs=self.other_feature_dim[1], num_channels=[64, 64, 64, 64], kernel_size=7,
            attention=self.attention,
            dropout=self.dropout)

        self.encoder1 = nn.Linear(self.embedding_dim // 4 * 3, 64)
        self.encoder2 = nn.Linear(64*3, 64)
        self.encoder3 = nn.Linear(64*3, 64)

        self.fusion = MultimodalFusion(
            visual_dim=256,
            audio_dim=128,
            hidden_dim=256,
            num_heads=4
        )

        self.ln = nn.LayerNorm([3, 64])

        

    def forward(self, x, x1, x2):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()

        batch_size, channel, length, freq = x2.shape
        x2 = x2.permute(0, 2, 3, 1).contiguous()  
        x2 = x2.view(-1, freq, channel)  
        x2 = self.spatial1(x2)
        x2 = x2.view(batch_size, length, -1).unsqueeze(1)
        
        xa = self.temporalf(x).transpose(1, 2).contiguous()
        xb = self.temporalf1(x).transpose(1, 2).contiguous()
        x = self.temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)

        xa = xa.contiguous().view(num_batches * length, -1)
        xb = xb.contiguous().view(num_batches * length, -1)

        x1 = x1.squeeze().transpose(1, 2).contiguous().float()
        x2 = x2.squeeze().transpose(1, 2).contiguous().float()
        
        x1a = self.temporal1v(x1).transpose(1, 2).contiguous()
        x1b = self.temporal1v1(x1).transpose(1, 2).contiguous()
        x1 = self.temporal1(x1).transpose(1, 2).contiguous()
        
        x2a = self.temporal2m(x2).transpose(1, 2).contiguous()
        x2b = self.temporal2m1(x2).transpose(1, 2).contiguous()
        x2 = self.temporal2(x2).transpose(1, 2).contiguous()

        x = torch.cat([x, xa, xb], dim=-1) 
        x1 = torch.cat([x1, x1a, x1b], dim=-1) 
        x2 = torch.cat([x2, x2a, x2b], dim=-1)

        x0 = self.encoder1(x)
        x1 = self.encoder2(x1.contiguous().view(num_batches * length, -1))
        x2 = self.encoder3(x2.contiguous().view(num_batches * length, -1))

        x = self.fusion(x0, x1, x2)  

        return torch.tanh(output)
