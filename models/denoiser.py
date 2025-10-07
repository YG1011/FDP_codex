import torch
import torch.nn as nn
from typing import List
from .autoencoder import AutoEncoder
import torch.nn.functional as F
from purifier.FreqGuidedDiffuser3D import FreqGuidedDiffuser3D   # 你的 FDP 实现
from models.denoiser3d import DDPM3D   

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
        pass
    
    def forward(self, data, *args, **kwargs):
        return data

class Layer_Denoiser(nn.Module):
    def __init__(self, model:AutoEncoder, t:int):
        super(Layer_Denoiser, self).__init__()
        self.model = model
        self.t = t
    
    def forward(self, data:torch.tensor, layer:int) -> torch.tensor:
        if layer == self.model.args.layer_no:
            with torch.no_grad():
                data = data.permute((0,2,1))
                data, shift, scale = self.normalize_layer_data(data)
                code = self.model.encode(data)
                # Changed Encode -> Denoiser to call truncated_sample()
                recons = self.model.denoiser(data, t=self.t, flexibility=self.model.args.flexibility, context=code).detach()
                recons = recons * scale + shift
            return recons.permute((0,2,1))
        else:
            return data
    
    @staticmethod
    def normalize_layer_data(data:torch.tensor, normalization="unit_shape"):
        # if normalization == "unit_shape":
            # data.shape = [B, N, D]
        shift = data.mean(dim=1, keepdim=True) # (B, 1, D)
        scale = data.std(dim=[1,2], keepdim=True) # (B, 1, 1)
        data = (data - shift) / scale # Normalize layer data
        return data, shift, scale


class Multiple_Layer_Denoiser(nn.Module):
    def __init__(self, models:List[AutoEncoder], t_list:List[int]):
        super(Multiple_Layer_Denoiser, self).__init__()
        self.models = models
        self.layers = [i for i in range(len(t_list))]
        self.t_list = t_list
    
    def forward(self, data:torch.tensor, layer:int) -> torch.tensor:
        if layer in self.layers:
            idx = self.layers.index(layer)
            model = self.models[idx]
            if isinstance(model, Identity): return data
            if self.t_list[idx] != 0:
                with torch.no_grad():
                    data = data.permute((0,2,1))
                    data, shift, scale = self.normalize_layer_data(data)
                    code = model.encode(data)
                    # Changed Encode -> Denoiser to call truncated_sample()
                    recons = model.denoiser(data, t=self.t_list[idx], flexibility=model.args.flexibility, context=code).detach()
                    recons = recons * scale + shift
                return recons.permute((0,2,1))
            else:
                return data
        else:
            return data
    
    @staticmethod
    def normalize_layer_data(data:torch.tensor, normalization="unit_shape"):
        # if normalization == "unit_shape":
            # data.shape = [B, N, D]
        shift = data.mean(dim=1, keepdim=True) # (B, 1, D)
        scale = data.std(dim=[1,2], keepdim=True) # (B, 1, 1)
        data = (data - shift) / scale # Normalize layer data
        return data, shift, scale
    
    class Input_FDP_Denoiser(nn.Module):
        """
        让 FDP 以“denoiser”的形式挂到 PCLD 的 forward_denoised 流程里。
        只在 layer==0（输入点）时生效，其它层原样返回。
        """
        def __init__(self, fdp: FreqGuidedDiffuser3D):
            super().__init__()
            self.fdp = fdp

        @staticmethod
        def _to_BNC(x_BCN):
            # x: (B,C,N) -> (B,N,C)
            return x_BCN.permute(0, 2, 1).contiguous()

        @staticmethod
        def _to_BCN(x_BNC):
            # x: (B,N,C) -> (B,C,N)
            return x_BNC.permute(0, 2, 1).contiguous()

        @torch.no_grad()
        def forward(self, data: torch.Tensor, layer: int) -> torch.Tensor:
            """
            data: (B,C,N) 由 classifier.forward_denoised 传入
            layer: 当前层序号（0 表示输入点）
            """
            if layer != 0:
                return data

            # 1) (B,C,N)->(B,N,C) ; C=3 for input layer
            x = self._to_BNC(data)
            assert x.size(-1) == 3, f"Input layer should be 3D points, got {x.shape}"

            # 2) 调用 FDP：点->体素->频域逆扩散(ASE+PSP)->点
            x_pur = self.fdp(x)   # (B,N,3)

            # 3) 回到 (B,C,N)
            return self._to_BCN(x_pur)