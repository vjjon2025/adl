from pathlib import Path
from xml.parsers.expat import model

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False) #we will not backpropagate through this layer
        #Base quantized weights must be frozen 

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        
        self.qlora_a = torch.nn.Linear(in_features, lora_dim, bias=False) #Keep the LoRA layers in float32
        for grad_param in self.qlora_a.parameters():
            grad_param.requires_grad = True #trainable LoRA layers
        #grad_param.numel() for grad_param in model.parameters() if grad_param.requires_grad - TOTAL TRAINABLE PARAMS
        self.qlora_b = torch.nn.Linear(lora_dim, out_features, bias=False) #Keep the LoRA layers in float32 not HalfLinear
        for grad_param in self.qlora_b.parameters():
            grad_param.requires_grad = True #trainable LoRA layers
        self.alpha_div_rank = 1.0  / float(lora_dim) # Scaling factor for LoRA
        torch.nn.init.kaiming_uniform_(self.qlora_a.weight)
        torch.nn.init.zeros_(self.qlora_b.weight) #Typically LoRA is not used on conv layers becuase they are already efficient

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        #raise NotImplementedError()

        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_half_linear = super().forward(x)
        qlora_out = self.qlora_b.forward(self.qlora_a.forward(x.to(torch.float32)))
        qlora_out_adjusted = qlora_out * self.alpha_div_rank
        return base_half_linear + qlora_out_adjusted.to(x.dtype)        
        # DONE: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
        #raise NotImplementedError()


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim),
            )            
            # DONE: Implement me (feel free to copy and reuse code from bignet.py)
            #raise NotImplementedError()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        # DONE: Implement me (feel free to copy and reuse code from bignet.py)
        #raise NotImplementedError()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
