from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Implement a half-precision Linear Layer.
        Feel free to use the torch.nn.Linear class as a parent class (it makes load_state_dict easier, names match).
        Feel free to set self.requires_grad_ to False, we will not backpropagate through this layer.
        """
        super().__init__(in_features, out_features, bias)
        #self.weights = self.weight.to(torch.float16) not good practise and corect  way
        self.half() #same as self.to(torch.float16)
        #self.bfloat16() #same as self.to(torch.bfloat16)
        #Trainable params and Backward memory are near zero in our implementation 
        #because we do not backpropagate through this layer
        self.requires_grad_(False) #we will not backpropagate through this layer
        # DONE: Implement me
        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint: Use the .to method to cast a tensor to a different dtype (i.e. torch.float16 or x.dtype)
        # The input and output should be of x.dtype = torch.float32
        # DONE: Implement me
        # Only on GPU (fp16 Linear is not for CPU)
        device = x.device
        if not x.is_cuda:
            raise RuntimeError("HalfLinear requires CUDA")
        weight = self.weight.to(device)
        bias = self.bias.to(device) if self.bias is not None else None
        x_fp16 = x.to(torch.float16)
        y = torch.nn.functional.linear(x_fp16, weight, bias)
        #y = super().forward(x.to(torch.float16))
        return y.to(x.dtype)
        #raise NotImplementedError()


class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            #Done: feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )
            #raise NotImplementedError()

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # Done: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )
        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
