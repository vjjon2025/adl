import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    #Tensor shape B, H, W, C for input and output, B, H, W, codebook_bits for the latent code
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim
        self.linear_down = torch.nn.Linear(embedding_dim, codebook_bits)
        self.linear_up = torch.nn.Linear(codebook_bits, embedding_dim)
        #raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        B, H, W, C = x.shape
        x_flat = x.view(B*H*W, C)
        linear_down = self.linear_down(x_flat)   
        linear_down = linear_down.view(B, H, W, -1) #reshape back to B, H, W, codebook_bits
        x = torch.nn.functional.normalize(linear_down, p=2, dim=-1) #along channel is dim=1, dim=0 is batch, dim=2 is h, dim=3 is w 
        x = diff_sign(x)
        return x
        #    raise NotImplementedError()


    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        x = self.linear_up(x)
        return x
        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim) #Calls PatchAutoEncoder init
        #PatchAutoEncoder returns B, H, W, C for the latent code, we want to apply BSQ to this code, (co-pilot comment) ATTRIBUTINON
        # #which means we need to reshape it to B, H, W, embedding_dim and then apply BSQ to get B, H, W, codebook_bits
        BSQ.__init__(self, codebook_bits=codebook_bits, embedding_dim=latent_dim)        
        #raise NotImplementedError()

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        return BSQ._code_to_index(self,self.encode(x))
        #raise NotImplementedError()

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(BSQ._index_to_code(self,x))
        #raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        bottle = self.encoder(x) #encoder from PatchAutoEncoder
        bsquantized = BSQ.encode(self,bottle)#BSQ encode
        return bsquantized
        #raise NotImplementedError()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(BSQ.decode(self,x))
        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        quantized = self.encode(x)
        result = self.decode(quantized) #decoder from PatchAutoEncoder, BSQ decode
        return result, {}
        #raise NotImplementedError()
