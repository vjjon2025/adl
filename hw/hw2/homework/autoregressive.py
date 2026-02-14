import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """
    """
    Design your AutoregressiveModel in autoregressive.py. Many models work 
    here, but a decoder-only transformer might be the easiest. As with the 
    quantizer above, you will not require a large network to pass 
    this assignment. We recommend using torch.nn.TransformerEncoderLayer 
    (not a typo) with a causal mask 
    torch.nn.Transformer.generate_square_subsequent_mask. 
    For this to work, you should flatten your input image into a 
    sequence first. You'll need to take care handling the 
    auto-regressive prediction: The output at location (i, j) 
    should not see the input token at location(i, j) 
    which should predict and only see tokens preceding it. 
    You may use a positional embedding, but this is optional.

    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8)
        self.output_layer = torch.nn.Linear(d_latent, n_tokens)
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=6)

        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        x = x.view(B, h * w)  # Flatten the input to (B, h*w)
        x = self.token_embedding(x)  # Embed the tokens to (B, h*w, d_latent)
        x = x.permute(1, 0, 2)  # Permute to (h*w, B, d_latent) for transformer
        x = torch.nn.functional.pad(x, (0, 0, 1, 0), value=0)  # Shift the input by 1 position (pad at the beginning)
        x = self.transformer(x)  # Pass through the transformer
        x = x.permute(1, 0, 2)  # Permute back to (B, h*w, d_latent)
        x = self.output_layer(x)  # Output layer to get (B, h*w, n_tokens)
        x = x.view(B, h, w, -1)  # Reshape back to (B, h, w, n_tokens)
        return x, {}
        #raise NotImplementedError()

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        x = torch.zeros((B, h, w), dtype=torch.long, device=device)  # Start with a tensor of zeros
        for i in range(h):
            for j in range(w):
                output, _ = self.forward(x)  # Get the output probabilities
                probs = torch.nn.functional.softmax(output[:, i, j], dim=-1)  # Get probabilities for the current position
                x[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Sample from the distribution
        return x
        #raise NotImplementedError()
