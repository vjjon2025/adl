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
        self._d_latent = d_latent
        self._ntokens = n_tokens
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent) #each token has its own embeddings
        #h = 30 and w = 20 - 30x20 = 600 tokens, so we need a positional embedding for at least 600 tokens, we can use 1024 to be safe
        #wehave 1024 positional embeddings, which is enough for 32x32 images, and each embedding is d_latent dimensions
        self.pos_embedding = torch.nn.Embedding(1024, d_latent)  #positional embedding for sequence length of 1024, which is enough for 32*32 images, and d_latent dimensions
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8) #d_model: expected features in input
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=6) #stack multiple layers
        self.to_result_logits = torch.nn.Linear(d_latent, n_tokens)
        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #convolution layers need B, C, H,W, but transformer needs B, S, D where S is sequence length and D is embedding dimension
        #transformer output is: (B, H, W, n_tokens) (channel last)
        #to pass xfiormer output to conv if need be convert to hwc_to_chw(logits)
        
        while x.dim() > 3:
            x = x.squeeze(1)#Remove dimensionn if its size is 1 - causing failures ???
        #fix : ensure input is long for embedding
        x = x.long()

        B, H, W =   x.shape #x is (B, h, w) of integers representing tokens
        x_flat = x.flatten(1) #flatten the h and w dimensions into a sequence, now x_flat is (B, h*w)
        x_emb = self.token_embedding(x_flat) #embed each token into a d_latent dimensional vector, now x_emd is (B, h*w, d_latent)

        seq_len = x_emb.shape[1]
        pos_indices = torch.arange(seq_len, device=x.device).unsqueeze(0) #create a tensor of shape (B, h*w) where each row is [0, 1, 2, ..., h*w-1] representing the position of each token in the sequence
        pos_emb = self.pos_embedding(pos_indices) #get the positional embedding for each position,
        x_emb = x_emb + pos_emb #add the positional embedding to the token embedding, now x_emb is (B, h*w, d_latent) with positional information encoded

        #create a zero token to prepend to x_emb
        start_token = torch.zeros((B, 1, self._d_latent), device=x.device) #create a start token of shape (1, B, d_latent) to prepend to the input sequence
        # drop th elast token and prepend zero token for teacher forcing
        x_emb = torch.cat((start_token, x_emb[:, :-1, :]), dim=1) #prepend the start token to the beginning of the sequence and drop the last token, now x_emb is (B, h*w, d_latent) with the first token being the start token and the rest being the original tokens shifted by one position
        x_emb = x_emb.transpose(0, 1) #expects (S, B, D) for transformer, so transpose to (h*w, B, d_latent)

        #x_emb.shape[0] = H*W, which is the sequence length, so we generate a causal mask of size (H*W, H*W) to ensure that the transformer only attends to previous tokens in the sequence
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x_emb.shape[0]).to(x.device) #generate the causal mask for the transformer
        
        #x_emd: (h*w, B, d_latent) after embedding and transposing, mask: (h*w, h*w) causal mask for the transformer
        xform_result = self.transformer(x_emb, mask=mask) #flatten the input image into a sequence, then embed each token, then pass through transformer
        xform_result_batch_first = xform_result.transpose(0, 1) #(B, H*W, d_latent) transpose back to (B, H*W, d_latent) for the linear layer to project to n_tokens,

        #the one below has format: B, H*W, n_tokens(channels)
        logits = self.to_result_logits(xform_result_batch_first) #project the output of the transformer to the number of tokens to get logits for each token
        logits = logits.view(B, H, W, self._ntokens)  # (B, H, W, n_tokens)
        return logits, {}
        #raise NotImplementedError()

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        #ATTRIBUTION This block was prompted by copilot - on tabbing the function signature and docstring, and then edited by me to fit the requirements of the assignment.
        #generatying the image pixel by pixel, starting from the top left corner and moving to the right and down, using the model's forward function to get the probabilities for the next token at each position, and sampling from those probabilities to get the next token
        #token at a position (i, j) should only depend on the tokens at positions (0, 0) to (i, j-1) and (0, 0) to (i-1, w-1)
        device = device or next(self.parameters()).device
        x = torch.zeros((B, h, w), dtype=torch.long, device=device)
        for i in range(h):
            for j in range(w):
                logits, _ = self.forward(x)
                probs = torch.nn.functional.softmax(logits[:, i, j], dim=-1) #get the probabilities for the next token at position (i, j) from the logits
                x[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1) #sample from the probabilities to get the next token and assign it to x at position (i, j)
        return x
        #raise NotImplementedError()
