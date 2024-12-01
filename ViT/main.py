import torch
import torch.nn as nn
import einops

class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.heads * self.head_dim == self.embed_size), "Embedding Size must be divisible by no. of heads."

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fcout = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries): # No masking in ViT
        N = values.shape[0]
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        energy = torch.einsum('nqhd, nkhd->nhqk', queries, keys)
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum('nhql, nlhd->nqhd', attention, values)
        out = out.reshape(N, queries_len, self.embed_size)
        out = self.fcout(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, p):
        super(EncoderBlock, self).__init__()
        self.attention = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
                nn.Linear(embed_size, forward_expansion * embed_size),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(forward_expansion * embed_size, embed_size),
                nn.Dropout(p),
            )
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        attention = self.attention(x, x, x)
        x = self.dropout(self.norm1(attention + x))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class ViT(nn.Module):
    def __init__(self, num_layers, patch_size, height, width, channels, embed_size, heads, forward_expansion, out_dim, p):
        super(ViT, self).__init__()
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.height = height//patch_size
        self.width = width//patch_size
        assert(self.height * patch_size == height), 'Height must be divisible by Patch Size.'
        assert(self.width * patch_size == width), 'Width must be divisible by Patch Size.'

        self.embedding = nn.Linear(patch_size * patch_size * channels, embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.height * self.width + 1, embed_size))
        self.layers = nn.ModuleList(
                [
                    EncoderBlock(embed_size, heads, forward_expansion, p)
                    for _ in range(num_layers)
                ]
            )
        self.fcout = nn.Linear(embed_size, out_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        N = x.shape[0]
        x = einops.rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = self.embedding(x)
        x = torch.cat([self.cls_token.expand(N, 1, self.embed_size), x], dim=1)
        out = self.dropout(x + self.pos_embedding.expand(N, self.height * self.width + 1, self.embed_size))

        for layer in self.layers:
            out = layer(out)

        out = out[:, 0, :]
        out = self.fcout(out)

        return out