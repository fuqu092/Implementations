import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.head_dim = embed_size//heads
        self.heads = heads

        assert(self.head_dim * self.heads == self.embed_size), "Embedding size must be divisible by no of heads."

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        
        self.fcout = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask):
        # Shape = (N, len, embedding_size)

        N = queries.shape[0]
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy/self.embed_size ** (1/2), dim=3)

        out = torch.einsum("nhql, nlhd->nqhd", [attention, values])
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
                nn.ReLU(),
                nn.Linear(forward_expansion * embed_size, embed_size), 
                )
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(x + attention))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(x + forward))

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, p):
        super(DecoderBlock, self).__init__()

        self.attention1 = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention2 = Attention(embed_size, heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
                nn.Linear(embed_size, forward_expansion * embed_size),
                nn.ReLU(),
                nn.Linear(forward_expansion * embed_size, embed_size),
                )
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(p)

    def forward(self, values, keys, x, src_mask, trg_mask):
        attention1 = self.attention1(x, x, x, trg_mask)
        x = self.dropout(self.norm1(x + attention1))

        attention2 = self.attention2(values, keys, x, src_mask)
        x = self.dropout(self.norm2(x + attention2))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm3(forward + x))

        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, max_length, embed_size, num_layers, heads, forward_expansion, p):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(src_vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
                [
                    EncoderBlock(embed_size, heads, forward_expansion, p)
                    for _ in range(num_layers)
                    ]
                )
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask):
        N, length = x.shape
        pos = torch.arange(0, length).expand(N, length)
        out = self.dropout(self.embedding(x) + self.pos_embedding(pos))

        for layer in self.layers:
            out = layer(out, mask)

        return out


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, max_length, embed_size, num_layers, heads, forward_expansion, p):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
                [
                    DecoderBlock(embed_size, heads, forward_expansion, p)
                    for _ in range(num_layers)
                    ]
                )
        self.fcout = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(p)

    def forward(self, enc_out, x, src_mask, trg_mask):
        N, length = x.shape
        pos = torch.arange(0, length).expand(N, length)
        out = self.dropout(self.embedding(x) + self.pos_embedding(pos))

        for layer in self.layers:
            out = layer(enc_out, enc_out, out, src_mask, trg_mask)

        out = self.fcout(out)

        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, max_length, embed_size, num_layers, heads, forward_expansion, p, src_pad_idx, trg_pad_idx):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, max_length, embed_size, num_layers, heads, forward_expansion, p)
        self.decoder = Decoder(trg_vocab_size, max_length, embed_size, num_layers, heads, forward_expansion, p)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(enc_out, trg, src_mask, trg_mask)

        return dec_out