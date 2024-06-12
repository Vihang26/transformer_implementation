import torch
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(self,embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert(self.head_dim * heads == embed_size), "Emmbed size needs to be divsible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #Spliting embeedings into self.head pieces 
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, value_len, self.heads, self.head_dim)
        query = query.reshape(N, value_len, self.heads, self.head_dim)


        #Einsum does matrix multiplication for query*keys for each training example 
        energy = torch.einsum("nqhd,nkhd->nhqk", [query,keys])
        #queries shape: (N, query_len, heads, heads_dim)
        #keys shape: (N, key_len, heads, heads_dim)
        #energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))



        # According to the equation applying the softmax layer 
        attention = torch.softmax(energy/ (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


