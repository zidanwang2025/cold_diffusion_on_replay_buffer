import math
import torch
import numpy as np
from einops import rearrange, reduce, repeat

class TemporalTransformer(torch.nn.Module):
    def __init__(self, transition_dim, latent_dim=128, 
                 ff_size=512, num_layers=8, num_heads=4, dropout=0.0,
                 activation="gelu", **kargs):
        super().__init__()
        """
        TODO
        """
        # configuration
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        # embedding pose to vector
        self.query_emb_model = Query_emb_model(transition_dim, latent_dim)
        
        # embedding noise step
        self.steps_encoder = StepEncoding(latent_dim)
        
        # Transformer
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = torch.nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        
        # decoder
        self.output_layer = torch.nn.Sequential(
                            LinearBlock(latent_dim, int(latent_dim / 2), activation=activation),
                            LinearBlock(int(latent_dim / 2), int(latent_dim / 2), activation=activation),
                            LinearBlock(int(latent_dim / 2), transition_dim, activation="none"))
    
    def forward(self, x, cond, time):
        """
            x : [ batch x horizon x transition]
            out: [batch x horizon x transition]
        """
        # encode each transition to features
        query_feature = self.query_emb_model(x) # B H T -> B H F
        bs, nframes, nfeats = query_feature.shape
        query_feature = rearrange(query_feature, "B H F -> H B F")
        
        # encode noise time step into vector by MLP.
        time_emb = self.steps_encoder(time)
        query_feature = torch.cat([query_feature, time_emb], 0) # (H+1) B F
        
        # Add positional encoding
        query_feature = self.sequence_pos_encoder(query_feature) # (H+1) B F
         
        # transformer process
        transformer_feature = self.seqTransEncoder(query_feature) # (H + 1) B F
        transformer_feature = rearrange(transformer_feature[:nframes], "H B F -> B H F") # (H + 1) B F -> B H F
        
        # predict noise
        out = self.output_layer(transformer_feature) # B H T
        
        return out


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class StepEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.linear_layer = self.make_linear_model(dim, dim, "gelu")

    def forward(self, time):
        """
        -----------------------------
        inputs
        x: torch.tensor(S, B, D)
        time: torch.tensor(B)
        -----------------------------
        S: length of the sequence
        B: Batch size
        D: Dimension of feature
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_len) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = self.linear_layer(embeddings)
        embeddings = repeat(embeddings, "B D -> S B D", S=1)
        return embeddings
    
    def make_linear_model(self, input_dim, output_dim, act):
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            self.activation_layer(act),
            torch.nn.Linear(output_dim, output_dim * 2),
            self.activation_layer(act),
            torch.nn.Linear(output_dim * 2, output_dim))
        return model
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = torch.nn.ReLU()
        elif name == 'prelu':
            layer = torch.nn.PReLU()
        elif name == 'lrelu':
            layer = torch.nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = torch.nn.Tanh()
        elif name == 'sigmoid':
            layer = torch.nn.Sigmoid()
        elif name == 'gelu':
            layer = torch.nn.GELU()
        elif name == 'none':
            layer = torch.nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class Query_emb_model(torch.nn.Module):
    def __init__(self, input_dim, emb_dim=128, act="gelu"):
        """
        Input:
        input_dim: int, same as dim of transition.
        emd_dim: int, hyperparameter.
        """
        super().__init__()
        self.query_emb_model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, emb_dim),
            self.activation_layer(act),
            torch.nn.Linear(emb_dim, emb_dim * 2),
            self.activation_layer(act),
            torch.nn.Linear(emb_dim * 2, emb_dim))
        
    def forward(self, x):
        """
        Input
        x: torch.tensor, shape (batch horizon transition)
        Output:
        features: torch.tensor, shape (batch horizon emb_dim)
        """
        features = self.query_emb_model(x)
        return features
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = torch.nn.ReLU()
        elif name == 'prelu':
            layer = torch.nn.PReLU()
        elif name == 'lrelu':
            layer = torch.nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = torch.nn.Tanh()
        elif name == 'sigmoid':
            layer = torch.nn.Sigmoid()
        elif name == 'gelu':
            layer = torch.nn.GELU()
        elif name == 'none':
            layer = torch.nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class LinearBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, activation='prelu', norm=None):
        super(LinearBlock, self).__init__()

        self.linear = torch.nn.Linear(input_size, output_size)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)
        elif self.norm == 'none':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(False)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, False)
        elif self.activation == 'gelu':
            self.act = torch.nn.GELU()
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'none':
            self.activation = None
    
    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.linear(x))
        else:
            out = self.linear(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out