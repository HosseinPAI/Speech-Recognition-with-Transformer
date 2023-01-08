import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
def correct_input_sizes(input_sizes):
    input_sizes = ((input_sizes - 9)/2).to(torch.int)

    return input_sizes


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class input_embedding(nn.Module):
    def __init__(self):
        super(input_embedding, self).__init__()

        self.in_embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
            nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1),
            nn.Hardtanh(min_val=0, max_val=20),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1),
            nn.Hardtanh(min_val=0, max_val=20)
        )

    def forward(self, x):
        return self.in_embedding(x)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class positional_encoding(nn.Module):
    def __init__(self, dim_model=256, max_len=5000):
        super(positional_encoding, self).__init__()

        self.PE = torch.zeros(max_len, dim_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        sc_argument = pos / torch.pow(10000, (torch.arange(0, dim_model, 2).float() / float(dim_model)))
        self.PE[:, 0::2] = torch.sin(sc_argument)  # Even Positions
        self.PE[:, 1::2] = torch.cos(sc_argument)  # Odd Positions
        self.PE = self.PE.unsqueeze(0)

    def forward(self, x):
        x = self.PE[:, :x.size(1)]
        return x


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class attention(nn.Module):
    def __init__(self, dim_key=64):
        super(attention, self).__init__()

        self.dim_key = dim_key
        self.drpout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        output = torch.matmul(q, k.transpose(1, 2))
        output = output / math.sqrt(self.dim_key)
        output = self.softmax(output)
        output = self.drpout(output)
        output = torch.matmul(output, v)

        return output


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class multi_head_attention(nn.Module):
    def __init__(self, dim_model=256, num_heads=8, dim_key=64, dim_value=64):
        super(multi_head_attention, self).__init__()
        self.num_heads = num_heads  # 8
        self.dim_key = dim_key  # 64
        self.dim_value = dim_value  # 64
        # in_feature=256, out_feature=512
        self.query_linear = nn.Linear(in_features=dim_model, out_features=num_heads * dim_key)
        # in_feature=256, out_feature=512
        self.key_linear = nn.Linear(in_features=dim_model, out_features=num_heads * dim_key)
        # in_feature=256, out_feature=512
        self.value_linear = nn.Linear(in_features=dim_model, out_features=num_heads * dim_value)
        self.ScaledDotProductAttention = attention(dim_key=dim_key)
        self.LNorm = nn.LayerNorm(normalized_shape=(dim_model,), eps=1e-5)
        self.Linear = nn.Linear(in_features=num_heads * dim_value, out_features=dim_model)
        self.Drpout = nn.Dropout(p=0.1)

    def forward(self, q, k, v):
        input_data = q
        batch_size = q.size(0)
        q = self.query_linear(q)  # Batch_size*T*512
        k = self.key_linear(k)  # Batch_size*T*512
        v = self.value_linear(v)  # Batch_size*T*512
        # Reshape Data for Scale Dot Products
        q = q.reshape(batch_size, q.size(1), self.num_heads, self.dim_key).permute(0, 2, 1, 3).reshape(
            batch_size * self.num_heads, q.size(1), self.dim_key)  # (Batch_size*8)*T*64
        k = k.reshape(batch_size, k.size(1), self.num_heads, self.dim_key).permute(0, 2, 1, 3).reshape(
            batch_size * self.num_heads, k.size(1), self.dim_key)  # (Batch_size*8)*T*64
        v = v.reshape(batch_size, v.size(1), self.num_heads, self.dim_value).permute(0, 2, 1, 3).reshape(
            batch_size * self.num_heads, v.size(1), self.dim_value)  # (Batch_size*8)*T*64
        # SacaledDotProductAttention Network
        output = self.ScaledDotProductAttention(q, k, v)
        # Reshape Output for Linear Model
        output = output.reshape(batch_size, self.num_heads, -1, self.dim_key).permute(0, 2, 1, 3)
        output = output.reshape(batch_size, output.size(1), -1)  # Batch_size*T*512
        # Output Linear Model
        output = self.Linear(output)
        output = self.Drpout(output)
        # Add and Norm
        output = self.LNorm(output + input_data)

        return output


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class POS_FNN(nn.Module):
    def __init__(self, dim_model=256, dim_inner=1024):
        super(POS_FNN, self).__init__()

        self.Conv1 = nn.Conv1d(in_channels=dim_model, out_channels=dim_inner, kernel_size=(1,), stride=(1,))
        self.Conv2 = nn.Conv1d(in_channels=dim_inner, out_channels=dim_model, kernel_size=(1,), stride=(1,))
        self.Drout = nn.Dropout(p=0.1)
        self.LNorm = nn.LayerNorm((dim_model,), eps=1e-5)

    def forward(self, x):
        input_data = x
        # Feed Forward
        x = x.transpose(1, 2)
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Conv2(x)
        x = x.transpose(1, 2)
        x = self.Drout(x)
        # Add and Norm
        x = self.LNorm(x + input_data)

        return x


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class encoder_layer(nn.Module):
    def __init__(self, dim_model=256, num_heads=8, dim_key=64, dim_value=64, dim_inner=1024):
        super(encoder_layer, self).__init__()

        self.MultiHeadAttention = multi_head_attention(dim_model=dim_model, num_heads=num_heads,
                                                       dim_key=dim_key, dim_value=dim_value)
        self.PosFNN = POS_FNN(dim_model=dim_model, dim_inner=dim_inner)

    def forward(self, x):
        x = self.MultiHeadAttention(x, x, x)
        x = self.PosFNN(x)

        return x


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class encoder(nn.Module):
    def __init__(self, dim_input = 672, dim_model=256, num_heads=8, dim_key=64, dim_value=64, dim_inner=1024):
        super(encoder, self).__init__()

        self.Drout = nn.Dropout(p=0.1, inplace=False)
        self.Linear= nn.Linear(in_features=dim_input, out_features=dim_model)
        self.LNorm = nn.LayerNorm((dim_model, ), eps=1e-5)
        self.PositionalEncoding = positional_encoding(dim_model=dim_model)
        self.EncoderLayer1 = encoder_layer(dim_model=dim_model, num_heads=num_heads, dim_key=dim_key,
                                           dim_value=dim_value, dim_inner=dim_inner)
        self.EncoderLayer2 = encoder_layer(dim_model=dim_model, num_heads=num_heads, dim_key=dim_key,
                                           dim_value=dim_value, dim_inner=dim_inner)
        self.EncoderLayer3 = encoder_layer(dim_model=dim_model, num_heads=num_heads, dim_key=dim_key,
                                           dim_value=dim_value, dim_inner=dim_inner)
        self.EncoderLayer4 = encoder_layer(dim_model=dim_model, num_heads=num_heads, dim_key=dim_key,
                                           dim_value=dim_value, dim_inner=dim_inner)

    def forward(self, x, input_sizes, device):
        output1 = self.Linear(x)
        output1 = self.LNorm(output1)
        output2 = self.PositionalEncoding(x).to(device)
        output = self.Drout(output1 + output2)

        output = self.EncoderLayer1(output)
        output = self.EncoderLayer2(output)
        output = self.EncoderLayer3(output)
        output = self.EncoderLayer4(output)

        return output


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, dim_input=672, dim_model=256, num_heads=8, dim_key=64, dim_value=64, dim_inner=1024,
                 output_dim=32):
        super(Transformer, self).__init__()

        self.InputEmbedding = input_embedding()
        self.Encoder = encoder(dim_input=dim_input, dim_model=dim_model, num_heads=num_heads,
                               dim_key=dim_key, dim_value=dim_value, dim_inner=dim_inner)
        self.Linear = nn.Linear(in_features=dim_model, out_features=output_dim)

    def forward(self, inputs, input_sizes, device):
        # input Embedding
        output = self.InputEmbedding(inputs)  # output shape: Batch_size*32*21*T
        # Reshape
        output = output.reshape(output.size(0), output.size(1) * output.size(2),
                                output.size(3)).permute(0, 2, 1).contiguous()
        masked_input_sizes = correct_input_sizes(input_sizes)
        # Encoder
        output = self.Encoder(output, input_sizes, device)
        # Final Fully Connected
        output = self.Linear(output)  # B*T*C
        output = F.log_softmax(output, dim=2)

        return output, masked_input_sizes

