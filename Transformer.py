import torch
import torch.nn as nn
import numpy as np
import math

class Config(object):
    def __init__(self):
        self.vocab_size = 6

        self.d_model = 20
        self.n_heads = 2

        assert self.d_model % self.n_heads == 0
        self.dim_k = self.d_model // self.n_heads
        self.dim_v = self.d_model // self.n_heads

        self.padding_size = 30
        self.UNK = 5
        self.PAD = 4

        self.N = 6
        self.p = 0.1
config = Config()

class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        # 假定字典中只有6个词，词向量维度为20
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=config.PAD)

    def forward(self, x):
        # 根据每个句子的长度，进行padding, 短补长截
        for i in range(len(x)):
            if len(x[i]) < config.padding_size:
                # 5 * (30-len(x[i]))
                x[i].extend([config.UNK] * (config.padding_size - len(x[i])))
                #print(x[i])
            else:
                x[i] = x[i][:config.padding_size]
        # batch_size * seq_len * d_model
        x = self.embedding(torch.tensor(x))
        return x


class Position_Encoding(nn.Module):
    def __init__(self, d_model):
        super(Position_Encoding, self).__init__()
        self.d_model = d_model

    def forward(self, seq_len, embeddimg_dim):
        positional_encoding = np.zeros((seq_len, embeddimg_dim))
        for pos in range(positional_encoding.shape[0]):
            for i in range(positional_encoding.shape[1]):
                positional_encoding[pos][i] = math.sin(pos / (10000 ** (2 * i / self.d_model))) if i % 2 == 0 else math.cos(pos / (10000 ** (2 * i / self.d_model)))
        return positional_encoding

class Mutihead_Attention(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)

        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def generate_mask(self, dim):
        matrix = np.ones((dim, dim))
        # mask变为下三角矩阵
        mask = torch.Tensor(np.tril(matrix))
        # mask
        # [[True, False, False, False],
        #  [True, True, False, False],
        #  [True, True, True, False],
        #  [True, True, True, True]]
        return mask == 1

    def forward(self, x, y, requires_mask=False):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
        # 对 x 进行自注意力

        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)  # n_heads * batch_size * seq_len * dim_k
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)  # n_heads * batch_size * seq_len * dim_k
        V = self.v(x).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.n_heads)  # n_heads * batch_size * seq_len * dim_k
        print(Q.shape)
        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        print(attention_score)
        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            attention_score.masked_fill(mask, value=float("-inf"))  # 注意这里的小Trick，不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了

        output = torch.matmul(attention_score, V).reshape(y.shape[0], y.shape[1], -1)
        output = self.o(output)
        return output

class Feed_Forward(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output

class Add_Norm(nn.Module):
    def __init__(self):
        self.dropout = nn.Dropout(config.p)
        super(Add_Norm, self).__init__()

    # sub_layer可以是Feed Forward，也可以是Muti_head_Attention
    def forward(self, x, sub_layer, **kwargs):
        sub_output = sub_layer(x, **kwargs)
        x = self.dropout(x + sub_output)

        layer_norm = nn.LayerNorm(x.size()[1:])
        out = layer_norm(x)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.positional_encoding = Position_Encoding(config.d_model)
        self.muti_atten = Mutihead_Attention(config.d_model, config.dim_k, config.dim_v, config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)

        self.add_norm = Add_Norm()

    def forward(self, x):  # batch_size * seq_len 并且 x 的类型不是tensor，是普通list
        # NS + SD
        x += self.positional_encoding(x.shape[1], config.d_model)
        output = self.add_norm(x, self.muti_atten, y=x)
        output = self.add_norm(output, self.feed_forward)

        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.positional_encoding = Position_Encoding(config.d_model)
        self.muti_atten = Mutihead_Attention(config.d_model, config.dim_k, config.dim_v, config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)
        self.add_norm = Add_Norm()

    def forward(self, x, encoder_output):  # batch_size * seq_len 并且 x 的类型不是tensor，是普通list
        x += self.positional_encoding(x.shape[1], config.d_model)

        output = self.add_norm(x, self.muti_atten, y=x, requires_mask=True)
        output = self.add_norm(output, self.muti_atten, y=output, requires_mask=True)
        output = self.add_norm(output, self.feed_forward)
        return output

class Transformer_Layer(nn.Module):
    def __init__(self):
        super(Transformer_Layer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x_input, x_output = x
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(x_output, encoder_output)
        return (encoder_output, decoder_output)

class Transformer(nn.Module):
    def __init__(self, N, vocab_size, output_dim):
        super(Transformer, self).__init__()
        self.embedding_input = Embedding(vocab_size=vocab_size)
        self.embedding_output = Embedding(vocab_size=vocab_size)

        self.output_dim = output_dim
        self.linear = nn.Linear(config.d_model, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.model = nn.Sequential(
            *[Transformer_Layer() for _ in range(N)]
        )

    def forward(self, x):
        x_input, x_output = x
        x_input = self.embedding_input(x_input)
        x_output = self.embedding_output(x_output)

        _, output = self.model((x_input, x_output))

        output = self.linear(output)
        output = self.softmax(output)
        return output



# embedding = Embedding(6)
# mutihead_attention = Mutihead_Attention(20, 0, 0, 2)
# x = torch.randn(1, 2, 20)
# print(mutihead_attention(x, x).shape)
# word = torch.Tensor([[[1, 2, 3],
#         [2, 3, 4]]])
# print(embedding(word).shape)
# print(embedding(word))
# word = embedding(word)
# print(mutihead_attention(word, word))

# if __name__ == '__main__':
#     Transformer = Transformer(6, )