import jittor as jt
from jittor import nn
from jittor import init
import models.configs as configs
import copy
import numpy as np
from scipy import ndimage
from os.path import join as pjoin

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2jt(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return jt.array(weights)

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = (img_size, img_size)
        patch_size = config.patches["size"]
        # print("config.split", config.split)
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            stride = patch_size
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            stride = (config.slide_step, config.slide_step)
        else:
            raise ValueError('Invalid split method: {}'.format(config.split))
        self.patch_embeddings = nn.Conv(in_channels=in_channels, out_channels=config.hidden_size, kernel_size=patch_size, stride=stride)
        self.position_embeddings = jt.zeros((1, n_patches + 1, config.hidden_size))
        self.cls_token = jt.zeros((1, 1, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
    
    def execute(self, x):
        batch_size = x.shape[0]
        cls_tokens = jt.expand(self.cls_token, [batch_size, -1, -1])
        x = self.patch_embeddings(x)
        x = jt.flatten(x, 2, -1)
        x = jt.permute(x, (0, 2, 1))
        x = jt.concat((cls_tokens, x), dim=1)
        
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    def execute(self, x, target):
        logprobs = nn.log_softmax(x, dim=-1)
        nll_loss = -jt.misc.gather(logprobs, dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        
        self._init_weights()
    
    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.trunc_normal_(self.fc1.bias, a=-4, b=4)
        init.trunc_normal_(self.fc2.bias, a=-4, b=4)
    def execute(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        
        self.softmax = nn.Softmax(dim=-1)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(new_x_shape)
        return jt.permute(x, (0, 2, 1, 3))

    def execute(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = jt.matmul(query_layer, key_layer.permute(0, 1, 3, 2))
        attention_scores = attention_scores / jt.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)
        
        context_layer = jt.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.attn = Attention(config)
    
    def execute(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with jt.no_grad():
            query_weight = np2jt(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).transpose(0, 1)
            key_weight = np2jt(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).transpose(0, 1)
            value_weight = np2jt(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).transpose(0, 1)
            out_weight = np2jt(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).transpose(0, 1)

            query_bias = np2jt(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2jt(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2jt(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2jt(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight = query_weight
            self.attn.key.weight = key_weight
            self.attn.value.weight = value_weight
            self.attn.out.weight = out_weight
            self.attn.query.bias = query_bias
            self.attn.key.bias = key_bias
            self.attn.value.bias = value_bias
            self.attn.out.bias = out_bias

            mlp_weight_0 = np2jt(weights[pjoin(ROOT, FC_0, "kernel")]).transpose(0, 1)
            mlp_weight_1 = np2jt(weights[pjoin(ROOT, FC_1, "kernel")]).transpose(0, 1)
            mlp_bias_0 = np2jt(weights[pjoin(ROOT, FC_0, "bias")])
            mlp_bias_1 = np2jt(weights[pjoin(ROOT, FC_1, "bias")])

            self.ffn.fc1.weight = mlp_weight_0
            self.ffn.fc2.weight = mlp_weight_1
            self.ffn.fc1.bias = mlp_bias_0
            self.ffn.fc2.bias = mlp_bias_1

            self.attention_norm.weight = np2jt(weights[pjoin(ROOT, ATTENTION_NORM, "scale")])
            self.attention_norm.bias = np2jt(weights[pjoin(ROOT, ATTENTION_NORM, "bias")])
            self.ffn_norm.weight = np2jt(weights[pjoin(ROOT, MLP_NORM, "scale")])
            self.ffn_norm.bias = np2jt(weights[pjoin(ROOT, MLP_NORM, "bias")])
            
            
class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def execute(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = jt.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]
        max_inx, _ = jt.argmax(last_map, dim=2)
        return max_inx


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential()
        for _ in range(config.transformer["num_layers"] - 1):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.part_select = Part_Attention()
        self.part_layer = Block(config)
        self.part_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
    
    def execute(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)
        part_inx = self.part_select(attn_weights)
        part_inx = part_inx + 1
        parts = []
        batch_size, num = part_inx.shape
        for i in range(batch_size):
            parts.append(hidden_states[i, part_inx[i,:]])
        parts = jt.stack(parts)
        concat = jt.concat((hidden_states[:,0].unsqueeze(1), parts), dim=1)
        part_states, _ = self.part_layer(concat)
        part_encoded = self.part_norm(part_states)   
        
        return part_encoded
        
        
class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)
    def execute(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        part_encoded = self.encoder(embedding_output)
        return part_encoded



class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=448, num_classes=200, smoothing_value=0, zero_head=False, constrastive=1, margin=0.4):
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        
        self.transformer = Transformer(config, img_size)
        self.part_head = nn.Linear(config.hidden_size, num_classes)
        self.constrastive = constrastive
        self.margin = margin

    def con_loss(self, features, labels):
        B, _ = features.shape
        features = jt.normalize(features)
        cos_matrix = jt.matmul(features, features.transpose(0, 1))
        pos_label_matrix = jt.stack([jt.float32(labels == labels[i]) for i in range(B)])
        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = 1 - cos_matrix
        # print("margin", self.margin)
        neg_cos_matrix = cos_matrix - self.margin
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
        loss /= (B * B)
        return loss

    def execute(self, x, labels=None):
        part_tokens = self.transformer(x)
        part_logits = self.part_head(part_tokens[:, 0])
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = nn.CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            part_loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
            # print("constrastive", self.constrastive)
            
            if self.constrastive == 0:
                return part_loss, part_logits
            # print("contrast should be 1")
            contrast_loss = self.con_loss(part_tokens[:, 0], labels.view(-1))
            loss = part_loss + contrast_loss
            
            return loss, part_logits
        else:
            return part_logits

    def load_from(self, weights):
        with jt.no_grad():
            self.transformer.embeddings.patch_embeddings.weight = np2jt(weights["embedding/kernel"], conv=True)
            self.transformer.embeddings.patch_embeddings.bias = np2jt(weights["embedding/bias"])
            self.transformer.embeddings.cls_token = np2jt(weights["cls"])
            self.transformer.encoder.part_norm.weight = np2jt(weights["Transformer/encoder_norm/scale"])
            self.transformer.encoder.part_norm.bias = np2jt(weights["Transformer/encoder_norm/bias"])

            posemb = np2jt(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings = posemb
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
                    
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                
                self.transformer.embeddings.position_embeddings = (np2jt(posemb))
            for (i, block) in enumerate(self.transformer.encoder.layer):
                block.load_from(weights, i)
    
CONFIGS = {
    'debug': configs.get_debug_config(),
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}