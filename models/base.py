from torch import nn, softmax, tensor, long
import torch.nn.functional as F
import torch
from transformers import BertModel, BertTokenizer

class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_key_padding_mask=None):
        x2, _ = self.self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + x2)

        x2, _ = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(x + x2)

        x2 = self.ffn(x)
        x = self.norm3(x + x2)
        return x
    
class BaseModel(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 nhead=8, 
                 num_enc_layers=4,
                 num_cmd_dec_layers=4,
                 num_param_dec_layers=4,
                 num_cmd_classes=6,
                 max_param_len=20,
                 max_len=100):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert_proj = nn.Linear(768, d_model)
        
        # Freeze BERT for efficiency
        for p in self.bert.parameters():
            p.requires_grad = False
        
        self.encoder = nn.Sequential(*[EncoderBlock(d_model, nhead) for _ in range(num_enc_layers)])
        self.cmd_decoder = nn.Sequential(*[DecoderBlock(d_model, nhead) for _ in range(num_cmd_dec_layers)])
        self.param_decoder = nn.Sequential(*[DecoderBlock(d_model, nhead) for _ in range(num_param_dec_layers)])
        
        self.tgt_embedding = nn.Embedding(num_cmd_classes, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        self.cmd_head = nn.Linear(d_model, num_cmd_classes)
        self.param_head = nn.Linear(d_model, max_param_len)
        
    def forward(self, input_ids, tgt_seq, attention_mask):
        x = self.bert(input_ids).last_hidden_state
        x = self.bert_proj(x)
        
        src_key_padding_mask = (attention_mask == 0)  # [B, S]
        
        enc_out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        print(x.dtype, enc_out.dtype)
        dec = self.tgt_embedding(tgt_seq)   # [B, T] → [B, T, d_model]
        dec = dec + self.pos_embedding[:, :dec.size(1)]
        
        
    
        for block in self.cmd_decoder:
            dec = block(dec, enc_out)

        cmd_logits = self.cmd_head(dec)
        params = self.param_head(dec)

        return cmd_logits, params
    
class BaseModelLoss(nn.Module):
    def __init__(self, cmd_loss_weight=1.0, param_loss_weight=1.0):
        super().__init__()
        self.cmd_loss_weight = cmd_loss_weight
        self.param_loss_weight = param_loss_weight
        self.cmd_loss_fn = nn.CrossEntropyLoss()

    def forward(self, 
                cmd_logits, 
                param_preds, 
                cmd_targets, 
                param_targets):
        
        # -----------------------
        # Command loss
        # -----------------------
        cmd_loss = self.cmd_loss_fn(
            cmd_logits.reshape(-1, cmd_logits.size(-1)),
            cmd_targets.reshape(-1)
        )

        # -----------------------
        # Soft weights from logits
        # -----------------------
        probs = F.softmax(cmd_logits, dim=-1)

        confidence, _ = probs.max(dim=-1, keepdim=True)  # (B, T, 1)
        confidence = confidence.detach()

        # -----------------------
        # Parameter loss
        # -----------------------
        diff = param_preds - param_targets
        mse = diff ** 2

        # ensure broadcasting is correct
        if confidence.shape != mse.shape:
            confidence = confidence.expand_as(mse)

        weighted_mse = mse * confidence

        norm = confidence.sum().clamp(min=1.0)
        param_loss = weighted_mse.sum() / norm

        # -----------------------
        # Total loss
        # -----------------------
        total_loss = (
            self.cmd_loss_weight * cmd_loss +
            self.param_loss_weight * param_loss
        )

        return total_loss, cmd_loss, param_loss

SPECIAL_TOKENS = {
    "<curve_end>",
    "<loop_end>",
    "<face_end>",
    "<sketch_end>",
    "<extrude_end>"
}

CMD_PARAM_COUNT = {
    "line": 2,
    "arc": 4,   
    "circle": 8,
    "add": 17,
    "cut": 17,
    "intersect": 17
}

import numpy as np

SPECIAL_TOKENS = {
    "<curve_end>",
    "<loop_end>",
    "<face_end>",
    "<sketch_end>",
    "<extrude_end>"
}

CMD_PARAM_COUNT = {
    "line": 2,
    "arc": 4,
    "circle": 8,
    "add": 17,
    "cut": 17,
    "intersect": 17
}

MAX_PARAM = max(CMD_PARAM_COUNT.values())


import numpy as np

SPECIAL_TOKENS = {
    "<curve_end>",
    "<loop_end>",
    "<face_end>",
    "<sketch_end>",
    "<extrude_end>"
}

# mapper: how many params each command uses
PARAM_MAP = {
    "line": 2,
    "arc": 4,        # adjust if needed
    "circle": 8,
    "add": 17,
    "cut": 17,
    "intersect": 17
}

param_start_post = {
    key : sum(list(PARAM_MAP.values())[:i]) for i, key in enumerate(PARAM_MAP.keys())
}
param_max_len = sum(PARAM_MAP.values())

CMD_VOCAB = {
    "PAD": 0,
    "SOS": 1,
    "EOS": 2,
    "line": 3,
    "arc": 4,
    "circle": 5,
    "add": 6,
    "cut": 7,
    "intersect": 8
}

def extract_commands_and_params_from_string(seq: str):
    commands = []
    raw_params = []

    seq = seq.replace("\n", " ").strip()
    tokens = seq.split()

    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue

        parts = token.split(",")
        if len(parts) < 2:
            continue

        cmd = parts[0].lower()

        if cmd not in PARAM_MAP:
            continue

        try:
            nums = [float(p) for p in parts[1:] if p != ""]
        except ValueError:
            continue

        expected = PARAM_MAP[cmd]
        nums = nums[:expected]  # truncate if too long

        # pad if too short
        if len(nums) < expected:
            nums = nums + [0.0] * (expected - len(nums))

        commands.append(cmd)
        raw_params.append(nums)


    L = len(commands)
    param_matrix = np.zeros((L, param_max_len), dtype=np.float32)

    # fill matrix
    for i, c in enumerate(commands):
        dim = PARAM_MAP[c]
        offset = param_start_post[c]
        param_matrix[i, offset:offset+dim] = raw_params[i]

    return commands, param_matrix

CMD_TOKEN_MAP = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "line": 3,
    "arc": 4,
    "circle": 5,
    "add": 6,
    "cut": 7,
    "intersect": 8
}

def tokenize_commands_and_params(seq: str, max_len=None):
    commands, param_matrix = extract_commands_and_params_from_string(seq)

    L, D = param_matrix.shape

    # ---- tokenize commands ----
    cmd_tokens = [CMD_TOKEN_MAP["<SOS>"]]
    cmd_tokens += [CMD_TOKEN_MAP[c] for c in commands]
    cmd_tokens += [CMD_TOKEN_MAP["<EOS>"]]

    # ---- expand params ----
    # add zero rows for SOS and EOS
    param_tokens = np.zeros((L + 2, D), dtype=np.float32)
    param_tokens[1:L+1] = param_matrix

    # ---- padding (optional) ----
    if max_len is not None:
        pad_len = max_len - len(cmd_tokens)

        if pad_len > 0:
            cmd_tokens += [CMD_TOKEN_MAP["<PAD>"]] * pad_len

            pad_params = np.zeros((pad_len, D), dtype=np.float32)
            param_tokens = np.vstack([param_tokens, pad_params])

        else:
            # truncate
            cmd_tokens = cmd_tokens[:max_len]
            param_tokens = param_tokens[:max_len]

    cmd_tokens = np.array(cmd_tokens, dtype=np.int64)

    return cmd_tokens, param_tokens

def reconstruct_string(commands, param_matrix):
    tokens = []

    L = len(commands)

    for i in range(L):
        cmd = commands[i]

        # compute offset layout (same as encoder!)
        offset = 0
        for j, c in enumerate(commands):
            dim = PARAM_MAP[c]

            if i == j:
                params = param_matrix[i, offset:offset+dim]
                break

            offset += dim

        # clean params (remove padding zeros at the end if needed)
        params = params.tolist()

        # optional: trim trailing zeros
        while len(params) > 0 and abs(params[-1]) < 1e-6:
            params.pop()

        # format
        param_str = ",".join(str(int(p)) if p.is_integer() else str(p) for p in params)
        tokens.append(f"{cmd},{param_str}")

        # add curve_end for sketch commands
        if cmd in {"line", "arc", "circle"}:
            tokens.append("<curve_end>")

    # simple heuristic for structure
    # everything before first extrude command = sketch
    sketch_tokens = []
    extrude_tokens = []

    for t in tokens:
        if any(t.startswith(c) for c in ["add", "cut", "intersect"]):
            extrude_tokens.append(t)
        else:
            sketch_tokens.append(t)

    result = []

    if sketch_tokens:
        result.extend(sketch_tokens)
        result.append("<loop_end>")
        result.append("<face_end>")
        result.append("<sketch_end>")

    if extrude_tokens:
        result.extend(extrude_tokens)
        result.append("<extrude_end>")

    return " ".join(result)

def get_tokenizer_and_model(max_len=100):
    model = BaseModel(
        d_model=256,
        nhead=8,
        num_enc_layers=4,
        num_cmd_dec_layers=4,
        num_param_dec_layers=4,
        num_cmd_classes=len(CMD_PARAM_COUNT),
        max_param_len=sum(CMD_PARAM_COUNT.values()),  
        max_len=max_len
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    loss = BaseModelLoss()
    return tokenizer, model, loss
    