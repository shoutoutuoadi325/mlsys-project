import math
import os

import torch
import torch.nn.functional as F


def create_engine(model_config: dict, weight_dir: str, device: str = "cuda"):
    return Engine(model_config, weight_dir, device)


def load_state_dict(weight_path):
    try:
        return torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location="cpu")


class Engine:
    def __init__(self, config, weight_dir, device):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.config = config
        self.device = torch.device(device)
        self.dtype = self._select_dtype(config)

        weight_path = os.path.join(weight_dir, "model.pt")
        state_dict = load_state_dict(weight_path)
        self.w = {
            name: tensor.to(device=self.device, dtype=self.dtype)
            for name, tensor in state_dict.items()
        }

        self.num_layers = int(config["num_hidden_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.hidden_size = int(config["hidden_size"])
        self.eps = float(config.get("rms_norm_eps", 1e-5))
        self.rope_theta = float(config.get("rope_theta", 10000.0))

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.requests = {}

    def _select_dtype(self, config):
        if self.device.type != "cuda":
            return torch.float32

        dtype = str(config.get("torch_dtype", "float16")).lower()
        if dtype in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float16

    def prefill(self, request_ids, input_ids):
        input_list = self._normalize_input_ids(input_ids)
        outputs = []

        for rid, ids in zip(request_ids, input_list):
            rid = int(rid)
            ids = ids.to(device=self.device, dtype=torch.long)
            self.requests[rid] = ids.clone()

            logits = self._forward_full(ids.unsqueeze(0))
            outputs.append(logits[0, -1, :])

        return torch.stack(outputs, dim=0)

    def decode(self, request_ids, token_ids):
        token_ids = self._normalize_token_ids(token_ids)
        outputs = []

        for rid, token in zip(request_ids, token_ids):
            rid = int(rid)
            if rid not in self.requests:
                raise KeyError(f"unknown request_id {rid}; call prefill first")

            token = token.reshape(1).to(device=self.device, dtype=torch.long)
            ids = torch.cat([self.requests[rid], token], dim=0)
            self.requests[rid] = ids

            logits = self._forward_full(ids.unsqueeze(0))
            outputs.append(logits[0, -1, :])

        return torch.stack(outputs, dim=0)

    def remove(self, request_ids):
        for rid in request_ids:
            self.requests.pop(int(rid), None)

    def _normalize_input_ids(self, input_ids):
        if torch.is_tensor(input_ids):
            if input_ids.dim() == 1:
                return [input_ids]
            return [row for row in input_ids]
        return list(input_ids)

    def _normalize_token_ids(self, token_ids):
        if torch.is_tensor(token_ids):
            return [x for x in token_ids.reshape(-1)]
        return [torch.tensor(x, device=self.device, dtype=torch.long) for x in token_ids]

    def _rmsnorm(self, x, weight):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return x_norm.to(x.dtype) * weight

    def _apply_rope(self, q, k, seqlen):
        dim = q.shape[-1]
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, device=self.device, dtype=torch.float32) / dim)
        )
        positions = torch.arange(seqlen, device=self.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(q.dtype)[None, None, :, :]
        sin = freqs.sin().to(q.dtype)[None, None, :, :]

        def rotate(x):
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            x_rotated = torch.stack(
                (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
                dim=-1,
            )
            return x_rotated.flatten(-2)

        return rotate(q), rotate(k)

    def _forward_full(self, input_ids):
        x = self.w["embed_tokens.weight"][input_ids]
        batch, seqlen, _ = x.shape

        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen),
                float("-inf"),
                device=self.device,
                dtype=torch.float32,
            ),
            diagonal=1,
        )[None, None, :, :]

        for layer_idx in range(self.num_layers):
            prefix = f"layers.{layer_idx}"

            residual = x
            x_norm = self._rmsnorm(x, self.w[f"{prefix}.input_layernorm.weight"])

            q = F.linear(x_norm, self.w[f"{prefix}.self_attn.q_proj.weight"])
            k = F.linear(x_norm, self.w[f"{prefix}.self_attn.k_proj.weight"])
            v = F.linear(x_norm, self.w[f"{prefix}.self_attn.v_proj.weight"])

            q = q.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q, k = self._apply_rope(q, k, seqlen)

            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)

            attn = torch.matmul(q.float(), k.float().transpose(-1, -2))
            attn = attn / math.sqrt(self.head_dim)
            attn = attn + causal_mask
            attn = torch.softmax(attn, dim=-1).to(x.dtype)

            y = torch.matmul(attn, v)
            y = y.transpose(1, 2).contiguous().view(batch, seqlen, self.hidden_size)
            y = F.linear(y, self.w[f"{prefix}.self_attn.o_proj.weight"])
            x = residual + y

            residual = x
            x_norm = self._rmsnorm(x, self.w[f"{prefix}.post_attention_layernorm.weight"])
            gate = F.linear(x_norm, self.w[f"{prefix}.mlp.gate_proj.weight"])
            up = F.linear(x_norm, self.w[f"{prefix}.mlp.up_proj.weight"])
            hidden = F.silu(gate) * up
            mlp_out = F.linear(hidden, self.w[f"{prefix}.mlp.down_proj.weight"])
            x = residual + mlp_out

        x = self._rmsnorm(x, self.w["norm.weight"])
        return F.linear(x, self.w["lm_head.weight"])
