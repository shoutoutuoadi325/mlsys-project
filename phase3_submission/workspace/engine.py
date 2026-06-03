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


class CacheBlock:
    def __init__(self, capacity, keys, values):
        self.capacity = capacity
        self.keys = keys
        self.values = values


class RequestState:
    def __init__(self, length, block, row):
        self.length = length
        self.block = block
        self.row = row


class Engine:
    def __init__(self, config, weight_dir, device):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.config = config
        self.device = torch.device(device)
        self.dtype = self._select_dtype(config)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

        weight_path = os.path.join(weight_dir, "model.pt")
        state_dict = load_state_dict(weight_path)
        self.w = {
            name: tensor.to(device=self.device, dtype=self.dtype).contiguous()
            for name, tensor in state_dict.items()
        }

        self.num_layers = int(config["num_hidden_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.hidden_size = int(config["hidden_size"])
        self.eps = float(config.get("rms_norm_eps", 1e-5))
        self.rope_theta = float(config.get("rope_theta", 10000.0))
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.embed_weight = self.w["embed_tokens.weight"]
        self.norm_weight = self.w["norm.weight"]
        self.lm_head_weight = self.w["lm_head.weight"]
        self.layers = []
        for layer_idx in range(self.num_layers):
            prefix = f"layers.{layer_idx}"
            qkv_weight = torch.cat(
                [
                    self.w[f"{prefix}.self_attn.q_proj.weight"],
                    self.w[f"{prefix}.self_attn.k_proj.weight"],
                    self.w[f"{prefix}.self_attn.v_proj.weight"],
                ],
                dim=0,
            ).contiguous()
            gate_up_weight = torch.cat(
                [
                    self.w[f"{prefix}.mlp.gate_proj.weight"],
                    self.w[f"{prefix}.mlp.up_proj.weight"],
                ],
                dim=0,
            ).contiguous()
            self.layers.append(
                (
                    self.w[f"{prefix}.input_layernorm.weight"],
                    self.w[f"{prefix}.post_attention_layernorm.weight"],
                    qkv_weight,
                    self.w[f"{prefix}.self_attn.o_proj.weight"],
                    gate_up_weight,
                    self.w[f"{prefix}.mlp.down_proj.weight"],
                )
            )
            for suffix in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            ):
                self.w.pop(f"{prefix}.{suffix}", None)

        max_positions = int(config.get("max_position_embeddings", 2048))
        self.rope_cos = None
        self.rope_sin = None
        self._build_rope_cache(max(max_positions, 16))

        self.causal_masks = {}
        self.requests = {}
        if self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self._forward_prefill_batch = torch.compile(
                    self._forward_prefill_batch,
                    mode="reduce-overhead",
                )
            except Exception:
                pass

    def _select_dtype(self, config):
        if self.device.type != "cuda":
            return torch.float32

        dtype = str(config.get("torch_dtype", "float16")).lower()
        if dtype in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float16

    def prefill(self, request_ids, input_ids):
        request_ids = [int(rid) for rid in request_ids]
        inputs = self._normalize_input_ids(input_ids)
        if len(request_ids) == 0:
            vocab = int(self.config["vocab_size"])
            return torch.empty((0, vocab), device=self.device, dtype=self.dtype)
        if len(request_ids) != len(inputs):
            raise ValueError("request_ids and input_ids must have the same length")

        groups = {}
        for out_idx, (rid, ids) in enumerate(zip(request_ids, inputs)):
            ids = ids.to(device=self.device, dtype=torch.long).reshape(-1)
            if ids.numel() == 0:
                raise ValueError("prefill input_ids must be non-empty")
            groups.setdefault(int(ids.numel()), []).append((out_idx, rid, ids))

        outputs = [None] * len(request_ids)
        with torch.inference_mode():
            for seqlen, items in groups.items():
                batch_ids = torch.stack([ids for _, _, ids in items], dim=0)
                logits, keys, values = self._forward_prefill_batch(batch_ids)

                capacity = self._initial_capacity(seqlen)
                block_keys = []
                block_values = []
                for layer_idx in range(self.num_layers):
                    k_buf = torch.empty(
                        (len(items), self.num_kv_heads, capacity, self.head_dim),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    v_buf = torch.empty_like(k_buf)
                    k_buf[:, :, :seqlen, :].copy_(keys[layer_idx])
                    v_buf[:, :, :seqlen, :].copy_(values[layer_idx])
                    block_keys.append(k_buf)
                    block_values.append(v_buf)
                block = CacheBlock(capacity=capacity, keys=block_keys, values=block_values)

                for local_idx, (out_idx, rid, _) in enumerate(items):
                    self.requests[rid] = RequestState(
                        length=seqlen,
                        block=block,
                        row=local_idx,
                    )
                    outputs[out_idx] = logits[local_idx]

        return torch.stack(outputs, dim=0)

    def decode(self, request_ids, token_ids):
        request_ids = [int(rid) for rid in request_ids]
        tokens = self._normalize_token_ids(token_ids)
        if len(request_ids) == 0:
            vocab = int(self.config["vocab_size"])
            return torch.empty((0, vocab), device=self.device, dtype=self.dtype)
        if len(request_ids) != int(tokens.numel()):
            raise ValueError("request_ids and token_ids must have the same length")

        groups = {}
        for out_idx, rid in enumerate(request_ids):
            if rid not in self.requests:
                raise KeyError(f"unknown request_id {rid}; call prefill first")
            groups.setdefault(self.requests[rid].length, []).append(out_idx)

        outputs = [None] * len(request_ids)
        with torch.inference_mode():
            for _, out_indices in groups.items():
                group_rids = [request_ids[i] for i in out_indices]
                group_tokens = tokens[out_indices].to(device=self.device, dtype=torch.long)
                logits = self._decode_group(group_rids, group_tokens)
                for local_idx, out_idx in enumerate(out_indices):
                    outputs[out_idx] = logits[local_idx]

        return torch.stack(outputs, dim=0)

    def remove(self, request_ids):
        for rid in request_ids:
            self.requests.pop(int(rid), None)

    def _initial_capacity(self, length):
        capacity = 16
        target = int(length) + 16
        while capacity < target:
            capacity *= 2
        return capacity

    def _ensure_capacity(self, state, needed_length):
        block = state.block
        if needed_length <= block.capacity:
            return
        old_capacity = block.capacity
        new_capacity = old_capacity
        while new_capacity < needed_length:
            new_capacity *= 2
        for layer_idx in range(self.num_layers):
            old_k = block.keys[layer_idx]
            old_v = block.values[layer_idx]
            k_buf = torch.empty(
                (old_k.shape[0], self.num_kv_heads, new_capacity, self.head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            v_buf = torch.empty_like(k_buf)
            k_buf[:, :, :old_capacity, :].copy_(old_k[:, :, :old_capacity, :])
            v_buf[:, :, :old_capacity, :].copy_(old_v[:, :, :old_capacity, :])
            block.keys[layer_idx] = k_buf
            block.values[layer_idx] = v_buf
        block.capacity = new_capacity

    def _normalize_input_ids(self, input_ids):
        if torch.is_tensor(input_ids):
            if input_ids.dim() == 1:
                return [input_ids]
            return [row for row in input_ids]
        return list(input_ids)

    def _normalize_token_ids(self, token_ids):
        if torch.is_tensor(token_ids):
            return token_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        return torch.tensor(list(token_ids), device=self.device, dtype=torch.long).reshape(-1)

    def _build_rope_cache(self, length):
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        positions = torch.arange(length, device=self.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.rope_cos = freqs.cos()
        self.rope_sin = freqs.sin()

    def _ensure_rope(self, needed_length):
        if needed_length <= int(self.rope_cos.shape[0]):
            return
        new_length = max(needed_length, int(self.rope_cos.shape[0]) * 2)
        self._build_rope_cache(new_length)

    def _causal_mask(self, seqlen):
        mask = self.causal_masks.get(seqlen)
        if mask is None or mask.device != self.device:
            mask = torch.triu(
                torch.full(
                    (seqlen, seqlen),
                    float("-inf"),
                    device=self.device,
                    dtype=torch.float32,
                ),
                diagonal=1,
            )[None, None, :, :]
            self.causal_masks[seqlen] = mask
        return mask

    def _rmsnorm(self, x, weight):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return x_norm.to(x.dtype) * weight

    def _apply_rope_prefill(self, q, k, seqlen):
        self._ensure_rope(seqlen)
        cos = self.rope_cos[:seqlen].to(dtype=q.dtype)[None, None, :, :]
        sin = self.rope_sin[:seqlen].to(dtype=q.dtype)[None, None, :, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _apply_rope_decode(self, q, k, positions):
        max_position = int(positions.max().item()) + 1
        self._ensure_rope(max_position)
        cos = self.rope_cos.index_select(0, positions).to(dtype=q.dtype)[:, None, :]
        sin = self.rope_sin.index_select(0, positions).to(dtype=q.dtype)[:, None, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _apply_rope_decode_position(self, q, k, position):
        self._ensure_rope(position + 1)
        cos = self.rope_cos[position].to(dtype=q.dtype)[None, None, :]
        sin = self.rope_sin[position].to(dtype=q.dtype)[None, None, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            dim=-1,
        )
        return x_rotated.flatten(-2)

    def _repeat_kv(self, x):
        if self.kv_repeat == 1:
            return x
        return x.repeat_interleave(self.kv_repeat, dim=1)

    def _attention_prefill(self, q, k, v):
        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)
        return F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            dropout_p=0.0,
            is_causal=True,
        )

    def _attention_decode(self, q, k, v):
        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)
        return F.scaled_dot_product_attention(
            q.unsqueeze(2),
            k_full,
            v_full,
            dropout_p=0.0,
            is_causal=False,
        ).squeeze(2)

    def _forward_prefill_batch(self, input_ids):
        x = self.embed_weight[input_ids]
        batch, seqlen, _ = x.shape
        all_keys = []
        all_values = []

        for layer_idx in range(self.num_layers):
            input_norm_w, post_norm_w, qkv_w, o_w, gate_up_w, down_w = self.layers[layer_idx]

            residual = x
            x_norm = self._rmsnorm(x, input_norm_w)

            qkv = F.linear(x_norm, qkv_w)
            q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)

            q = q.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope_prefill(q, k, seqlen)
            k = k.contiguous()
            v = v.contiguous()
            all_keys.append(k)
            all_values.append(v)

            y = self._attention_prefill(q, k, v)
            y = y.transpose(1, 2).contiguous().view(batch, seqlen, self.hidden_size)
            y = F.linear(y, o_w)
            x = residual + y

            residual = x
            x_norm = self._rmsnorm(x, post_norm_w)
            gate_up = F.linear(x_norm, gate_up_w)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            mlp_out = F.linear(hidden, down_w)
            x = residual + mlp_out

        x = self._rmsnorm(x, self.norm_weight)
        return F.linear(x[:, -1, :], self.lm_head_weight), all_keys, all_values

    def _decode_group(self, request_ids, token_ids):
        states = [self.requests[rid] for rid in request_ids]
        batch = len(states)
        old_len = states[0].length
        new_len = old_len + 1
        for state in states:
            self._ensure_capacity(state, new_len)
        first_block = states[0].block
        same_block = all(state.block is first_block for state in states)
        rows = [state.row for state in states]
        row_start = rows[0]
        contiguous_rows = same_block and rows == list(range(row_start, row_start + batch))
        row_tensor = None
        if same_block and not contiguous_rows:
            row_tensor = torch.tensor(rows, device=self.device, dtype=torch.long)

        x = self.embed_weight[token_ids].view(batch, self.hidden_size)

        for layer_idx in range(self.num_layers):
            input_norm_w, post_norm_w, qkv_w, o_w, gate_up_w, down_w = self.layers[layer_idx]

            residual = x
            x_norm = self._rmsnorm(x, input_norm_w)

            qkv = F.linear(x_norm, qkv_w)
            q, k_new, v_new = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)

            q = q.view(batch, self.num_heads, self.head_dim)
            k_new = k_new.view(batch, self.num_kv_heads, self.head_dim)
            v_new = v_new.view(batch, self.num_kv_heads, self.head_dim)
            q, k_new = self._apply_rope_decode_position(q, k_new, old_len)

            if contiguous_rows:
                layer_k = first_block.keys[layer_idx]
                layer_v = first_block.values[layer_idx]
                layer_k[row_start : row_start + batch, :, old_len, :].copy_(k_new)
                layer_v[row_start : row_start + batch, :, old_len, :].copy_(v_new)
                k_cache = layer_k[row_start : row_start + batch, :, :new_len, :]
                v_cache = layer_v[row_start : row_start + batch, :, :new_len, :]
            elif same_block:
                layer_k = first_block.keys[layer_idx]
                layer_v = first_block.values[layer_idx]
                layer_k[:, :, old_len, :].index_copy_(0, row_tensor, k_new)
                layer_v[:, :, old_len, :].index_copy_(0, row_tensor, v_new)
                k_cache = layer_k[:, :, :new_len, :].index_select(0, row_tensor)
                v_cache = layer_v[:, :, :new_len, :].index_select(0, row_tensor)
            else:
                for idx, state in enumerate(states):
                    state.block.keys[layer_idx][state.row, :, old_len, :].copy_(k_new[idx])
                    state.block.values[layer_idx][state.row, :, old_len, :].copy_(v_new[idx])
                k_cache = torch.stack(
                    [
                        state.block.keys[layer_idx][state.row, :, :new_len, :]
                        for state in states
                    ],
                    dim=0,
                )
                v_cache = torch.stack(
                    [
                        state.block.values[layer_idx][state.row, :, :new_len, :]
                        for state in states
                    ],
                    dim=0,
                )

            y = self._attention_decode(q, k_cache, v_cache)
            y = y.contiguous().view(batch, self.hidden_size)
            y = F.linear(y, o_w)
            x = residual + y

            residual = x
            x_norm = self._rmsnorm(x, post_norm_w)
            gate_up = F.linear(x_norm, gate_up_w)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            mlp_out = F.linear(hidden, down_w)
            x = residual + mlp_out

        for state in states:
            state.length += 1

        x = self._rmsnorm(x, self.norm_weight)
        return F.linear(x, self.lm_head_weight)

