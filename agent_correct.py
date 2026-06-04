import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path


BASELINE_ENGINE_SOURCE = r'''
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
'''


SYSTEM_PROMPT = """You are an MLSys Phase 3 code-generation agent.
Generate one complete Python file named engine.py for a decoder-only LLaMA-like runtime.
Return only Python code, with no Markdown fences and no explanation."""


def canonical_weight_schema(config):
    num_layers = int(config.get("num_hidden_layers", 0) or 0)
    keys = ["embed_tokens.weight"]
    layer_templates = [
        "layers.{i}.input_layernorm.weight",
        "layers.{i}.self_attn.q_proj.weight",
        "layers.{i}.self_attn.k_proj.weight",
        "layers.{i}.self_attn.v_proj.weight",
        "layers.{i}.self_attn.o_proj.weight",
        "layers.{i}.post_attention_layernorm.weight",
        "layers.{i}.mlp.gate_proj.weight",
        "layers.{i}.mlp.up_proj.weight",
        "layers.{i}.mlp.down_proj.weight",
    ]
    if num_layers:
        for i in range(num_layers):
            keys.extend(template.format(i=i) for template in layer_templates)
    else:
        keys.extend(layer_templates)
    keys.extend(["norm.weight", "lm_head.weight"])
    return "\n".join(f"- {key}" for key in keys)


def load_weight_schema(root, config):
    candidates = [
        root / "target" / "weights" / "model.pt",
        Path("/target/weights/model.pt"),
    ]
    weight_path = next((path for path in candidates if path.exists()), candidates[0])
    if not weight_path.exists():
        return (
            f"Weight file not available at generation time. Use this required naming pattern:\n"
            f"{canonical_weight_schema(config)}"
        )

    try:
        import torch

        try:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weight_path, map_location="cpu")
        lines = [f"Observed weight file: {weight_path}", "Exact state_dict keys and shapes:"]
        for name, tensor in state_dict.items():
            shape = tuple(int(x) for x in getattr(tensor, "shape", ()))
            lines.append(f"- {name}: {shape}")
        return "\n".join(lines)
    except Exception as exc:
        return (
            f"Could not inspect {weight_path}: {exc}\n"
            f"Use this required naming pattern:\n{canonical_weight_schema(config)}"
        )


def build_runtime_prompt(config, weight_schema):
    config_text = json.dumps(config, indent=2, sort_keys=True) if config else "{}"
    return textwrap.dedent(
        f"""
        Build workspace/engine.py for Phase 3 Automated LLM Inference Runtime.

        Required public interface:
        - def create_engine(model_config: dict, weight_dir: str, device: str = "cuda")
        - Engine.prefill(request_ids, input_ids) -> logits [batch, vocab_size].
          input_ids is a Python list of 1D torch.Tensor sequences, not a padded tensor.
        - Engine.decode(request_ids, token_ids) -> logits [batch, vocab_size].
          token_ids is a 1D torch.Tensor with one token per request.
        - Engine.remove(request_ids) -> None

        Correctness contract:
        - Load weight_dir/model.pt as a PyTorch state_dict.
        - Use exactly the state_dict key names listed below. Do not guess HuggingFace names
          such as model.embed_tokens.weight, tok_embeddings.weight, wte.weight, q_proj.bias,
          or any key that is not listed.
        - Match the reference logits within torch.allclose atol=1e-2, rtol=1e-2.
        - Correctness is more important than speed. If an optimized KV-cache implementation
          is uncertain, generate a full-recompute implementation that is exactly correct.
        - Use model_config dynamically; do not hard-code dimensions from this public config.
        - prefill creates or replaces only the listed request states.
        - decode appends exactly one token to each listed existing request.
        - remove deletes finished request states without disturbing others.

        Exact model math required:
        - LLaMA-like decoder-only stack.
        - embed_tokens.weight lookup.
        - RMSNorm: x_float = x.float(); variance = x_float.pow(2).mean(dim=-1, keepdim=True);
          output = (x_float * torch.rsqrt(variance + rms_norm_eps)).to(x.dtype) * weight.
        - Per layer: input RMSNorm, q/k/v projections with F.linear(x_norm, weight) and no bias.
        - q shape is [batch, num_attention_heads, seqlen, head_dim].
        - k/v shape is [batch, num_key_value_heads, seqlen, head_dim].
        - RoPE uses even/odd pairs:
          x_even=x[...,0::2], x_odd=x[...,1::2],
          rotated stack is (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos), then flatten last dims.
        - RoPE positions are absolute token positions starting at 0 for prefill and current
          cached length for decode.
        - Causal attention for full sequences is:
          attn = torch.matmul(q.float(), k.float().transpose(-1, -2)) / sqrt(head_dim)
          then add an upper-triangular -inf mask, softmax over last dim, cast to x.dtype.
        - For decode with KV cache, attention over all cached keys is not causal-masked because
          the query is only the newest token and may attend to the whole prefix plus itself.
        - Then o projection, residual, post-attention RMSNorm, SwiGLU MLP with gate/up/down, residual.
        - Support num_attention_heads != num_key_value_heads by repeating KV heads.
        - Final RMSNorm and lm_head.weight projection.
        - Use float32 for RMSNorm variance and attention softmax stability as needed.

        Engineering goal:
        - A simple full-recompute implementation is acceptable only as a fallback, but you
          should generate an optimized runtime.
        - Implement a per-layer KV cache keyed by request_id, so decode computes only the
          new token and appends one K/V vector per layer.
        - Batch same-length prefill requests and return logits in caller order.
        - In decode, group requests by current cache length when useful. Preserve caller order.
        - Precompute or cache RoPE cos/sin tables. Grow them dynamically for hidden traces.
        - Avoid torch.nn.functional.scaled_dot_product_attention unless you are certain its
          masking and dtype behavior matches the explicit reference above.
        - Minimize Python overhead without sacrificing correctness.
        - Avoid imports outside the Python standard library and torch.
        - Return syntactically valid, fully indented Python code only.

        Public config observed by the agent:
        {config_text}

        State dict schema observed by the agent:
        {weight_schema}
        """
    ).strip()


def extract_python_code(text):
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    text = text.strip()
    if "def create_engine" not in text or "class Engine" not in text:
        raise ValueError("candidate does not define the required engine interface")
    return text + "\n"


def call_deepseek_api(prompt):
    api_key = os.environ.get("API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None, "no DeepSeek API key found in API_KEY or DEEPSEEK_API_KEY"

    base_url = (
        os.environ.get("BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    model = os.environ.get("BASE_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return extract_python_code(content), f"generated by DeepSeek model {model} via {endpoint}"
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
        return None, f"DeepSeek generation failed: {exc}"


def generate_candidate_with_repair(prompt, root, engine_path):
    attempts = []
    candidate, note = call_deepseek_api(prompt)
    attempts.append(note)
    if candidate is None:
        return None, "\n".join(attempts)

    max_validations = 4
    for attempt in range(max_validations):
        try:
            syntax_check(candidate, engine_path)
            engine_path.write_text(candidate, encoding="utf-8")
            ok, validation = run_public_correctness(root, engine_path)
            attempts.append(validation)
            if ok:
                return candidate, "\n".join(attempts)
            if attempt == max_validations - 1:
                break
            repair_prompt = textwrap.dedent(
                f"""
                The generated engine.py failed validation. Produce a corrected complete
                engine.py. Return only Python code.

                Validation failure:
                {validation}

                Important reminders:
                - prefill input_ids is list[torch.Tensor], not a tensor with .shape.
                - Use only the exact state_dict keys listed in the original task.
                - Match the explicit RMSNorm, RoPE, causal attention, and SwiGLU math from
                  the original task. Correct full recompute is better than a wrong KV cache.

                Original task:
                {prompt}
                """
            ).strip()
            candidate, note = call_deepseek_api(repair_prompt)
            attempts.append(note)
            if candidate is None:
                break
        except Exception as exc:
            attempts.append(f"candidate validation failed: {exc}")
            if attempt == max_validations - 1:
                break
            repair_prompt = textwrap.dedent(
                f"""
                The generated engine.py failed before correctness testing. Produce a corrected
                complete engine.py. Return only Python code.

                Error:
                {exc}

                Important reminders:
                - Return syntactically valid, fully indented Python.
                - prefill input_ids is list[torch.Tensor], not a tensor with .shape.
                - Use only the exact state_dict keys listed in the original task.
                - Correct full recompute is better than a wrong optimized runtime.

                Original task:
                {prompt}
                """
            ).strip()
            candidate, note = call_deepseek_api(repair_prompt)
            attempts.append(note)
            if candidate is None:
                break

    return None, "\n".join(attempts)


def syntax_check(source, path):
    compile(source, str(path), "exec")


def run_public_correctness(root, engine_path):
    evaluator = root / "evaluator" / "test_correctness.py"
    config = root / "target" / "model_config.json"
    weights = root / "target" / "weights"
    if not evaluator.exists() or not config.exists() or not weights.exists():
        return True, "public correctness test not available in this directory"

    cmd = [
        sys.executable,
        str(evaluator),
        "--engine",
        str(engine_path),
        "--model-config",
        str(config),
        "--weight-dir",
        str(weights),
        "--device",
        "auto",
    ]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=240)
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    return False, (proc.stdout + "\n" + proc.stderr).strip()


def read_config(root):
    for path in (root / "target" / "model_config.json", Path("/target/model_config.json")):
        if path.exists():
            with path.open() as f:
                return json.load(f), path
    return {}, root / "target" / "model_config.json"


def build_output_report(
    config,
    config_path,
    weight_schema,
    engine_source,
    decision,
    validation_log,
):
    report = "\n".join(
        [
            "# Phase 3 Agent Output",
            "",
            "The agent generated `workspace/engine.py` during `run.sh`.",
            "",
            "Generation strategy:",
            "- Keep one embedded correctness baseline as a fallback.",
            "- Ask the DeepSeek-compatible course API for an optimized runtime when credentials are available.",
            "- Accept a generated candidate only after interface extraction, Python syntax validation,",
            "  and the public correctness test when the public evaluator is present.",
            "- If validation fails, ask the model once more with the concrete error before falling back.",
            "- Fall back to the single baseline if generation or validation fails.",
            "",
            f"Decision: {decision}",
            f"Generated engine size: {len(engine_source)} bytes",
            f"Config path used: {config_path}",
            "",
            "Validation log:",
            "",
            "```text",
            validation_log,
            "```",
            "",
            "Observed config:",
            "",
            "```json",
            json.dumps(config, indent=2, sort_keys=True) if config else "{}",
            "```",
            "",
            "Runtime prompt:",
            "",
            "```text",
            build_runtime_prompt(config, weight_schema),
            "```",
        ]
    )
    return report + "\n"


def main():
    root = Path(__file__).resolve().parent
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    engine_path = workspace / "engine.py"
    extra_engine_paths = []
    extra_output_paths = []
    if root.name == "workspace" or str(root) == "/workspace":
        extra_engine_paths.append(root / "engine.py")
        extra_output_paths.append(root / "output3.md")

    config, config_path = read_config(root)
    weight_schema = load_weight_schema(root, config)
    prompt = build_runtime_prompt(config, weight_schema)

    baseline_source = BASELINE_ENGINE_SOURCE.lstrip() + "\n"
    candidate_source, generation_note = generate_candidate_with_repair(prompt, root, engine_path)

    decision = "baseline fallback"
    validation_log = generation_note
    engine_source = baseline_source

    if candidate_source is not None:
        try:
            syntax_check(candidate_source, engine_path)
            engine_path.write_text(candidate_source, encoding="utf-8")
            decision = "accepted DeepSeek-generated candidate"
            engine_source = candidate_source
            validation_log = generation_note
        except Exception as exc:
            validation_log = f"{generation_note}\nCandidate validation failed: {exc}"
            engine_path.write_text(baseline_source, encoding="utf-8")
    else:
        engine_path.write_text(baseline_source, encoding="utf-8")

    syntax_check(engine_source, engine_path)
    engine_path.write_text(engine_source, encoding="utf-8")
    for path in extra_engine_paths:
        path.write_text(engine_source, encoding="utf-8")

    output_report = build_output_report(
        config,
        config_path,
        weight_schema,
        engine_source,
        decision,
        validation_log,
    )
    (workspace / "output3.md").write_text(output_report, encoding="utf-8")
    for path in extra_output_paths:
        path.write_text(output_report, encoding="utf-8")

    print(f"generated {engine_path}")
    print(f"generated {workspace / 'output3.md'}")
    for path in extra_engine_paths:
        print(f"generated {path}")
    for path in extra_output_paths:
        print(f"generated {path}")
    print(f"decision: {decision}")
    print(validation_log)


if __name__ == "__main__":
    main()
