import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path


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


def read_config(root):
    candidates = (root / "target" / "model_config.json", Path("/target/model_config.json"))
    for path in candidates:
        if path.exists():
            with path.open() as f:
                return json.load(f), path
    return {}, candidates[0]


def load_weight_schema(root, config):
    candidates = [
        root / "target" / "weights" / "model.pt",
        Path("/target/weights/model.pt"),
    ]
    weight_path = next((path for path in candidates if path.exists()), candidates[0])
    if not weight_path.exists():
        return (
            "Weight file is unavailable at generation time. Use this required naming pattern:\n"
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


def build_runtime_prompt(config, weight_schema, feedback=""):
    config_text = json.dumps(config, indent=2, sort_keys=True) if config else "{}"
    feedback_text = feedback.strip() or "No previous candidate feedback."
    return textwrap.dedent(
        f"""
        Build /workspace/engine.py for Phase 3 Automated LLM Inference Runtime.

        You must output a complete, importable Python module. The module must provide
        create_engine(model_config, weight_dir, device="cuda"). The returned runtime
        object must expose three methods named prefill(request_ids, input_ids),
        decode(request_ids, token_ids), and remove(request_ids). Use only Python,
        PyTorch, and the standard library.
        Every output line must be syntactically valid Python. Do not include reasoning,
        TODO placeholders, bracketed pseudo-code, English implementation notes, partial
        refactor notes, Markdown, or comments that are not valid Python comments.

        Correctness is mandatory:
        - Load weight_dir/model.pt as a PyTorch state_dict.
        - Use the exact state_dict keys listed below; do not invent HuggingFace names,
          bias tensors, or alternate layouts.
        - Do not build a torch.nn.Module and call load_state_dict(strict=True) if that
          would introduce unlisted keys such as rotary_emb.inv_freq. Plain Python
          containers holding the listed tensors are safer for this assignment.
        - Respect model_config dynamically. Do not bake in dimensions, batch sizes,
          sequence lengths, or trace patterns from the public case.
        - model_config is a Python dict. Always read values with config["key"] or
          config.get("key"), never with config.key or self.config.key.
        - prefill creates or replaces only the listed request IDs.
        - decode appends exactly one token to each listed existing request.
        - remove deletes only the requested states.
        - Match the official reference logits with torch.allclose atol=1e-2, rtol=1e-2.
        - prefill and decode must return only last-token logits with shape
          [batch_size, vocab_size], never [batch_size, sequence_length, vocab_size].

        Exact model math:
        - Decoder-only LLaMA-like stack with token embedding, per-layer RMSNorm,
          self-attention, residual, post-attention RMSNorm, SwiGLU MLP, residual,
          final RMSNorm, and lm_head projection.
        - RMSNorm computes variance in float32, then casts back before multiplying
          by the learned norm weight.
        - Attention projections are bias-free F.linear calls. Query uses
          num_attention_heads, while key/value use num_key_value_heads.
        - RoPE is the even/odd pair rotation used by the reference: rotate pairs as
          (even*cos - odd*sin, even*sin + odd*cos), then flatten the last pair axis.
          Do not use the HuggingFace rotate_half convention unless it is mathematically
          identical for this even/odd layout. A common bug is making cos/sin width
          head_dim/2 and multiplying it directly with a head_dim tensor.
        - Prefill uses absolute positions starting at zero and causal attention.
        - Decode uses the current cached sequence length as the new token position.
          A single newest query attends to all cached keys including itself, so no
          causal mask is needed for decode.
        - If attention heads differ from key/value heads, repeat KV heads to match
          attention heads.

        Performance target:
        - The evaluation cases include large batched prefill and mixed serving traces.
          Aim for roughly 45k tokens/s on the prefill-heavy case and more than
          10k tokens/s on the mixed case for the public tiny LLaMA config.
        - Implement real per-layer KV cache so decode computes only the new token.
        - Batch same-length prefill requests and preserve caller order.
        - For same-length prefill groups, stack sequences directly. Avoid padding and
          padding masks unless absolutely necessary, because public traces often have
          equal-length groups and padding mistakes commonly break last-token indexing.
        - input_ids passed to prefill is already a list of 1D torch.Tensor objects.
          Never call torch.tensor(list_of_tensors). For a same-length group use
          torch.stack([ids.to(device=device, dtype=torch.long) for ids in group], dim=0).
        - In decode, group requests by current cached length and batch each group.
        - Precompute or dynamically grow RoPE tables and reuse causal masks.
        - In the weight-loading/setup path, combine q/k/v projection matrices and
          combine gate/up projection matrices to reduce matmul launch count.
        - Prefer torch.inference_mode for public runtime calls.
        - Use PyTorch scaled_dot_product_attention on CUDA when it preserves the
          required math; otherwise implement explicit float32 softmax attention.
        - Avoid expensive Python loops over tokens. Per-layer loops are acceptable.
        - Avoid storing full token histories; store compact per-request K/V state.
        - Do not use complete inference frameworks.

        Common failure fixes to apply proactively:
        - If logits have shape [batch, seq, vocab], return logits[:, -1, :].
        - If load_state_dict complains about rotary_emb.inv_freq, stop registering that
          buffer and keep RoPE tables as ordinary cached tensors not loaded from weights.
        - If RoPE has a head_dim versus head_dim/2 mismatch, split q/k into even and odd
          components, multiply those by cos/sin, stack the two rotated components, and
          flatten back to head_dim.
        - If correctness fails after padding prompts, group requests by exact prompt
          length and run each group without padding.
        - If PyTorch says "only integer tensors of a single element can be converted
          to an index", replace torch.tensor(list_of_tensors) with torch.stack after
          moving each tensor to the target device and dtype.

        Public config observed by this agent:
        {config_text}

        State dict schema observed by this agent:
        {weight_schema}

        Previous candidate feedback:
        {feedback_text}
        """
    ).strip()


def extract_python_code(text):
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    text = text.strip()
    if "def create_engine" not in text:
        raise ValueError("candidate does not define create_engine")
    banned_imports = ("vllm", "transformers", "llama_cpp", "exllama")
    lowered = text.lower()
    for name in banned_imports:
        if name in lowered:
            raise ValueError(f"candidate imports or references disallowed framework: {name}")
    return text + "\n"


def call_model(prompt, temperature=0.15):
    api_key = os.environ.get("API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("no API key found in API_KEY or DEEPSEEK_API_KEY")

    base_url = (
        os.environ.get("BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    model = (
        os.environ.get("PHASE3_MODEL")
        or os.environ.get("BASE_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or "deepseek-chat"
    )
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
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
    with urllib.request.urlopen(request, timeout=240) as response:
        data = json.loads(response.read().decode("utf-8"))
    return extract_python_code(data["choices"][0]["message"]["content"])


def syntax_check(source, engine_path):
    compile(source, str(engine_path), "exec")


def auto_repair_source(source):
    for key in (
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "torch_dtype",
    ):
        source = source.replace(f"self.config.{key}", f"self.config[{key!r}]")
        source = source.replace(f"config.{key}", f"config[{key!r}]")

    list_comp_pattern = re.compile(
        r"torch\.tensor\(\s*"
        r"\[([A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?)\s+for\s+([^\]]+)\]"
        r"\s*,\s*device=([^,\)]+),\s*dtype=torch\.long\s*\)"
    )
    source = list_comp_pattern.sub(
        r"torch.stack([\1.to(device=\3, dtype=torch.long) for \2], dim=0)",
        source,
    )

    for name in ("batch_ids", "batch_seqs", "group_seqs", "seqs", "input_list"):
        pattern = re.compile(
            r"torch\.tensor\(\s*"
            + re.escape(name)
            + r"\s*,\s*device=([^,\)]+),\s*dtype=torch\.long\s*\)"
        )
        source = pattern.sub(
            f"torch.stack([x.to(device=\\1, dtype=torch.long) for x in {name}], dim=0)",
            source,
        )
    return source


def run_command(cmd, cwd, timeout):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return proc.returncode == 0, (proc.stdout + "\n" + proc.stderr).strip()


def run_public_correctness(root, engine_path):
    evaluator = root / "evaluator" / "test_correctness.py"
    config = root / "target" / "model_config.json"
    weights = root / "target" / "weights"
    if not evaluator.exists() or not config.exists() or not weights.exists():
        return True, "public correctness evaluator not available"
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
    return run_command(cmd, root, timeout=240)


def run_public_benchmark(root, engine_path):
    evaluator = root / "evaluator" / "benchmark_throughput.py"
    config = root / "target" / "model_config.json"
    weights = root / "target" / "weights"
    if not evaluator.exists() or not config.exists() or not weights.exists():
        return True, "public benchmark evaluator not available"
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
        "--warmup",
        "1",
        "--repeat",
        "3",
    ]
    return run_command(cmd, root, timeout=360)


def summarize_benchmark_for_feedback(text):
    try:
        match = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)
        if not match:
            return text
        results = json.loads(match.group(0))
    except Exception:
        return text
    lines = ["Benchmark results:"]
    for row in results:
        name = row.get("case_name")
        tps = row.get("tokens_per_second")
        dtps = row.get("decode_tokens_per_second")
        mem = row.get("peak_memory_mb")
        lines.append(f"- {name}: tokens/s={tps}, decode_tokens/s={dtps}, peak_memory_mb={mem}")
    return "\n".join(lines)


def candidate_is_fast_enough(benchmark_text):
    try:
        match = re.search(r"\[\s*\{.*\}\s*\]", benchmark_text, flags=re.DOTALL)
        if not match:
            return False
        results = json.loads(match.group(0))
    except Exception:
        return False
    by_name = {str(row.get("case_name")): row for row in results}
    prefill = by_name.get("prefill") or by_name.get("1")
    mixed = by_name.get("mixed") or by_name.get("4")
    if not prefill or not mixed:
        return False
    return (
        float(prefill.get("tokens_per_second", 0.0)) >= 40000.0
        and float(mixed.get("tokens_per_second", 0.0)) >= 9500.0
    )


def generate_candidate_with_feedback(root, engine_path, config, weight_schema):
    attempts = []
    feedback = ""
    best_source = None
    best_log = ""

    for attempt in range(6):
        prompt = build_runtime_prompt(config, weight_schema, feedback)
        try:
            source = call_model(prompt, temperature=0.1 if attempt else 0.15)
            source = auto_repair_source(source)
            syntax_check(source, engine_path)
            engine_path.write_text(source, encoding="utf-8")
        except Exception as exc:
            feedback = f"Generation or syntax failed on attempt {attempt + 1}: {exc}"
            attempts.append(feedback)
            continue

        ok, correctness_log = run_public_correctness(root, engine_path)
        attempts.append(f"attempt {attempt + 1} correctness:\n{correctness_log}")
        if not ok:
            feedback = textwrap.dedent(
                f"""
                The previous candidate failed correctness. Fix the complete module.
                Validation output:
                {correctness_log}
                """
            ).strip()
            continue

        ok, benchmark_log = run_public_benchmark(root, engine_path)
        attempts.append(f"attempt {attempt + 1} benchmark:\n{benchmark_log}")
        best_source = source
        best_log = "\n\n".join(attempts)
        if ok and candidate_is_fast_enough(benchmark_log):
            return source, best_log, "accepted generated candidate meeting public speed target"

        feedback = textwrap.dedent(
            f"""
            The previous candidate passed correctness but was not fast enough.
            Keep correctness identical and improve throughput.

            {summarize_benchmark_for_feedback(benchmark_log)}

            Focus on real KV cache decode, batched same-length prefill, grouped decode,
            fused qkv/gate-up projections, cached RoPE/masks, inference_mode, and avoiding
            unnecessary contiguous clones of per-request cache slices.
            """
        ).strip()

    if best_source is not None:
        return best_source, best_log, "accepted best correct generated candidate"
    raise RuntimeError("\n\n".join(attempts) or "no candidate generated")


def build_output_report(config, config_path, weight_schema, decision, validation_log):
    return (
        "# Phase 3 Agent Output\n\n"
        "The agent generated `workspace/engine.py` from natural-language runtime instructions.\n\n"
        "Generation strategy:\n"
        "- Read model_config.json and model.pt schema dynamically.\n"
        "- Ask the provided LLM endpoint for a complete runtime implementation.\n"
        "- Validate syntax, public correctness, and public throughput when available.\n"
        "- Feed concrete correctness or benchmark feedback back to the model for repair.\n\n"
        f"Decision: {decision}\n"
        f"Config path used: {config_path}\n\n"
        "Validation log:\n\n"
        "```text\n"
        f"{validation_log}\n"
        "```\n\n"
        "Observed config:\n\n"
        "```json\n"
        f"{json.dumps(config, indent=2, sort_keys=True) if config else '{}'}\n"
        "```\n\n"
        "Observed weight schema:\n\n"
        "```text\n"
        f"{weight_schema}\n"
        "```\n"
    )


def main():
    root = Path(__file__).resolve().parent
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    engine_path = workspace / "engine.py"
    output_path = workspace / "output3.md"

    extra_engine_paths = []
    extra_output_paths = []
    if root.name == "workspace" or str(root) == "/workspace":
        extra_engine_paths.append(root / "engine.py")
        extra_output_paths.append(root / "output3.md")

    config, config_path = read_config(root)
    weight_schema = load_weight_schema(root, config)

    try:
        source, validation_log, decision = generate_candidate_with_feedback(
            root, engine_path, config, weight_schema
        )
    except Exception as exc:
        validation_log = f"Agent failed to generate a valid runtime: {exc}"
        output_path.write_text(
            build_output_report(config, config_path, weight_schema, "failed", validation_log),
            encoding="utf-8",
        )
        raise

    syntax_check(source, engine_path)
    engine_path.write_text(source, encoding="utf-8")
    for path in extra_engine_paths:
        path.write_text(source, encoding="utf-8")

    report = build_output_report(config, config_path, weight_schema, decision, validation_log)
    output_path.write_text(report, encoding="utf-8")
    for path in extra_output_paths:
        path.write_text(report, encoding="utf-8")

    print(f"generated {engine_path}")
    print(f"generated {output_path}")
    for path in extra_engine_paths:
        print(f"generated {path}")
    for path in extra_output_paths:
        print(f"generated {path}")
    print(f"decision: {decision}")
    print(validation_log)


if __name__ == "__main__":
    main()
