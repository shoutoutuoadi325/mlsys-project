import json
import os
import subprocess
from pathlib import Path
from llm.openai_client import client

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "agent" / "prompts"
STATE_FILE = ROOT / "output.json"
FIRST_ITERATION_METRICS_FILE = "/target/target_spec.json"

class ProfilingAgent:
    def __init__(self):
        self.state = self.load_state()
        self.benchmark = "gemm"  # Focus on matrix multiplication

    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        else:
            return {
                "iteration": 0,
                "code_version": 0,
                "current_version": 0,
                "metrics_history": [],
                "analysis_history": [],
                "recommended_metrics_history": [],
                "new_benchmarks": [],
                "error_history": [],
                "current_bottleneck": None,
                "done": False
            }

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def load_prompt(self, file_name: str) -> str:
        prompt_path = PROMPT_DIR / file_name
        return prompt_path.read_text(encoding="utf-8")

    def get_current_benchmark_name(self) -> str:
        if self.state["current_version"] == 0:
            return self.benchmark
        return f"{self.benchmark}_v{self.state['current_version']}"

    def run_profile(self, metrics=None):
        benchmark_name = self.get_current_benchmark_name()
        cmd = ["python", str(ROOT / "runner" / "run.py"), "--benchmark", benchmark_name, "--profile"]
        if metrics:
            if isinstance(metrics, list):
                metrics = ",".join(metrics)
            cmd.extend(["--metrics", metrics])
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Profiling failed: {result.stderr}")
        return result.stdout + result.stderr

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end == -1:
                end = len(text)
            return text[start:end].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end == -1:
                end = len(text)
            return text[start:end].strip()
        return text.strip()

    def load_first_iteration_metrics(self) -> list[str]:
        if not FIRST_ITERATION_METRICS_FILE.exists():
            raise FileNotFoundError(
                f"First-iteration metrics file not found: {FIRST_ITERATION_METRICS_FILE}"
            )

        with open(FIRST_ITERATION_METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", [])
        if not isinstance(metrics, list) or not all(isinstance(m, str) and m.strip() for m in metrics):
            raise ValueError(
                f"Invalid metrics format in {FIRST_ITERATION_METRICS_FILE}, expected "
                f'{{"metrics": ["metric1", "metric2", ...]}}'
            )

        return [m.strip() for m in metrics]

    def analyze_metrics(self, profile_output, error_context=None, retry: int = 0):
        # Use LLM to analyze the profiling output
        if self.state["iteration"] == 1:
            baseline_metrics = self.load_first_iteration_metrics()
            metrics_instruction = (
                "Use the baseline metrics for the first iteration from the metrics file: "
                + ", ".join(baseline_metrics)
                + "."
            )
        else:
            metrics_instruction = (
                "This is not the first iteration. Analyze the output and recommend a focused set of ncu metrics"
                " for the next profiling pass. Include the recommended metrics in the JSON response."
            )

        template = self.load_prompt("analyze_metrics.txt")
        prompt = template.format(
            iteration=self.state["iteration"],
            profile_output=profile_output,
            metrics_instruction=metrics_instruction,
        )
        if error_context:
            prompt += f"\nThe previous LLM response or execution had an issue:\n{error_context}\nPlease correct it and return valid JSON only."

        response = client.chat.completions.create(
            model=os.getenv("BASE_MODEL", ""),
            messages=[{"role": "user", "content": prompt}]
        )
        analysis = response.choices[0].message.content
        # print("LLM analysis response:\n", analysis)
        json_str = self._extract_json(analysis)
        try:
            analysis_json = json.loads(json_str)
        except json.JSONDecodeError as exc:
            error_text = f"JSON decode error: {exc}\nResponse:\n{analysis}"
            self.state["error_history"].append({
                "iteration": self.state["iteration"],
                "type": "json_parse",
                "message": error_text
            })
            if retry < 1:
                return self.analyze_metrics(profile_output, error_context=error_text, retry=retry + 1)
            return {"bottleneck": "unknown", "key_metrics": analysis, "new_benchmark_description": "No new benchmark suggested"}
        return analysis_json

    def _parse_metrics(self, metrics):
        if not metrics:
            return []
        if isinstance(metrics, list):
            return [str(m).strip() for m in metrics if str(m).strip()]
        if isinstance(metrics, str):
            return [m.strip() for m in metrics.split(",") if m.strip()]
        return []

    def generate_new_benchmark(self, description, error_context=None, retry: int = 0):
        prompt_template = self.load_prompt("generate_benchmark.txt")
        prompt = prompt_template.format(description=description)
        if error_context:
            prompt += f"\nThe previous code generation or profiling step failed with this error:\n{error_context}\nPlease fix the CUDA code and return valid CUDA source only."

        response = client.chat.completions.create(
            model=os.getenv("BASE_MODEL", "")
            messages=[{"role": "user", "content": prompt}]
        )
        new_code = response.choices[0].message.content
        code_text = new_code
        if "```cuda" in new_code:
            start = new_code.find("```cuda") + 7
            end = new_code.find("```", start)
            if end == -1:
                end = len(new_code)
            code_text = new_code[start:end].strip()
        elif "```" in new_code:
            start = new_code.find("```") + 3
            end = new_code.find("```", start)
            if end == -1:
                end = len(new_code)
            code_text = new_code[start:end].strip()

        if not code_text.strip():
            error_text = f"Generated code was empty. Raw response:\n{new_code}"
            self.state["error_history"].append({
                "iteration": self.state["iteration"],
                "type": "code_generation",
                "message": error_text
            })
            if retry < 1:
                return self.generate_new_benchmark(description, error_context=error_text, retry=retry + 1)
            raise RuntimeError(error_text)

        new_version = self.state["code_version"] + 1
        new_code_path = ROOT / "benchmarks" / f"{self.benchmark}_v{new_version}.cu"
        with open(new_code_path, 'w') as f:
            f.write(code_text)
        self.state["code_version"] = new_version
        self.state["current_version"] = new_version
        return new_version

    def iterate(self):
        while not self.state["done"] and self.state["iteration"] < 1:  # Max 3 iterations for demo
            self.state["iteration"] += 1
            print(f"Iteration {self.state['iteration']}")

            metrics = None
            if self.state["iteration"] > 1 and self.state.get("recommended_metrics_history"):
                metrics = self.state["recommended_metrics_history"][-1]
                if metrics:
                    print(f"Using recommended metrics for this iteration: {metrics}")

            try:
                self.generate_new_benchmark("")
                profile_output = self.run_profile(metrics=metrics)
                print(profile_output)
            except RuntimeError as exc:
                error_info = str(exc)
                self.state["error_history"].append({
                    "iteration": self.state["iteration"],
                    "type": "profile",
                    "message": error_info
                })
                print(f"Profile failed: {error_info}")
                last_description = self.state["new_benchmarks"][-1] if self.state["new_benchmarks"] else "Generate a CUDA benchmark program for matrix multiplication profiling."
                try:
                    self.generate_new_benchmark(last_description, error_context=error_info)
                    self.state["new_benchmarks"].append(last_description)
                    self.save_state()
                    continue
                except RuntimeError as gen_exc:
                    print(f"Failed to regenerate benchmark after profile error: {gen_exc}")
                    break

            self.state["metrics_history"].append(profile_output)

            analysis = self.analyze_metrics(profile_output)
            self.state["analysis_history"].append(analysis)
            self.state["current_bottleneck"] = analysis["bottleneck"]

            print(f"Bottleneck: {analysis['bottleneck']}")
            print(f"Key Metrics: {analysis['key_metrics']}")
            print(f"Recommended Metrics: {analysis.get('recommended_metrics')}")
            print(f"New Benchmark Description: {analysis['new_benchmark_description']}")

            if not analysis["new_benchmark_description"] or analysis["new_benchmark_description"] == "No new benchmark suggested":
                self.state["done"] = True
                break

            recommended_metrics = self._parse_metrics(analysis.get("recommended_metrics"))
            self.state.setdefault("recommended_metrics_history", []).append(recommended_metrics)
            self.state["analysis_history"][-1]["recommended_metrics"] = recommended_metrics

            try:
                self.generate_new_benchmark(analysis["new_benchmark_description"])
                self.state["new_benchmarks"].append(analysis["new_benchmark_description"])
            except RuntimeError as exc:
                error_info = str(exc)
                self.state["error_history"].append({
                    "iteration": self.state["iteration"],
                    "type": "benchmark_generation",
                    "message": error_info
                })
                print(f"Benchmark generation failed: {error_info}")
                # Let the next iteration retry the benchmark generation from the same description
                continue

            self.save_state()

        print("Agent finished.")

if __name__ == "__main__":
    agent = ProfilingAgent()
    agent.iterate()
