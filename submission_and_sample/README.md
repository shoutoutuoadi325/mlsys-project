
# !!! NOTICE !!!

- The submission workflow is similar to Phase 1, but please note that the submission and output interfaces have changed, and your submission count is now **unlimited**:

  * `/submit` -> `/submit2`
  * `/outputs` -> `/outputs2`
  * `/submit_status` remains unchanged

- No `/submit-test` is provided. You can use the deepseek api key in `/submit2` now, or just use your own api key. 

- Your submission file name is also changed:
  * `/workspace/report.*` -> `/workspace/report2.*`
  * `/workspace/output_id.txt` -> `/workspace/output_id2.txt`

- You don't need to keep the project files in stage 1.

- The evaluation system will run your agent first. After your agent finishes or times out, the system will collect the generated:

```bash
/workspace/optimized_lora.cu
```

# Submit Your Report

You are required to submit a brief report describing the agent you have built, along with the **best output ID** (Required) produced by your agent. 

```bash
/workspace/report2.* # No restriction on file format
/workspace/output_id2.txt # Your output ID
# e.g. 
# > cat /workspace/output_id2.txt
# > a1234567890e212b00b339965f515d46
```


# Submit Your Agent

Please follow the instructions in `gpu_service_guide.md` to upload your code to the server at `10.176.37.31`. This server may also be used for development, but if other servers still have available GPUs, please do not use this one for development for the time being.


If you have prepared your code in your workspace, you can use `/submit2` to run it in the evaluation container.

## Before you submit

### 1. Files requirement

Please make sure your workspace contains:

* `run.sh`
* all code and files needed by `run.sh`

Your `run.sh` should be located at:

```bash
/workspace/run.sh
```

In general, our evaluation environment already includes common packages, including `openai` and the relevant profiling tools. If you need additional pip packages, you may install them in `run.sh` using the following command.

```bash
pip3 install <your_package> -i --default-timeout 0.3 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

Your agent must generate a cuda file and place it at 

```bash
/workspace/optimized_lora.cu
```

See the details about this output at https://memxlife.github.io/books/mlsys/project_phase2.html

Your agent may generate a report file and place it at 
```bash
/workspace/output.*
``` 
This file is used to evaluate your agent's reasoning process and optimization methodology.
The file format is not restricted.
Only one such file should be generated.
It may contain your agent's search process, candidate comparison, profiling results, benchmark logs, design choices, or final summary.


### 2. Model/API interface requirement

You may use **your own API key** for your agent. You can specify the API configuration in the `run.sh`, for example:

```bash
export API_KEY=<YOUR_API_KEY>
export OPENAI_API_KEY=<YOUR_API_KEY>
export BASE_MODEL=<YOUR_MODEL>
export BASE_URL=<YOUR_URL>

python your_code.py
```

If you do not specify your own API configuration, the evaluation environment will provide a default DeepSeek API interface through environment variables.

The following environment variables may be provided:

* `API_KEY`
* `BASE_MODEL`
* `BASE_URL`


A recommended coding style is:

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("API_KEY", ""),
    base_url=os.getenv("BASE_URL", "")
)

response = client.chat.completions.create(
    model=os.getenv("BASE_MODEL", ""),
    messages=[{"role": "user", "content": prompt}]
)
```

### 3. Rules

* Each student can have **at most one active task** at a time

  * if you already have a running `/start` environment, you cannot `/submit2`
  * if you already have a running `/submit2` task, you cannot start or submit again
* Each student can submit before 5/19 8 am and need to submit **at least one time** before 5/12 8 am.
* Each submission can run for **at most 30 minutes**. Submissions exceeding this limit may be terminated automatically.
* You may encounter API rate limiting or excessive request frequency sometimes, just wait for a short period before submitting again.



## Submit

```bash
# linux or mac
curl -X POST http://<server>:8080/submit2 \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 1 }'
```

Parameters:

* `id`: your student ID
* `gpu`:
  * `1` means submit with a GPU (needed)


Example response:

```json
{
  "ok": true,
  "user_id": "23210240000",
  "status": "running",
  "require_gpu": true,
  "gpu_id": 0,
  "output_file": "xxx",
  "submit_count": 0,
  "submit_limit": 2,
  "remaining_submit_count": 2
}
```

Please keep the returned `output_file`.
It is the identifier for checking your submit status.


### Check submit status

```bash
curl http://<server>:8080/submit_status/<output_file>
```

Possible status values include:

* `running`
* `succeeded`
* `failed`
* `killed`

Example response:

```json
{
  "ok": true,
  "output_file": "7f3d6d3b0d4f0b2f7a6d6d43b4b9fabc",
  "status": "succeeded",
  "gpu_id": 0,
  "started_at": 1713123456, 
  "finished_at": 1713123510, // null if not finished
  "submit_count": 1,
  "submit_limit": 2,
  "remaining_submit_count": 1
}
```

## View and download outputs

Open the following page in your browser:

```text
http://<server>:8080/outputs2
```

This page will show a simple list of all anonymous output files.
You can click any link to download the corresponding file directly.

Download your output file with your output ID.
Or you can check your output file Locally by start your container.

Your agent's standard output and standard error will be written to:
```bash
/workspace/results.log
````

## Minimal example workflow

```bash
# 1. upload your code to /workspace
# 2. make sure /workspace/run.sh exists
# 3. make sure your agent reads /target/target_spec.json
# 4. submit

curl -X POST http://<server>:8080/submit2 \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 1 }'

# 5. check status
curl http://<server>:8080/submit_status/<output_file>

# 6. open outputs page in browser
# http://<server>:8080/outputs2
```
