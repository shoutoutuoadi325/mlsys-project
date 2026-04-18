

This is a very rudimentary agent implementation. It does not provide modularization between the agent and the tools, and the implementations of its various functions are highly coupled. It is provided for reference only. 

# !!! NOTICE !!!
- A new server with access to the external internet (such as ChatGPT) has been added: `10.176.34.113`

- The new server supports **preemptive scheduling** with the following priority order:

`/submit > /submit-test > /start`

If there are no idle resources available, a higher-priority request may directly terminate a lower-priority task that started earlier.

If you start a CPU-only container without using a GPU and only want to access your files, this operation does not participate in preemptive scheduling, and the container will not be preempted.

- The `/start` interface now returns a `mission_id`.

After starting a task, you should use `/submit_status` with the `mission_id` to check the SSH port assigned to the task.


- Each container started by `/start` can run for **at most 10 hours**.

After 10 hours, the container will be automatically terminated by the system.


# Submit Your Agent

Please follow the instructions in `gpu_service_guide.md` to upload your code to the server at `10.176.37.34`. This server may also be used for development, but if other servers still have available GPUs, please do not use this one for development for the time being.


If you have prepared your code in your workspace, you can use `/submit` to run it in the evaluation container.

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

Your agent should read the evaluation target specification from:

```bash
/target/target_spec.json
```

Your agent should generate a report file and place it at 
```bash
/workspace/output.*
``` 
The file format is not restricted, but **only one** such file should be generated.

### 2. Model/API interface requirement

Your agent should expose the model interface through environment variables, so that we can inject the model and API credentials used in evaluation.

The following environment variables may be provided:

* `API_KEY`
* `BASE_MODEL`
* `BASE_URL`

If you use your own API key, you do not need to keep these interfaces, but this is **not recommended**. The evaluation model `GPT-5.4` is already one of the strongest coding models available.

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

  * if you already have a running `/start` environment, you cannot `/submit`
  * if you already have a running `/submit` task, you cannot start or submit again
* Each student can submit **at most two times** before 4/21 8 am.
* Each submission can run for **at most 30 minutes**. Submissions exceeding this limit may be terminated automatically.
* You may encounter API rate limiting or excessive request frequency sometimes, just wait for a short period before submitting again.



## Submit

```bash
# linux or mac
curl -X POST http://<server>:8080/submit \
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

### Submit for test

This submission entry is for testing purposes only. There is no limit on the number of submissions, and it uses the DeepSeek API. However, it also has an overall API token limit. So, please still use it sparingly.

```bash
# linux or mac
curl -X POST http://<server>:8080/submit-test \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 1 }'
```



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
http://<server>:8080/outputs
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

curl -X POST http://<server>:8080/submit \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 1 }'

# 5. check status
curl http://<server>:8080/submit_status/<output_file>

# 6. open outputs page in browser
# http://<server>:8080/outputs
```

# Submit Your Report

You are required to submit a brief report describing the agent you have built, along with one **output ID** produced by your agent that you consider representative of its better performance. 

```bash
/workspace/report.* # No restriction on file format
/workspace/output_id.txt # Your output ID
```
