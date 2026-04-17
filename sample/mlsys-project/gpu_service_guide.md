
# GPU Service Guide

To support students in completing the course project, we provide GPU resources distributed across the following four servers:

```bash
10.176.34.113
10.176.34.117
10.176.34.118
10.176.37.31
````

There are 32 GPUs in total. Since resources are limited, please strictly follow the rules below to avoid occupying resources needed by others:

1. Each user can occupy **only one GPU at a time**. We enforce this rule within a single server. Cross-server enforcement is not implemented, so please follow this rule voluntarily.
2. **Release your resources immediately when not in use.**
3. **Do NOT use the server resources for purposes unrelated to course experiments.**

Below are the usage instructions. `<server>` refers to any of the four hosts listed above.

---

## 1. Initialize Your Account

Before first use, upload your SSH public key. If you are not familiar with SSH keys, you can generate one using:

```bash
ssh-keygen
```

### Example:

```bash
# Linux or macOS
curl -X POST http://<server>:8080/init \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "ssh_pub_key": "ssh-ed25519 AAAA... alice@laptop" }'

# Windows PowerShell (not cmd)
Invoke-RestMethod -Method POST `
  -Uri "http://<server>:8080/init" `
  -ContentType "application/json" `
  -Body '{ "id": "23210240000", "ssh_pub_key": "ssh-ed25519 AAAA... alice@laptop" }'
```

### Parameters:

* `id`: your student ID
* `ssh_pub_key`: your SSH public key (**copy the full content from your public key file**)

You only need to do this once per server.
If you uploaded the wrong key, simply run `/init` again.
If you switch to a different server, you must initialize again.

---

## 2. Check Available GPUs

To check GPU availability on a server:

```bash
# Linux or macOS
curl http://<server>:8080/list

# Windows PowerShell
Invoke-RestMethod -Method GET `
  -Uri "http://<server>:8080/list"
```

### Example Response:

```json
{
  "ok": true,
  "total_gpu_count": 8,
  "free_gpu_count": 5,
  "used_gpu_count": 3
}
```

In most cases, you only need to check `free_gpu_count`.

---

## 3. Start an Environment

If there is an available GPU, you can start an environment:

### With GPU:

```bash
curl -X POST http://<server>:8080/start \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 1 }'
```

### Without GPU (for file access, etc.):

```bash
curl -X POST http://<server>:8080/start \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 0 }'
```

### Parameters:

* `id`: your user ID
* `gpu`:

  * `1` → request a GPU
  * `0` → no GPU needed

---

### Returned Information

If successful, you will receive:

#### 1. SSH Connection

```bash
ssh root@<server> -p <ssh_port> -i <your_private_key_path>
```

* You will log in as **root**
* `<your_private_key_path>` should be the private key PATH of the public key you submit before.
* **IMPORTANT: Save the `ssh_port`**, or you will not be able to reconnect

---

#### 2. User Ports

To support agent systems, we expose 3 ports:

| Container Port | Host Port |
| -------------- | --------- |
| 8080           | mapped    |
| 8081           | mapped    |
| 8082           | mapped    |

Example:

```json
{
  "ok": true,
  "user_id": "23210240000",
  "require_gpu": true,
  "gpu_id": 0,
  "ssh_port": 40221,
  "user_ports": [
    {"container_port": 8080, "host_port": 40222},
    {"container_port": 8081, "host_port": 40223},
    {"container_port": 8082, "host_port": 40224}
  ]
}
```

Use this information to access your environment.

### Workspace
`/workspace` is a persistent mounted directory tied to your ID, but it is not shared across different servers.

---

## 4. Stop Your Environment

When finished, **release your resources immediately**:

```bash
curl -X POST http://<server>:8080/finish \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000" }'
```

⚠️ **You MUST release the resources after use**

⚠️ **You MUST release the resources after use**

⚠️ **You MUST release the resources after use**

---

## Common Situations

### 1. Not initialized

Run `/init` first, then retry `/start`.

---

### 2. No GPU available

* No free GPU currently
* Try later
* Or start without GPU (`"gpu": 0`)

---

### 3. Environment already exists

* You already have a running environment
* Stop it first using `/finish`, then restart

---

### 4. GPU is shared unexpectedly

If you find other processes using your GPU:

* These servers may be shared with higher-priority users
* Your GPU may be preempted
* Recommended action:

  * Release your current environment
  * Start a new one on another server

---

## Full Example

```bash
# 1. Initialize
curl -X POST http://<server>:8080/init \
  -H "Content-Type: application/json" \
  -d '{"id":"23210240000","ssh_pub_key":"ssh-ed25519 AAAA... alice@laptop"}'

# 2. Check resources
curl http://<server>:8080/list

# 3. Start with GPU
curl -X POST http://<server>:8080/start \
  -H "Content-Type: application/json" \
  -d '{"id":"23210240000","gpu":1}'

# 4. Connect using ssh_port and user_ports

# 5. Stop environment
curl -X POST http://<server>:8080/finish \
  -H "Content-Type: application/json" \
  -d '{"id":"23210240000"}'
```

