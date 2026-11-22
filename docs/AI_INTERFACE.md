# vdvr AI 表演接口

面向 AI / Agent 的本地 HTTP 接口：用自然语言让 ARDY 生成一段动作，在 VRChat 里**播放一次**，然后平滑回到默认姿势。**不循环**。

- 头 / 左手 / 右手：ARDY → GlassVR 虚拟 HMD/手柄（UDP 9100/9102/9101）
- hip / chest / feet / knees / elbows：ARDY → VRChat OSC Trackers（UDP 9000）
- VRChat 的 6 点 IK 负责把头和手接上；**不需要 8 点 FBIK**
- 不修改 VRChat，不注入，不做内存修改

机器可读的 OpenAPI 描述：[`ai_openapi.yaml`](./ai_openapi.yaml)

---

## 1. 启动

前置条件：

1. GlassVR / SteamVR / VRChat 已启动，VRChat 处于 **VR 模式**且已开启 OSC。
2. 先校准一次 FBT（可用 `start.bat` / `run.py` 的 F7，或用一次
   `python -m vdvr.live --calibrate 5`），然后**停掉 `run.py`**。
   API 和 `run.py` 都会写 GlassVR 的 UDP 端口，不能同时运行。
3. 可选：先 `python tools/check_env.py` 确认端口/协议。

启动 API：

```powershell
cd D:\ProgramProject\vrchat-just-desktop-use-vr
python -m vdvr.live --api --api-port 7878 --height 1.70 --profile standard
```

- 监听地址默认 `127.0.0.1:7878`（仅本机，无鉴权）。
- 进程一启动 HTTP 就已经在监听；ARDY encoder / 模型在后台加载。
- 首次加载约 15–25 s。加载完成前：
  - `GET /health` 返回 `ready: false`
  - `POST /perform` 返回 `503`
- 加载完成后日志出现 `[ai] ready`。

停止：`Ctrl+C`。进程退出前会停止播放并关闭 worker/encoder。

---

## 2. 一次表演的语义

`POST /perform` 是**同步**接口：

1. 用 prompt 生成一段 `motion.npz`（encoder + ARDY denoiser）；
2. 用 vdvr 重定向成头/手/身体目标；
3. 通过 GlassVR + OSC **播放一次**（不循环）；
4. 在 `return_seconds` 内平滑回到默认姿势；
5. 播放结束（含回默认）后返回 JSON。

所以一次请求的耗时约为：

```
generate (1–4 s) + clip_seconds + return_seconds + ~0.5 s
```

AI 应该等一次 `/perform` 返回后再发下一条；不要并发发送（除非有意用 `wait: false` 并自己处理 409）。

- **不循环**：每个 prompt 只表演一次。
- **每次结束在默认姿势**：下一条从默认姿势开始。
- **同一时间只跑一个 performance**：默认 `wait: true` 排队；`wait: false` 时忙则 409。
- **`/stop`** 只打断当前播放（让它提前回默认），不取消正在进行的生成。

---

## 3. 接口

Base URL：`http://127.0.0.1:7878`

### `GET /health`

```json
{"ok": true, "ready": true, "state": "idle"}
```

`state`: `loading | idle | generating | playing | error`

### `GET /status`

```json
{
  "ok": true,
  "ready": true,
  "state": "playing",
  "error": null,
  "prompt": "a person waves with their right hand",
  "motion": "E:\\...\\outputs\\live\\motion_...npz"
}
```

- `prompt` / `motion` 是最近一次表演；空闲时也保留。
- `error` 是 worker 初始化失败原因（如有）。

### `POST /perform`

请求体：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | string | —（必填） | 自然语言动作描述；英文效果最好 |
| `duration` | float | `4.0` | 生成片段长度（秒），1–10 |
| `steps` | int | `10` | 扩散步数，1–50；越大越慢/越细 |
| `seed` | int | `0` | 随机种子；同 prompt+seed 可复现 |
| `return_seconds` | float | `1.2` | 播放完回默认姿势的时间（秒），0–10 |
| `wait` | bool | `true` | 忙时是否排队等待 |

成功 `200`：

```json
{
  "ok": true,
  "prompt": "a person waves with their right hand",
  "motion": "E:\\...\\outputs\\live\\motion_...npz",
  "frames": 120,
  "encoder": "service",
  "timings": {"encode_s": 0.202, "generate_s": 1.276},
  "clip_seconds": 4.0,
  "return_seconds": 1.2
}
```

错误：

| HTTP | 含义 |
|---|---|
| `400` | prompt 为空 / JSON 或字段类型错误 |
| `409` | 另一个表演正在运行，且 `wait: false` |
| `503` | ARDY 模型/encoder 还没就绪 |
| `500` | 生成或重定向失败（看 API 进程日志） |

### `POST /stop`

```json
{"ok": true, "state": "playing"}
```

请求当前播放提前回默认姿势。正在生成时调用它不会取消生成；等 `/perform` 返回即可。

---

## 4. 示例

### curl（Windows 用 `curl.exe`）

```powershell
curl.exe http://127.0.0.1:7878/health
curl.exe http://127.0.0.1:7878/status

curl.exe -X POST http://127.0.0.1:7878/perform `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"a person waves with their right hand\",\"duration\":3,\"steps\":10}"

curl.exe -X POST http://127.0.0.1:7878/stop
```

### Python（标准库）

```python
import json
import time
import urllib.request

BASE = "http://127.0.0.1:7878"

def perform(prompt: str, *, duration: float = 4.0, steps: int = 10, seed: int = 0,
            return_seconds: float = 1.2, timeout: float = 60.0) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "duration": duration,
        "steps": steps,
        "seed": seed,
        "return_seconds": return_seconds,
    }).encode("utf-8")
    request = urllib.request.Request(
        BASE + "/perform", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

# 等模型就绪
while not json.load(urllib.request.urlopen(BASE + "/health"))["ready"]:
    time.sleep(1.0)

for line in ["a person waves with their right hand",
             "a person walks forward and stops",
             "a person sits down on the floor"]:
    result = perform(line)
    print(result["prompt"], result["frames"], result["timings"])
```

### 一个 AI 表演脚本

```python
prompts = llm_plan_performance()          # 你的 LLM 生成的动作列表
for prompt in prompts:
    result = perform(prompt, duration=4.0)
    if not result.get("ok"):
        handle_error(result)
    # 这里已经回到默认姿势，可以继续下一段
```

---

## 5. 并发与排队

- `ThreadingHTTPServer`；`GET /status`、`POST /stop` 在表演期间仍然可用。
- `perform` 用一把锁串行化：
  - `wait: true`（默认）：第二个请求排队，直到当前表演结束再生成；
  - `wait: false`：忙时立刻 `409`。
- 需要打断当前表演：先 `POST /stop`，再发新的 `/perform`。
- AI 客户端超时请设置为 `>= duration + return_seconds + 20 s`（默认 60 s 足够）。

---

## 6. 延迟参考

RTX 4060 Laptop / VRChat 同时运行：

| 阶段 | 时间 |
|---|---|
| encoder（warm service） | 0.2–0.4 s |
| ARDY generate（4 s / 10 steps） | 1–4 s |
| 播放 | `clip_seconds` |
| 回默认 | `return_seconds` |
| 首次模型加载 | 15–25 s |

显存紧张时先关闭其他 GPU 程序；模型和 VRChat 共用 GPU。

---

## 7. 约束

- VRChat 必须在 **VR 模式**；桌面的普通模式不会使用 OSC trackers。
- 头/双手由 GlassVR 虚拟设备驱动；身体由 OSC trackers。VRChat 的 6 点 IK
  负责手臂；不需要胸/膝/肘的 8 点 FBIK。
- 如果 avatar 的 rig 有问题（例如日志里的 `FBIK chain length is 0`），
  手臂/脊柱的表现会受 VRChat 自身 IK 限制。
- 接口仅绑定 `127.0.0.1`，无鉴权；不要暴露到公网。
- 生成文件在 `E:\...\ardy-0.5b-infer\outputs\live\`，可用于复盘/回放。
- `run.py`（鼠标模拟器）必须停止，否则两边同时写 GlassVR UDP。
- 校准只需在启动 API 前做一次；API 本身不进入 Calibrate FBT。

---

## 8. 排错

| 现象 | 检查 |
|---|---|
| `/health` 一直 `ready:false` | 看 API 进程日志的 `[ardy]`/`[live]`；模型/encoder 是否在加载；GPU/显存 |
| `/perform` 503 | 模型还没 ready；等 `/health` |
| `/perform` 409 | 上一段还在播放；等它返回，或 `/stop` |
| 500 | 看日志里的生成/重定向异常；`motion` 文件是否存在 |
| 头/手不动 | GlassVR 是否在跑、端口 9100/9101/9102 是否正确、`run.py` 是否已停 |
| 腿不动 | VRChat OSC 是否开启、`--profile` 是否至少 `minimal` |
| 角色整体偏移/漂浮 | 先做一次 FBT 校准；检查身高（`--height`）与 VRChat 身高 |

---

## 9. 相关文件

- `vdvr/prompt_worker.py`：ARDY 环境内的常驻 text→motion worker（JSON lines）。
- `vdvr/live.py`：live 控制器（console 模式 + `--api` 模式）。
- `vdvr/ai.py`：`PerformanceController` / HTTP server。
- `vdvr/ardy.py`：GlassVR + OSC 的一次播放器。
- `docs/ai_openapi.yaml`：机器可读接口定义。
