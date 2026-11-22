# vdvr — 用桌面键鼠玩 VR 模式的 VRChat，并伪造「全追」

> 目标：VRChat 以 **VR 模式** 启动（有头显、有双手、有全身 IK），但操作全部来自键盘鼠标；
> 同时通过 VRChat 官方 **OSC Trackers** 把合成出来的髋/胸/脚/膝/肘数据喂给游戏，
> 让游戏和别的玩家都认为你戴着全身追踪。

---

## 0. 最短路径

```powershell
# 1. 检查环境（Steam / SteamVR / VRChat / GlassVR / 端口 / 协议版本）
python tools/check_env.py

# 2. 桌面假 VR 模式：启用 GlassVR 虚拟 HMD/手柄
python tools/glassvr_mode.py vdvr
# 重启 SteamVR

# 3. 启动输入桥（会自动聚焦 Headset Window）
start.bat

# 4. 启动 VRChat（带 OSC 端口）
launch_vrchat_vr.bat

# 5. VRChat 里 Calibrate FBT 提示时按 F7
```

正常玩真实 VR 前：

```powershell
python tools/glassvr_mode.py normal
# 关闭 vdvr，重启 SteamVR，然后正常启动 VRChat
```

---

## 0.5 ARDY 动作回放（头/手 + 身体）

这个模式把鼠标输入源换成 ARDY 的 `motion.npz`：

- 头 / 左手 / 右手 → GlassVR UDP 9100 / 9102 / 9101（VRChat 6 点 IK 的头和手）
- hip / chest / feet / knees / elbows → VRChat OSC 9000
- 因此 **不需要 8 点 FBIK，也不需要改 VRChat 的 IK/身高设置**

```powershell
# 先停掉 start.bat / run.py（鼠标模拟器）；否则两边会同时写 GlassVR UDP
python -m vdvr.ardy E:\...\motion.npz --height 1.70 --profile standard --calibrate 5
```

- `--calibrate 5`：保持 T-pose + 双扳机 5 秒，方便 VRChat `Calibrate FBT`；完成后自动播放。
- 槽位与 `vdvr/body.py` 一致：`1=hip, 2=chest, 3/4=feet, 5/6=knees, 7/8=elbows`。
- 其他参数：`--loop`、`--fade-in`、`--return-seconds`、`--rate`、`--dry-run`。
- 需要 `numpy`（`pip install -r requirements-ardy.txt`）；ARDY 核心已 vendor 到
  `vdvr/ardy_core/`，不再依赖 `ardy-action-to-vrchat` 仓库。
- 播放期间手柄按键/菜单不工作（输入源被 ARDY 接管）；Ctrl+C 会平滑回默认姿势并退出。

---

## 0.6 自然语言实时生成

先停 `run.py`，保持 GlassVR/SteamVR/VRChat 运行：

```powershell
cd D:\ProgramProject\vrchat-just-desktop-use-vr
python -m vdvr.live --height 1.70 --profile standard
```

它会：

- 用 ARDY 的 python 环境启动 `vdvr/prompt_worker.py`（常驻，模型只加载一次）；
- 如果 7871 的 student encoder 没在跑，自动启动；
- 读入自然语言，生成 `motion.npz`，再通过 GlassVR + OSC 播放；
- 当前动作循环，直到你输入下一句。

```
prompt> a person waves with their right hand
prompt> a person walks forward and stops
prompt> a person sits down on the floor
```

- 首次加载约 20 s；之后每句约 1–4 s（取决于 GPU/步数）。
- `--calibrate 5`：第一段前保持 T-pose + 双扳机 5 秒，方便 Calibrate FBT。
- `--duration 4 --steps 10`：生成片段长度 / 扩散步数。
- `--dry-run`：只生成+重定向，不发送 GlassVR/OSC。
- 默认路径可用 `--infer-repo` / `--worker-python` 覆盖。
- 生成文件在 `ardy-0.5b-infer/outputs/live/`。
- ARDY 模型和 VRChat 共用 GPU；8 GB 显存紧张时先关掉其他 GPU 程序。

---

## 1. 结论（可行，且不需要任何追踪硬件）

VRChat 的 **OSC Trackers** 最多支持 8 个点（髋、胸、左右脚、左右膝、左右肘），
以世界坐标 + 欧拉角直接写入它原有的 FBT IK 系统。因此：

- **「全追」不需要 SteamVR tracker、基站、摄像头或 IMU**，只要有一个程序持续发送 OSC。
- 但 OSC tracker **只在 VR 模式下生效**；桌面模式不工作。
- 所以还需要让 VRChat 进入 VR 模式：SteamVR 必须提供一个 **虚拟 HMD + 两个虚拟 Index 手柄**。
- 本方案用开源虚拟驱动 **[GlassVR](https://github.com/DaniXmir/GlassVr)** 作为「虚拟硬件」，
  本仓库的 Python 程序负责：键鼠 → HMD/手柄姿态 + OSC tracker 合成 + OSC 输入。

```
 键盘/鼠标
    │  (全局轮询, 仅目标窗口前台时生效)
    ▼
 vdvr (Python)
    ├── UDP 9100/9101/9102 ──► GlassVR 驱动 ──► SteamVR ──► VRChat (VR 模式: 头显+双手)
    └── UDP 9000 ──► VRChat OSC
            ├── /tracking/trackers/{1..8}/position|rotation   (伪造 FBT)
            ├── /tracking/trackers/head/position|rotation     (空间对齐)
            └── /input/*                                      (跳跃/跑步/菜单/麦克风)
```

**效果**

- 鼠标 = 转头（HMD yaw/pitch）；WASD = 左摇杆移动；空格 = 跳跃；鼠标左右键 = 扳机/握持。
- 站立时双手固定在**身体空间**：转头不会带着手一起转，所以射线会扫过画面。
- 需要像鼠标一样独立瞄准 UI 时：**按住 Alt + 移动鼠标**转动右手激光，**左键**点击，松开 Alt 恢复转头。
- 站着不动时有呼吸/重心摆动；走路时双腿交替摆动、髋部起伏；蹲下时髋部下降、膝盖前屈。

---

## 2. 为什么是这套方案（方案对比）

| 方案 | 能否 VR 模式 | 键鼠控制 | 免硬件全追 | 结论 |
| --- | --- | --- | --- | --- |
| VRChat 桌面模式 | 否 | 原生 | 否（OSC tracker 桌面模式无效） | 不满足 |
| SteamVR `null` 驱动 | 是（静止 HMD） | 无头转/无手柄 | 可加 OSC | 不满足「完全控制」 |
| **虚拟 OpenVR 驱动 + OSC Trackers** | 是 | 是 | 是 | **本方案** |
| Driver4VR（商业） | 是 | 支持键鼠/虚拟手柄 | Puppeteer 可模拟 FBT | 可用但要付费、闭源 |
| 摄像头/手机 FBT | 是 | 仍需 VR 手柄 | 是 | 仍需头显，不符合「只用 desktop」 |

参考：VRChat 官方 OSC Trackers 文档明确说「hardware-agnostic」，支持 SlimeVR、Kinect、
普通摄像头、动捕服等，并提供了 Unity 发送示例。

---

## 3. 环境要求

- Windows 10/11
- Steam + **SteamVR**
- **VRChat**（Steam 版）
- **Python 3.10+**（纯标准库，无 pip 依赖）
- **GlassVR**：<https://github.com/DaniXmir/GlassVr/releases>
- 不需要头显、基站、tracker、手柄。

> 本机已验证：Steam 在 `E:\Steam`，SteamVR + VRChat 已安装，GlassVR v2.3 驱动已安装，
> `activateMultipleDrivers = true`，虚拟 HMD + 两个 Knuckles 手柄可被 SteamVR/VRChat 识别。

---

## 4. 安装与配置

### 4.1 安装 GlassVR 驱动

1. 从 Releases 下载压缩包，解压，运行 `main.exe` / `GlassVR.exe`。
2. **Driver** 标签页 → **install**（装到 `SteamVR\drivers\glassvrdriver`）。
3. 允许 SteamVR 同时加载多个驱动：

   ```powershell
   python tools/check_env.py --apply-drivers
   ```

   该命令会备份 `steamvr.vrsettings` 后把 `"activateMultipleDrivers"` 设为 `true`。

### 4.2 GlassVR 设备与端口

| 设备 | pos mode | rot mode | input mode | UDP 端口 |
| --- | --- | --- | --- | --- |
| HMD | `UDP` | `UDP` | — | **9100**（`hmd port`） |
| 右手柄 | `UDP` | `UDP` | `UDP`（Index `idx`） | **9101**（`cr port`） |
| 左手柄 | `UDP` | `UDP` | `UDP`（Index `idx`） | **9102**（`cl port`） |

补充：

- HMD 需要勾选 **enable hmd**；`display mode` 选 `virtual` 或扩展模式。
- 端口必须与 `config.json` 的 `glassvr.hmd_port / right_port / left_port` 一致。
- **把所有 offset 保持为 0**（local/world x/y/z、yaw/pitch/roll、playspace yaw）。
- 如果 GlassVR 自带键鼠/SDL 输入模块，对这三个设备**关闭**，由 vdvr 统一驱动。
- **GlassVR 版本协议差异**：v2.3 的 UDP 标签是 1 字节 `P`/`R`/`I`，v2.4+ 是 3 字节
  `pos`/`rot`/`idx`。`glassvr.protocol` 默认 `both`，同时兼容；确认版本后可改成
  `v23`/`v24` 把包量减半。
- 坐标系差异由 `vdvr/glassvr.py` 自动处理：VRChat/OSC 用 Unity（+Z 前、左手系），
  OpenVR/GlassVR 用（−Z 前、右手系），位置翻转 Z、四元数取 `(w, −x, −y, z)`。

### 4.3 启动 VRChat（VR 模式 + OSC）

首次登录建议先用**桌面模式**，避免 VR 登录页的激光/QR 不好点：

```powershell
launch_vrchat_desktop.bat
```

用普通鼠标登录（优先点 **Steam** 或邮箱密码，别一直点 QR；QR 轮询容易被 429 限流）。
登录成功后关闭 VRChat，再用：

```powershell
launch_vrchat_vr.bat
```

`launch_vrchat_vr.bat` 等价于：

```
VRChat.exe --osc=9000:127.0.0.1:9001
```

进入游戏后：

- **Action Menu → OSC → Enabled**
- **Quick Menu → Settings → Tracking & IK**：
  - `Allow FullBody Tracking FBT` → On
  - `FBT Locomotion Animation` → **Off**
- 如果桌面看不到画面：在 SteamVR 菜单里打开 **Display VR View**。

### 4.4 运行 vdvr

推荐顺序：**SteamVR → vdvr → VRChat**。

```powershell
python run.py
# 或双击 start.bat
```

启动后会**自动尝试把 Headset Window 切到前台**；如果 Windows 阻止切换，按 **F6**。
状态行显示 `active=Y` 才表示鼠标/键盘正在驱动游戏。
需要操作桌面时按 **F8** 释放鼠标。

### 4.5 FBT 校准（必须做一次）

1. VRChat 菜单 → **Calibrate FBT**。
2. 提示「站成 T-pose 并扣动双手扳机」时，按 **F7**。
3. vdvr 保持 3.5 秒 T-pose，并把左右扳机同时按住。
4. 如果身体偏高/偏低/前后偏移，用 **Adjust FBT** 微调，或调整 `--height`。

> 建议先用 `--profile minimal`（只发髋 + 双脚）完成第一次校准，稳定后再试 `standard` / `full`。

---

## 5. 键位

| 输入 | 作用 |
| --- | --- |
| 鼠标移动 | 转头（HMD yaw / pitch） |
| **Alt（按住）+ 鼠标** | **独立瞄准右手激光/UI 指针**（松开恢复转头） |
| 鼠标左键 / `E` | 右手扳机（交互 / 点击 UI） |
| 鼠标右键 / `F` | 右手握持（抓取 / 放下） |
| `Q` / `R` | 左手扳机 / 左手握持 |
| `W A S D` | 移动（左摇杆） |
| `Shift` | 跑步（OSC `/input/Run`） |
| `Space` | 跳跃（右手 A + OSC `/input/Jump`） |
| `Ctrl`（按住） / `C`（切换） | 蹲下 |
| `←` `→` | 右摇杆左右（Comfort Turning 时为快速转身） |
| `B` / `Esc` | 打开快捷菜单（右手 B） |
| `Tab` | SteamVR 系统键（呼出 SteamVR 面板） |
| `M` | 麦克风开关（OSC `/input/Voice`） |
| `F6` | 把 **Headset Window** 切到前台 |
| `F7` | FBT 校准姿态（T-pose + 双扳机，3.5 秒） |
| `F8` | 开/关输入抓取（释放/锁定鼠标） |
| `F9` | 视角 + 身体 + 指针全部回正 |
| `F10` | 紧急释放：停用抓取、所有轴归零、松开按键 |
| `Ctrl + Alt + Q` | 退出 |

> 鼠标视角是「无限旋转」：每帧把系统光标拉回目标窗口中心。
> 要操作桌面/登录窗口，先按 **F8**。

---

## 6. 调参

`config.json`（`python run.py --make-config config.json` 可重新生成）：

```jsonc
{
  "vrchat":  { "host": "127.0.0.1", "port": 9000, "enabled": true },
  "glassvr": { "host": "127.0.0.1", "enabled": true,
               "hmd_port": 9100, "left_port": 9102, "right_port": 9101,
               "protocol": "both" },
  "capture": { "processes": ["VRChat.exe", "vrcompositor.exe", "vrmonitor.exe"],
               "mouse_sensitivity": 0.045, "invert_y": false, "recenter_mouse": true },
  "body":    { "height": 1.70, "profile": "standard" },
  "input":   { "movement_via": "controller", "menu_via": "controller", "jump_via": "both",
               "pointer_key": "ALT", "pointer_sensitivity": 0.08 },
  "start":   { "enabled": true, "countdown": 5, "focus_game": true }
}
```

- `body.height`：真实身高，直接影响髋高、腿长、步幅；**校准前务必设对**。
- `body.profile`：`minimal`（髋+双脚）/ `standard`（+胸）/ `full`（8 点）。
- `capture.mouse_sensitivity`：度/像素；建议 0.03–0.06。
- `input.pointer_sensitivity`：Alt 指针模式的度/像素；建议 0.06–0.12。
- `input.jump_via`：A 在某些组合下不进 legacy 掩码，默认 `both` 用 OSC 兜底。

---

## 7. 正常 VR 与桌面模式切换（重要）

GlassVR 的虚拟 HMD/手柄默认是**启用**的。真实头显插上时，SteamVR 可能优先选中
`glassvr.GlassVRHmd`，导致真实头显用不了。正常玩真实 VR 前切换一下：

### 正常 VR

1. 关闭 vdvr（`python run.py` 窗口 / Ctrl+C）
2. 双击 **`normal_vr.bat`**（`enable hmd/cr/cl = false`）
3. 重启 SteamVR（菜单 → Quit，再启动）
4. 正常从 Steam 启动 VRChat / 使用真实头显

### 切回桌面假 VR

1. 双击 **`fake_vr.bat`**（重新启用虚拟 HMD/手柄、UDP 端口）
2. 重启 SteamVR
3. `start.bat`，再 `launch_vrchat_vr.bat`

`steamvr.activateMultipleDrivers = true` 可以一直保留，对正常 VR 无害；
`launch_vrchat_vr.bat` 只是多带 `--osc`，不会强制 VR 模式。

**完全移除**：关闭 SteamVR 后，在 GlassVR UI → Driver → **uninstall**，
或删除 `SteamVR\drivers\glassvrdriver`，并把 `activateMultipleDrivers` 改回 false。

---

## 8. 故障排查

先跑：

```powershell
python tools/check_env.py
```

| 现象 | 处理 |
| --- | --- |
| 登录页点不了 / 看不到射线 | 先用桌面模式登录：`launch_vrchat_desktop.bat`，用普通鼠标点 **Steam** 或邮箱密码（QR 429 时限流，等 10 分钟）。 |
| 射线不动 / 点不到按钮 | 按住 **Alt** 移动鼠标，看状态行 `ptr=(...)` 是否变化；按 **B** 打开快捷菜单后射线才显示；确认 `active=Y`。 |
| 鼠标不能转头 | 控制台状态必须是 `active=Y`；按 **F6** 聚焦 Headset Window；确认 `python run.py` 进程还在。 |
| 有 HMD/手柄但不动 | 99% 是 GlassVR 版本协议：`glassvr.protocol` 设为 `both`（默认）或对应 `v23`/`v24`。 |
| 进 VR 后没有手 | GlassVR 手柄设备未启用/端口不对；确认 `enable cr/cl`、`input mode=UDP`。 |
| 身体乱飞 / 穿地 | 身高不对、没做 FBT 校准、或 tracker 发太多。先用 `--profile minimal` + 重新校准。 |
| 腿不动 | `FBT Locomotion Animation` 没关。 |
| 真实头显被顶掉 | 关闭 vdvr，运行 `normal_vr.bat`，重启 SteamVR。 |

调试 OSC：

```powershell
python tools/osc_monitor.py --port 9000 --quiet   # vdvr 发了什么
python tools/osc_monitor.py --port 9001           # VRChat 回了什么
```

---

## 9. 已知限制 / 诚实说明

- **这是「合成」FBT，不是真追踪**：动作由身高、移动输入、头部朝向推出来，不会反映真实坐姿/抬腿。
- 手部没有独立位移（Alt 指针模式只在小范围内移动/旋转右手），没有手指追踪。
- 站立时身体/手不跟头转，是为了让射线扫过画面；移动时身体会跟随头。
- 本仓库的算法、OSC 协议、GlassVR 包结构（v2.3 + v2.4）、坐标转换已做 **27 个单元测试**
  + dry-run + UDP 抓包验证。
- **真机验证**：GlassVR v2.3 + SteamVR 2.18.4 + VRChat 2026.4.1p2。
  VRChat 日志确认 `Connected to glassvr(glassvr) : oculus_hmd`、
  `Connected controller: Knuckles EV3.0 - Left/Right`、`OSC enabled: True`；
  OpenVR 侧实时读到 HMD/双手姿态随 vdvr 变化；摇杆/扳机输入也在 SteamVR 中可见。
- 尚未验证：登录后实际完成 FBT 校准的画面（取决于账号/世界）。

---

## 10. 代码结构

```
run.py                   入口
start.bat                启动 vdvr
launch_vrchat_desktop.bat 桌面模式登录
launch_vrchat_vr.bat     VR 模式 + OSC 启动
normal_vr.bat            关闭虚拟 HMD/手柄（正常 VR）
fake_vr.bat              启用虚拟 HMD/手柄（桌面假 VR）
config.json              默认配置
vdvr/
  osc.py                 最小 OSC 1.0 编解码
  math3d.py              Unity 约定的向量/四元数（Z-X-Y 欧拉序）
  body.py                身体模型：8 点 tracker 合成（呼吸/走路/蹲/跳）
  glassvr.py             GlassVR UDP 协议（v2.3 单字节 / v2.4 三字节）+ OpenVR 坐标转换
  vrchat.py              VRChat OSC 发送（tracker/head/input，含 keepalive）
  input_win.py           Win32 键鼠轮询、鼠标无限旋转、窗口聚焦
  config.py              配置加载
  app.py                 主循环 / 键位 / Alt 指针 / 校准姿态
tools/
  check_env.py           Steam/SteamVR/VRChat/GlassVR/端口/协议检查
  glassvr_mode.py        normal / vdvr 模式切换
  osc_monitor.py         OSC 抓包打印
tests/                   27 个单元测试
```

运行测试：

```powershell
python -m unittest discover -s tests -t . -v
```

---

## 11. 参考

- VRChat OSC Trackers（官方）：<https://docs.vrchat.com/docs/osc-trackers>
- VRChat OSC 作为输入控制器（官方）：<https://docs.vrchat.com/docs/osc-as-input-controller>
- VRChat OSC Wiki：<https://wiki.vrchat.com/wiki/OSC>
- VRChat Settings（FBT/Height/Interaction Mode）：<https://wiki.vrchat.com/wiki/Settings>
- GlassVR（虚拟 HMD/手柄/tracker 驱动）：<https://github.com/DaniXmir/GlassVr>
- OpenVR Driver API（pose/input/aim）：<https://github.com/ValveSoftware/openvr/blob/master/docs/Driver_API_Documentation.md>
- OpenVR-driver-for-DIY（备选）：<https://github.com/r57zone/OpenVR-driver-for-DIY>
- VR-Emulator-Driver（备选）：<https://github.com/ar-zadeh/VR-Emulator-Driver>
- SteamVR null 驱动说明：<https://github.com/username223/SteamVRNoHeadset>
- Driver4VR（商业方案，含 Puppeteer FBT）：<https://support.driver4vr.com/>
