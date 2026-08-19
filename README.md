# MiniMax H3 on 4x NVIDIA DGX Spark (ComfyUI)

在 **4 台 NVIDIA DGX Spark**（GB10, Blackwell, 121GB 统一内存, aarch64）上通过 **ComfyUI** 部署 **MiniMax H3 视频生成**（fl2v / ref2v / t2v 等）的完整实战记录。

覆盖：单台首次部署、**完全离线跨机迁移**（高速网口直连）、第四台密码恢复、Qwen3.8-27B 复用到部署 LLM 的显存冲突，以及一路踩过的坑。

---

## 1. 环境总览

**4 台机器（同硬件同配置）**

| 项目 | 值 |
|---|---|
| 硬件 | NVIDIA DGX Spark x4（GB10 Superchip） |
| IP | 10.0.0.XXX / .7 / .8 / .9 |
| OS | Ubuntu 24.04（aarch64），DGX OS 7.2.3 |
| GPU | Blackwell，sm_120，CUDA 13 |
| 内存 | 121GB 统一内存（~124GB VRAM 视角），273GB/s 带宽 |
| 推理框架 | ComfyUI **v0.30.0**，Python 3.12 venv，PyTorch 2.13.0+cu130 |
| 服务端口 | **8188**（每台一个实例） |

**ComfyUI 访问**：`http://10.0.0.XXX:8188` / `:4.7:8188` / `:4.8:8188` / `:4.9:8188`

**启动脚本**：`/home/nvidia/ComfyUI/start_comfyui.sh`（`source venv/bin/activate && python3 main.py --listen 0.0.0.0 --port 8188`）

---

## 2. 模型：MiniMax H3（8 个文件，~73GB）

| 文件 | 大小 | 用途 |
|---|---|---|
| UNET fl2v | 20GB | 图生视频主干（first/last frame） |
| UNET ref2v | 20GB | 参考图视频主干 |
| Qwen3-VL 文本编码器 | 26GB | 文本/图像语义编码 |
| Video VAE | 4.9GB | 视频编解码 |
| Audio VAE | 578MB | 音频编解码 |
| LoRA（4 步加速） | 744MB | 加速采样 |
| CLIP-L | 235MB | 文本编码 |
| CLIP Vision | 817MB | 图像编码 |

**工作流**：`input/workflow_minimax_h3_t2v.json`
- t2v 文生视频：864×480，124 帧 ≈ 5s，24fps，25 步
- 支持全部任务类型：`t2v` / `i2v` / `fl2v` / `r2v` / `v2v` / `rv2v`
- 单条 prompt 执行耗时：~19-21 分钟（含模型加载）
- 模型加载后内存占用 **~90%**（121GB 内存被 73GB 模型占满属正常）

---

## 3. 集群拓扑：两个网段

```
10.0.0.XXX/24  (管理网, 普通千兆)
     .6 ───────── .9  (与互联网, 4.9 有外网)
     .7 ───────── .8

10.0.1.XXX/24  (高速直连网, 网口直连线缆, MTU 9000 jumbo frames)
     .6 <───> .7 / .8 / .9
```

高速直连用于传输大模型（rsync 实测 **350-460MB/s**），管理网做日常 ssh。

- 4.6 ↔ 4.9 的高速口是 nmcli 持久配置（MTU 9000）
- 只有 **4.9 有外网**（DNS+HTTPS 通，ICMP 被墙）→ 下载依赖全走 4.9，再离线分发

---

## 4. 部署路径

### 4.1 第一步：4.6 在线完整部署（源头机）

1. 装 ComfyUI（v0.30.0）+ 创建 venv + 装 PyTorch 2.13+cu130
   - pip 走国内镜像加速：`-i https://mirrors.cloud.tencent.com/pypi/simple --break-system-packages`
2. 下载 MiniMax H3 全套模型到 `models/`（约 73GB）
3. `start_comfyui.sh` 启动，验证 8188 HTTP 200 + 跑通 t2v 工作流
4. 此为**模板机**，后续 3 台全部从它迁移

### 4.2 第二步到四台：完全离线迁移（不走外网）

核心思路：**既然 4.6 已经能跑，其余机器完全复制它**，不重新下载任何东西。

```bash
# 1. 高速网口挂载（MTU 9000 已配好）
# 2. venv 整体 tar 过去（关键！）→ 相同绝对路径 /home/nvidia/ComfyUI/venv
tar -cf - -C /home/nvidia/ComfyUI venv | ssh <user>@<node7> tar -xf - -C /home/nvidia/ComfyUI

# 3. 源码 rsync（排除 venv / models / output）
rsync -av --no-whole-file --exclude venv --exclude models --exclude output \
  /home/nvidia/ComfyUI/ <user>@<node7>:/home/nvidia/ComfyUI/

# 4. 模型 rsync（73GB 走高速口 ~3 分钟）
rsync -av /home/nvidia/ComfyUI/models/ <user>@<node7>:/home/nvidia/ComfyUI/models/
```

**为什么 venv 用 tar 而不是 rsync**：venv 内有符号链接和相对路径，相同安装路径才能保证 Python 环境完整性。

**难点**：4.9 忘记系统密码 → **GRUB `init=/bin/bash` 恢复**，未重装系统：
```bash
# 启动时进 GRUB edit 内核参数加 init=/bin/bash
# rm 掉 passwd/shadow 并重置，账号合并为 【已隐藏】
```

迁移后第 4 台（.9）也部署 ComfyUI+MiniMax H3 成功，8188 HTTP 200 验证通过。

---

## 5. 离线迁移踩坑清单（重点）

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | **rsync 跳过同 mtime 文件** | `comfy/ldm/models/autoencoder.py` 被跳过 → 启动报 `ModuleNotFoundError: comfy.ldm.models` | 重跑 `rsync --ignore-existing` 强制重传 |
| 2 | **源头残留嵌套重复目录** | `diffusion_models/diffusion_models`、`text_encoders/text_encoders` 重复嵌套（~68GB）+ `._` AppleDouble 文件 | 目标机清理一层嵌套 + `find -name '._*' -delete` |
| 3 | **sudo 远程执行** | 需要权威凭证 | `sudo <cmd>` |
| 4 | **启动验证时机** | 刚起进程就 curl 会失败 | 等 10-20s 再 curl 8188 |
| 5 | **无开机自启** | 机器重启后 8188 不再自动起来 | 手动 `nohup bash /home/nvidia/ComfyUI/start_comfyui.sh > /tmp/comfy_4.8.log 2>&1 &` |
| 6 | **apt 源不可达** | `ports.ubuntu.com` 在 Spark 上连不通（只通 HTTPS） | 系统包走 pip：`-i https://mirrors.aliyun.com/pypi/simple/` |

第 3 台清理后 `models/` 最终体积 **72GB / 8 个文件**，HTTP 200 验证通过。

---

## 6. 第四台的二次演进：Qwen3.8-27B LLM（同一块板子上）

4.9 不止跑 ComfyUI——后续又部署了 Qwen3.8-27B（见兄弟仓库 [qwen38-dgx-spark-deployment](https://github.com/ShiningMeUp/qwen38-dgx-spark-deployment)）。

**模型下载**：走 ModelScope（不是 HF）：
```bash
modelscope download --model Qwen/Qwen3.8-27B --local_dir /home/nvidia/model
```
51.77GB / 18 个 BF16 safetensors 分片，架构 `Qwen3_5ForConditionalGeneration`（多模态：图像+视频）。

**vLLM 部署**（OpenAI 兼容 API :8000 + Gradio UI :8080）：
```bash
cd /home/nvidia/llm && PATH=./venv/bin:$PATH nohup ./venv/bin/vllm serve /home/nvidia/model \
  --port 8000 --host 0.0.0.0 > vllm_serve3.log 2>&1 &
```
- vLLM 0.27.1 + torch 2.13/CUDA13，aarch64 官方支持
- 实测多轮对话、流式输出正常

**两个大坑：**
1. **flashinfer JIT 找不到 ninja** → 启动必须 `PATH=venv/bin:$PATH`（ninja 在 venv/bin 里，默认 PATH 找不到）→ 否则 `FileNotFoundError: 'ninja'` 初始化失败
2. **显存冲突** → ComfyUI+MiniMax H3 加载后占 ~90% 内存，27B BF16（~52GB+KV）无法共存 → **现在彻底改成独立服务方案：用 LLM 时停 ComfyUI**。后续若共存需换 GGUF Q4（~17GB，即 ollama 方案）

---

## 7. 现在这台集群的状态（2026-08-19）

| 机器 | 服务 | 端口 | 状态 |
|---|---|---|---|
| 4.6 / 4.7 / 4.8 | ComfyUI + MiniMax H3 | 8188 | 视频生成 |
| 4.9 | ComfyUI + MiniMax H3 | 8188 | 视频生成 |
| 4.9 | vLLM BF16 27B + Gradio | 8000/8080 | LLM（需停 ComfyUI） |
| 4.9 | Ollama + qwen3.8:27b-q4_K_M | 11434 | LLM（41GB，可共存） |

4.9 现在的资源规划核心原则：**四个"大内存服务"按需停启，一次只跑一个**。

---

## 8. 快速参考

```bash
# 启动任一台的 ComfyUI
ssh <user>@<node6>
cd /home/nvidia/ComfyUI && nohup bash start_comfyui.sh > /tmp/comfy.log 2>&1 &

# 验证
curl -s http://127.0.0.1:8188/ -o /dev/null -w "%{http_code}\n"

# 远程 sudo
sudo <cmd>

# 清 AppleDouble
find /home/nvidia/ComfyUI/models -name '._*' -delete
```

---

## License

MIT
