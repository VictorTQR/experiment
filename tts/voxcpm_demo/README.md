# VoxCPM Demo

基于 [VoxCPM 2](https://voxcpm.readthedocs.io/zh-cn/latest/) 的声音合成与克隆示例，提供两个即插即用模块：

| 文件 | 定位 | 适用场景 |
|---|---|---|
| [`demo.py`](./demo.py) | 精简版 | 只需三种克隆模式 + 长文本 + 流式 |
| [`voxcpm_full.py`](./voxcpm_full.py) | 全功能版 | 在精简版基础上增加基础 TTS、音素输入、方言合成、非语言标签常量 |

两个文件均为 **单文件即插即用**：既是 Python 模块，也是命令行工具。

---

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [模块 API](#模块-api)
  - [demo.py — VoxCPMCloner](#demopy--voxcpmcloner)
  - [voxcpm_full.py — VoxCPMCloner](#voxcpm_fullpy--voxcpmcloner)
  - [GenerateConfig 生成参数](#generateconfig-生成参数)
  - [常量](#常量)
- [CLI 用法](#cli-用法)
- [官方文档对照](#官方文档对照)
- [常见问题](#常见问题)

---

## 安装

### 1. 安装依赖

```bash
pip install voxcpm soundfile modelscope
```

### 2. 下载模型

运行 [`down_model.py`](./down_model.py) 通过 ModelScope 下载 VoxCPM2 权重：

```bash
python down_model.py
```

默认保存到 `~/.cache/modelscope/hub/models/OpenBMB/VoxCPM2`。

### 3. 模型路径解析

模块按以下优先级自动查找模型，无需手动指定：

1. 环境变量 `VOXCPM_MODEL_PATH`
2. ModelScope 默认缓存 `~/.cache/modelscope/hub/models/OpenBMB/VoxCPM2`
3. 模型 ID `openbmb/VoxCPM2`（首次使用时自动下载）

如需指定自定义路径：

```python
# 模块方式
cloner = VoxCPMCloner(model_path="/path/to/VoxCPM2")

# CLI 方式
python demo.py --model-path /path/to/VoxCPM2 --text "你好" -o out.wav

# 或通过环境变量
export VOXCPM_MODEL_PATH=/path/to/VoxCPM2
```

---

## 快速开始

### 模块方式

```python
from voxcpm_full import VoxCPMCloner

cloner = VoxCPMCloner()  # 自动检测模型路径

# 基础 TTS
wav = cloner.basic_tts("你好，世界！")

# 声音设计：用文字描述音色
wav = cloner.voice_design("你好", control="年轻女性，温柔甜美")

# 可控克隆：参考音频 + 可选风格
wav = cloner.clone("你好", reference_wav="speaker.wav")
wav = cloner.clone("你好", reference_wav="speaker.wav", control="cheerful, faster")

# 高保真克隆：参考音频 + 逐字转写
wav = cloner.hifi_clone(
    text="你好",
    prompt_wav="speaker.wav",
    prompt_text="这是 speaker.wav 的逐字转写",  # 建议用 ASR 获取
)

# 保存
cloner.save(wav, "out.wav")
```

### CLI 方式

```bash
# 基础 TTS
python voxcpm_full.py --text "你好，世界！" -o out.wav

# 声音设计
python voxcpm_full.py --text "你好" --control "年轻女性，温柔甜美" -o out.wav

# 可控克隆
python voxcpm_full.py --text "你好" --reference speaker.wav -o out.wav

# 高保真克隆
python voxcpm_full.py --text "你好" --prompt-wav speaker.wav --prompt-text "转写" -o out.wav

# 长文本分段合成
python voxcpm_full.py --text "很长的文本..." --reference speaker.wav --long -o out.wav

# 流式生成
python voxcpm_full.py --text "你好" --reference speaker.wav --stream -o out.wav
```

---

## 模块 API

### `demo.py` — VoxCPMCloner

精简版，三种克隆模式 + 长文本 + 流式。

#### 构造函数

```python
VoxCPMCloner(
    model_path: str | None = None,    # 模型路径或 ID，None 时自动检测
    load_denoiser: bool = False,      # 是否加载降噪器（denoise=True 时需要）
    device: str | None = None,        # 设备（如 'cuda'/'cpu'），None 自动选择
    lazy: bool = False,               # True 时首次调用才加载模型
)
```

#### 方法

| 方法 | 模式 | 必填参数 | 可选参数 | 说明 |
|---|---|---|---|---|
| `voice_design(text, control=None, config=None)` | 声音设计 | `text` | `control`、`config` | 用文字描述生成全新音色 |
| `clone(text, reference_wav, control=None, config=None)` | 可控克隆 | `text`、`reference_wav` | `control`、`config` | 参考音频提供音色，可选风格控制 |
| `hifi_clone(text, prompt_wav, prompt_text, reference_wav=None, config=None)` | 高保真克隆 | `text`、`prompt_wav`、`prompt_text` | `reference_wav`、`config` | 参考音频 + 逐字转写，相似度最高 |
| `synthesize_long(segments, reference_wav=None, control=None, config=None)` | 长文本 | `segments` | `reference_wav`、`control`、`config` | 分段合成后拼接 |
| `stream(text, reference_wav=None, control=None, config=None)` | 流式 | `text` | `reference_wav`、`control`、`config` | 返回音频块迭代器 |
| `save(wav, output_path)` | 保存 | `wav`、`output_path` | — | 写入文件 |

#### 属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `model` | `VoxCPM` | 底层模型实例（懒加载） |
| `sample_rate` | `int` | 模型采样率 |

---

### `voxcpm_full.py` — VoxCPMCloner

全功能版，在精简版基础上扩展。

#### 新增方法

| 方法 | 说明 |
|---|---|
| `basic_tts(text, config=None)` | 基础文本转语音，无控制无参考，使用模型默认音色 |
| `phoneme_tts(phonemes, config=None)` | 音素输入，中文 `{ni3}{hao3}` / 英文 `{HH AH0 L OW1}`，自动关闭 normalize |
| `synthesize_dialect(text, dialect, reference_wav=None, config=None)` | 方言合成，自动套用方言控制指令 |

#### 扩展方法

`synthesize_long()` 增加 `mode` 参数：

| mode | 行为 |
|---|---|
| `"auto"`（默认） | 有 `reference_wav` 用 `clone`，否则用 `voice_design` |
| `"basic"` | 强制用 `basic_tts`（忽略 reference_wav 和 control） |
| `"clone"` | 强制用 `clone`（需要 reference_wav） |
| `"design"` | 强制用 `voice_design`（忽略 reference_wav） |

#### 示例

```python
from voxcpm_full import VoxCPMCloner, DIALECT_PRESETS, NON_VERBAL_TAGS

cloner = VoxCPMCloner()

# 基础 TTS
wav = cloner.basic_tts("你好")

# 音素输入（中文带声调拼音）
wav = cloner.phoneme_tts("{ni3}{hao3}{shi4}{jie4}")

# 方言合成（粤语）
wav = cloner.synthesize_dialect(
    text="伙計，唔該一個A餐，凍奶茶少甜！",
    dialect="cantonese",
)

# 非语言标签（拼接到文本中）
wav = cloner.basic_tts("嗯[Uhm]让我想想看，[laughing]这个问题真有趣。")

# 长文本基础 TTS 模式
from voxcpm_full import split_long_text
segments = split_long_text("第一段。第二段。第三段。")
wav = cloner.synthesize_long(segments, mode="basic")
```

---

### `GenerateConfig` 生成参数

所有生成方法的 `config` 参数类型，默认值与官方文档一致。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `cfg_value` | `float` | `2.0` | 引导强度，范围 1.0–3.0。值越高越贴文本，但难例上易出噪声 |
| `inference_timesteps` | `int` | `10` | 扩散步数，范围 4–30。步数越多质量越好但越慢 |
| `normalize` | `bool` | `False` | 文本规范化（展开数字、日期等），原始文本建议启用 |
| `denoise` | `bool` | `False` | 对 prompt/参考音频降噪（不影响生成结果本身）。需构造时 `load_denoiser=True` |
| `retry_badcase` | `bool` | `True` | 生成音频异常偏短/偏长时自动重试 |

```python
from voxcpm_full import GenerateConfig

config = GenerateConfig(
    cfg_value=1.6,         # 长音频发嗡时降到 1.5-1.6
    inference_timesteps=20,  # 更高质量
    normalize=True,        # 自动展开"5640平方米"
)
wav = cloner.clone("总面积5640平方米", reference_wav="speaker.wav", config=config)
```

---

### 常量

`voxcpm_full.py` 导出两个官方 Cookbook 推荐的常量：

#### `NON_VERBAL_TAGS`

非语言标签（建议使用英文方括号小写形式）：

```python
{
    "笑与叹息":   ["[laughing]", "[sigh]"],
    "停顿与思考": ["[Uhm]", "[Shh]"],
    "疑问语气":   ["[Question-ah]", "[Question-ei]", "[Question-en]", "[Question-oh]"],
    "情绪":       ["[Surprise-wa]", "[Surprise-yo]", "[Dissatisfaction-hnn]"],
}
```

用法：拼接到目标文本中，**点到为止，不要在一句话里叠太多**。

```python
cloner.basic_tts("哈哈[laughing]我就知道会是这样。[sigh]")
```

#### `DIALECT_PRESETS`

中文方言预设：

| 键 | 方言 | 控制指令 | 地道示例 |
|---|---|---|---|
| `cantonese` | 粤语 | `Cantonese` | 伙計，唔該一個A餐，凍奶茶少甜！ |
| `sichuanese` | 四川话 | `Sichuan dialect` | 幺儿，哈戳戳得你屋头来噶！ |
| `northeastern` | 东北话 | `Northeastern dialect` | 你搁这整啥玩意儿呢？ |
| `henanese` | 河南话 | `Henan dialect` | 恁这是弄啥嘞？晌午吃啥饭？ |

```python
cloner.synthesize_dialect(
    text="你搁这整啥玩意儿呢？",
    dialect="northeastern",
)
```

---

## CLI 用法

两个模块都内置 CLI，参数完全一致（`voxcpm_full.py` 多 `--phoneme` 和 `--dialect`）。

### 通用参数

```
文本与模式：
  --text, -t TEXT          目标文本（与 --phoneme 二选一）
  --control, -c TEXT       控制指令（如 '年轻女性，温柔甜美'）
  --reference, -r PATH     参考音频路径（克隆模式）
  --prompt-wav PATH        prompt 音频路径（高保真克隆模式）
  --prompt-text TEXT       prompt 音频的逐字转写（高保真模式必填）

模型与输出：
  --model-path PATH        模型路径或 ID（默认自动检测）
  --output, -o PATH        输出音频路径（默认 output.wav）
  --device TEXT            设备（如 'cuda' 或 'cpu'）

生成参数：
  --cfg FLOAT              CFG 强度（默认 2.0）
  --timesteps INT          推理步数（默认 10）
  --normalize              启用文本规范化（展开数字、日期）
  --denoise                对 prompt/参考音频降噪
  --no-retry               禁用异常长度重试
  --load-denoiser          加载降噪器模型（--denoise 时自动启用）

模式开关：
  --long                   长文本分段合成模式
  --stream                 流式生成模式
```

### `voxcpm_full.py` 独有参数

```
  --phoneme TEXT           音素输入（如 '{ni3}{hao3}'，与 --text 二选一）
  --dialect {cantonese,sichuanese,northeastern,henanese}
                           方言模式（自动套用对应控制指令）
```

### 模式选择逻辑

CLI 按以下优先级自动判断模式（无需显式指定）：

```
1. --phoneme              → 音素输入模式
2. --long                 → 长文本分段模式
3. --stream               → 流式模式
4. --prompt-wav           → 高保真克隆模式
5. --reference            → 可控克隆模式
6. --control 或 --dialect → 声音设计模式
7. 都没有                  → 基础 TTS 模式
```

### 完整示例

```bash
# 基础 TTS
python voxcpm_full.py --text "你好，世界！" -o out.wav

# 声音设计
python voxcpm_full.py --text "你好" --control "年轻女性，温柔甜美" -o out.wav

# 可控克隆（含风格控制）
python voxcpm_full.py --text "你好" --reference speaker.wav --control "cheerful, faster" -o out.wav

# 高保真克隆
python voxcpm_full.py --text "你好" --prompt-wav speaker.wav --prompt-text "转写文本" -o out.wav

# 音素输入
python voxcpm_full.py --phoneme "{ni3}{hao3}{shi4}{jie4}" -o out.wav

# 方言合成（粤语）
python voxcpm_full.py --text "伙計，唔該一個A餐" --dialect cantonese -o out.wav

# 长文本分段合成
python voxcpm_full.py --text "很长的文本..." --reference speaker.wav --long -o out.wav

# 流式生成
python voxcpm_full.py --text "你好" --reference speaker.wav --stream -o out.wav

# 启用文本规范化（展开数字、日期）
python voxcpm_full.py --text "总面积5640平方米" --normalize -o out.wav

# 参考音频嘈杂时启用降噪
python voxcpm_full.py --text "你好" --reference noisy.wav --denoise -o out.wav

# 自定义生成参数
python voxcpm_full.py --text "你好" --reference speaker.wav --cfg 1.6 --timesteps 20 -o out.wav
```

---

## 官方文档对照

本模块所有功能均对应官方文档，便于查阅原始资料：

| 本模块功能 | 官方文档位置 |
|---|---|
| 三种克隆模式 | [使用指南 → 三种生成模式](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#three-generation-modes) |
| `GenerateConfig` 各参数 | [使用指南 → 生成参数](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#generation-parameters) |
| 文本规范化 | [使用指南 → 文本规范化](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#text-normalization) |
| 音素输入 | [使用指南 → 普通文本与音素输入](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#regular-text-vs-phoneme-input) |
| 参考音频建议 | [使用指南 → 参考音频](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#reference-audio) |
| CFG 调优 | [使用指南 → CFG 参数](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#cfg-value) |
| 长文本切分 | [使用指南 → 长文本](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#long-text) |
| 流式生成 | [使用指南 → 流式生成](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#streaming) |
| 降噪 | [使用指南 → 降噪](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html#denoise) |
| 方言建议 | [Cookbook → 中文方言](https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html#local-specialties-chinese-dialects) |
| 非语言标签 | [Cookbook → 非语言标签](https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html#extra-spice-non-verbal-tags) |
| 声音设计控制指令 | [Cookbook → 设定声音风格](https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html#step-2-choose-your-flavor-profile-voice-design) |

---

## 常见问题

### Q: 长音频发糊、发嗡怎么办？

降低 `cfg_value` 到 1.5–1.6，或用 `synthesize_long()` 切段合成。

```python
config = GenerateConfig(cfg_value=1.6)
wav = cloner.synthesize_long(segments, reference_wav="speaker.wav", config=config)
```

### Q: 数字被逐位朗读怎么办？

启用 `normalize=True`：

```python
config = GenerateConfig(normalize=True)
wav = cloner.basic_tts("总建筑面积为5640平方米", config=config)
```

### Q: 参考音频嘈杂导致克隆变差怎么办？

构造时 `load_denoiser=True`，生成时 `denoise=True`：

```python
cloner = VoxCPMCloner(load_denoiser=True)
config = GenerateConfig(denoise=True)
wav = cloner.clone("你好", reference_wav="noisy.wav", config=config)
```

注意：降噪器在 16 kHz 流水线中运行，可能轻微改变声线。若克隆反而变差，关闭 `denoise`。

### Q: 高保真克隆的 `prompt_text` 怎么获取？

建议用 ASR 模型自动转写，不要手打：

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)（推荐，官方网页演示使用）
- [Whisper](https://github.com/openai/whisper)

`prompt_text` 必须与 `prompt_wav` **逐字对齐**，不匹配是首尾伪影最常见原因。

### Q: 如何保持多次调用音色一致？

1. 每次复用同一段参考音频
2. 使用 `clone()`（基于 `reference_wav_path`）而非 `basic_tts()`（每次随机）
3. 生产级一致性建议 LoRA 微调，见 [官方微调指南](https://voxcpm.readthedocs.io/zh-cn/latest/finetuning/finetune.html)

### Q: `denoise=True` 报错怎么办？

`denoise=True` 需要降噪器模型，构造时必须 `load_denoiser=True`：

```python
# 错误：会抛 RuntimeError
cloner = VoxCPMCloner()
cloner.clone("...", reference_wav="...", config=GenerateConfig(denoise=True))

# 正确
cloner = VoxCPMCloner(load_denoiser=True)
cloner.clone("...", reference_wav="...", config=GenerateConfig(denoise=True))
```

CLI 中 `--denoise` 会自动启用 `--load-denoiser`，无需手动指定。

### Q: Hi-Fi 模式下控制指令为什么不生效？

这是官方设计：启用 Hi-Fi 模式（提供 `prompt_wav_path` + `prompt_text`）时，控制指令会被忽略。如需风格控制，请用 `clone()` 模式。
