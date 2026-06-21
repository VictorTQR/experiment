"""VoxCPM 全功能声音合成与克隆模块。

涵盖官方文档全部能力：
  基础合成：
    - basic_tts()        纯文本转语音（无控制、无参考）
    - voice_design()     声音设计：文本描述生成全新音色
    - clone()            可控克隆：参考音频 + 可选风格控制
    - hifi_clone()       高保真克隆：参考音频 + 逐字转写

  高级输入：
    - phoneme_tts()      音素输入（中文 {ni3}{hao3} / 英文 {HH AH0 L OW1}）
    - 非语言标签常量      NON_VERBAL_TAGS（[laughing] / [sigh] 等）
    - 方言预设           DIALECT_PRESETS（粤语/四川话/东北话/河南话）

  长文本与流式：
    - synthesize_long()  长文本分段合成后拼接
    - stream()           流式生成音频块

  工具：
    - split_long_text()  长文本切分
    - save()             保存音频
    - CLI 入口           python voxcpm_full.py --text ... [选项]

作为模块使用：
    from voxcpm_full import VoxCPMCloner, DIALECT_PRESETS, NON_VERBAL_TAGS

    cloner = VoxCPMCloner()
    wav = cloner.clone("你好", reference_wav="speaker.wav")
    cloner.save(wav, "out.wav")

作为 CLI 使用：
    python voxcpm_full.py --text "你好" -o out.wav
    python voxcpm_full.py --text "你好" --control "年轻女性" -o out.wav
    python voxcpm_full.py --text "你好" --reference speaker.wav -o out.wav
    python voxcpm_full.py --text "你好" --prompt-wav speaker.wav --prompt-text "转写" -o out.wav
    python voxcpm_full.py --phoneme "{ni3}{hao3}{shi4}{jie4}" -o out.wav
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import soundfile as sf
from voxcpm import VoxCPM

PathLike = Union[str, Path]

# ---------------------------------------------------------------------- #
# 常量：非语言标签与方言预设（来自官方 Cookbook）
# ---------------------------------------------------------------------- #

#: 推荐的非语言标签（Cookbook 建议使用英文方括号小写形式）
NON_VERBAL_TAGS: dict[str, list[str]] = {
    "笑与叹息": ["[laughing]", "[sigh]"],
    "停顿与思考": ["[Uhm]", "[Shh]"],
    "疑问语气": ["[Question-ah]", "[Question-ei]", "[Question-en]", "[Question-oh]"],
    "情绪": ["[Surprise-wa]", "[Surprise-yo]", "[Dissatisfaction-hnn]"],
}

#: 中文方言预设（Cookbook 推荐地道说法）
DIALECT_PRESETS: dict[str, dict[str, str]] = {
    "cantonese": {
        "name": "粤语",
        "control": "Cantonese",
        "sample": "伙計，唔該一個A餐，凍奶茶少甜！",
    },
    "sichuanese": {
        "name": "四川话",
        "control": "Sichuan dialect",
        "sample": "幺儿，哈戳戳得你屋头来噶！",
    },
    "northeastern": {
        "name": "东北话",
        "control": "Northeastern dialect",
        "sample": "你搁这整啥玩意儿呢？",
    },
    "henanese": {
        "name": "河南话",
        "control": "Henan dialect",
        "sample": "恁这是弄啥嘞？晌午吃啥饭？",
    },
}

#: 默认模型路径候选（按优先级降序）
_DEFAULT_MODEL_CANDIDATES: list[Optional[str]] = [
    os.environ.get("VOXCPM_MODEL_PATH"),
    str(Path.home() / ".cache" / "modelscope" / "hub" / "models" / "OpenBMB" / "VoxCPM2"),
    "openbmb/VoxCPM2",
]


def _resolve_model_path(model_path: Optional[PathLike] = None) -> Union[str, Path]:
    """解析模型路径：显式指定 > 环境变量 > ModelScope 缓存 > 模型 ID。"""
    if model_path is not None:
        return model_path
    for candidate in _DEFAULT_MODEL_CANDIDATES:
        if candidate is None:
            continue
        if "/" in candidate and not Path(candidate).exists():
            return candidate
        if Path(candidate).exists():
            return candidate
    return "openbmb/VoxCPM2"


@dataclass
class GenerateConfig:
    """生成参数配置，默认值与官方文档一致。"""

    cfg_value: float = 2.0  # 引导强度 1.0-3.0
    inference_timesteps: int = 10  # 扩散步数 4-30
    normalize: bool = False  # 文本规范化（展开数字、日期）
    denoise: bool = False  # 对 prompt/参考音频降噪
    retry_badcase: bool = True  # 异常长度自动重试

    def as_kwargs(self) -> dict:
        return {
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.inference_timesteps,
            "normalize": self.normalize,
            "denoise": self.denoise,
            "retry_badcase": self.retry_badcase,
        }


# 中英文句末标点切分
_SENT_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?\.\n])\s*")


def split_long_text(text: str, max_chars: int = 200) -> list[str]:
    """将长文本切分为较短段落，避免长文生成不稳定。

    Args:
        text: 输入文本
        max_chars: 每段最大字符数，过短的相邻段会被合并

    Returns:
        切分后的文本段列表（已 strip，非空）
    """
    sentences = [s.strip() for s in _SENT_SPLIT_PATTERN.split(text) if s.strip()]
    merged: list[str] = []
    for s in sentences:
        if merged and len(merged[-1]) + len(s) < max_chars:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    return merged


class VoxCPMCloner:
    """VoxCPM 全功能声音合成与克隆器。

    覆盖官方四种合成方式 + 音素输入 + 长文本 + 流式。

    Args:
        model_path: 模型路径或 ID。None 时按 环境变量 > ModelScope 缓存 > 模型 ID 自动检测。
        load_denoiser: 是否加载降噪器。若生成时要用 denoise=True，必须为 True。
        device: 可选设备字符串（如 'cuda'、'cpu'），None 时由 VoxCPM 决定。
        lazy: True 时延迟到首次使用才加载模型。
    """

    def __init__(
        self,
        model_path: Optional[PathLike] = None,
        load_denoiser: bool = False,
        device: Optional[str] = None,
        lazy: bool = False,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.load_denoiser = load_denoiser
        self.device = device
        self._model: Optional[VoxCPM] = None
        if not lazy:
            self._load()

    def _load(self) -> None:
        if self._model is None:
            kwargs: dict = {"load_denoiser": self.load_denoiser}
            if self.device is not None:
                kwargs["device"] = self.device
            self._model = VoxCPM.from_pretrained(self.model_path, **kwargs)

    @property
    def model(self) -> VoxCPM:
        """懒加载属性，首次访问时加载模型。"""
        self._load()
        assert self._model is not None
        return self._model

    @property
    def sample_rate(self) -> int:
        """模型采样率。"""
        return self.model.tts_model.sample_rate

    def _check_denoiser(self, config: GenerateConfig) -> None:
        if config.denoise and not self.load_denoiser:
            raise RuntimeError(
                "config.denoise=True 需要在构造时设置 load_denoiser=True，"
                "例如 VoxCPMCloner(load_denoiser=True)"
            )

    @staticmethod
    def _wrap_control(text: str, control: Optional[str]) -> str:
        """将控制指令包裹在括号内拼接到文本前。"""
        return f"({control}){text}" if control else text

    # ------------------------------------------------------------------ #
    # 基础合成模式
    # ------------------------------------------------------------------ #

    def basic_tts(
        self,
        text: str,
        config: Optional[GenerateConfig] = None,
    ) -> np.ndarray:
        """基础文本转语音。无控制指令、无参考音频，使用模型默认音色。

        对应官方 1_basic_tts.py 示例。每次调用音色可能不同，
        若需音色一致请使用 clone() 或 voice_design()。

        Args:
            text: 目标文本
            config: 生成参数，None 使用默认值

        Returns:
            numpy 一维音频数组
        """
        config = config or GenerateConfig()
        self._check_denoiser(config)
        return self.model.generate(text=text, **config.as_kwargs())

    def voice_design(
        self,
        text: str,
        control: Optional[str] = None,
        config: Optional[GenerateConfig] = None,
    ) -> np.ndarray:
        """模式 1：声音设计。无需参考音频，用文字描述生成全新音色。

        Args:
            text: 目标文本
            control: 控制指令，写在括号内的风格描述。
                     例如 "年轻女性，温柔甜美" 或 "A young woman, gentle"。
                     为 None 时退化为 basic_tts。
            config: 生成参数

        Returns:
            numpy 一维音频数组
        """
        config = config or GenerateConfig()
        self._check_denoiser(config)
        full_text = self._wrap_control(text, control)
        return self.model.generate(text=full_text, **config.as_kwargs())

    def clone(
        self,
        text: str,
        reference_wav: PathLike,
        control: Optional[str] = None,
        config: Optional[GenerateConfig] = None,
    ) -> np.ndarray:
        """模式 2：可控声音克隆。参考音频提供音色，可选风格控制。

        Args:
            text: 目标文本
            reference_wav: 参考音频路径，建议 5-30 秒干净音频
            control: 可选风格指令（如 "speaking faster, cheerful tone"），
                     不影响音色身份，仅调节情绪/语速/风格
            config: 生成参数

        Returns:
            numpy 一维音频数组
        """
        config = config or GenerateConfig()
        self._check_denoiser(config)
        full_text = self._wrap_control(text, control)
        return self.model.generate(
            text=full_text,
            reference_wav_path=str(reference_wav),
            **config.as_kwargs(),
        )

    def hifi_clone(
        self,
        text: str,
        prompt_wav: PathLike,
        prompt_text: str,
        reference_wav: Optional[PathLike] = None,
        config: Optional[GenerateConfig] = None,
    ) -> np.ndarray:
        """模式 3：高保真克隆。同时提供参考音频与逐字转写，相似度最高。

        Args:
            text: 目标文本
            prompt_wav: prompt 音频路径
            prompt_text: prompt 音频的逐字转写（建议用 ASR 获取，如 SenseVoice/Whisper）
            reference_wav: 额外参考音频，None 时与 prompt_wav 相同
            config: 生成参数

        Note:
            启用 Hi-Fi 模式时控制指令会被忽略。
        """
        config = config or GenerateConfig()
        self._check_denoiser(config)
        ref = reference_wav if reference_wav is not None else prompt_wav
        return self.model.generate(
            text=text,
            prompt_wav_path=str(prompt_wav),
            prompt_text=prompt_text,
            reference_wav_path=str(ref),
            **config.as_kwargs(),
        )

    # ------------------------------------------------------------------ #
    # 高级输入
    # ------------------------------------------------------------------ #

    def phoneme_tts(
        self,
        phonemes: str,
        config: Optional[GenerateConfig] = None,
    ) -> np.ndarray:
        """音素输入合成，用于细粒度发音控制。

        中文使用带声调数字的拼音，英文使用 CMUDict 式音素：
            中文：phoneme_tts("{ni3}{hao3}{shi4}{jie4}")
            英文：phoneme_tts("{HH AH0 L OW1}")

        Args:
            phonemes: 音素字符串，每个音素组用花括号包裹
            config: 生成参数（normalize 会被强制设为 False）

        Returns:
            numpy 一维音频数组
        """
        config = config or GenerateConfig()
        config.normalize = False  # 音素输入必须关闭文本规范化
        self._check_denoiser(config)
        return self.model.generate(text=phonemes, **config.as_kwargs())

    def synthesize_dialect(
        self,
        text: str,
        dialect: str,
        reference_wav: Optional[PathLike] = None,
        config: Optional[GenerateConfig] = None,
    ) -> np.ndarray:
        """方言合成。使用方言地道说法 + 方言控制指令。

        Args:
            text: 方言文本（须用该方言地道说法，勿用普通话硬套）
            dialect: 方言键名，可选值见 DIALECT_PRESETS 的 key
                     (cantonese / sichuanese / northeastern / henanese)
            reference_wav: 可选参考音频，提供时用克隆模式
            config: 生成参数

        Returns:
            numpy 一维音频数组
        """
        if dialect not in DIALECT_PRESETS:
            raise ValueError(
                f"未知方言 '{dialect}'，可选: {list(DIALECT_PRESETS.keys())}"
            )
        preset = DIALECT_PRESETS[dialect]
        config = config or GenerateConfig()
        if reference_wav:
            return self.clone(
                text=text,
                reference_wav=reference_wav,
                control=preset["control"],
                config=config,
            )
        return self.voice_design(text=text, control=preset["control"], config=config)

    # ------------------------------------------------------------------ #
    # 长文本与流式
    # ------------------------------------------------------------------ #

    def synthesize_long(
        self,
        segments: list[str],
        reference_wav: Optional[PathLike] = None,
        control: Optional[str] = None,
        config: Optional[GenerateConfig] = None,
        mode: str = "auto",
    ) -> np.ndarray:
        """长文本分段合成后拼接，避免长文生成不稳定。

        Args:
            segments: 文本段列表（可用 split_long_text() 切分）
            reference_wav: 非空时使用克隆模式，否则用声音设计模式
            control: 可选风格控制
            config: 生成参数
            mode: 合成模式
                - "auto": 有 reference_wav 用 clone，否则用 voice_design（默认）
                - "basic": 强制用 basic_tts（忽略 reference_wav 和 control）
                - "clone": 强制用 clone（需要 reference_wav）
                - "design": 强制用 voice_design（忽略 reference_wav）

        Returns:
            拼接后的完整音频
        """
        config = config or GenerateConfig()
        if mode == "basic":
            use_mode = "basic"
        elif mode == "clone":
            if not reference_wav:
                raise ValueError("mode='clone' 需要 reference_wav")
            use_mode = "clone"
        elif mode == "design":
            use_mode = "design"
        elif mode == "auto":
            use_mode = "clone" if reference_wav else "design"
        else:
            raise ValueError(f"未知 mode='{mode}'，可选: auto / basic / clone / design")

        wavs: list[np.ndarray] = []
        for i, seg in enumerate(segments, 1):
            print(f"  合成第 {i}/{len(segments)} 段...")
            if use_mode == "basic":
                wavs.append(self.basic_tts(seg, config=config))
            elif use_mode == "clone":
                wavs.append(
                    self.clone(seg, reference_wav=reference_wav, control=control, config=config)
                )
            else:  # design
                wavs.append(self.voice_design(seg, control=control, config=config))
        return np.concatenate(wavs)

    def stream(
        self,
        text: str,
        reference_wav: Optional[PathLike] = None,
        control: Optional[str] = None,
        config: Optional[GenerateConfig] = None,
    ) -> Iterator[np.ndarray]:
        """流式生成音频块，适合交互场景。

        Args:
            text: 目标文本
            reference_wav: 可选参考音频（克隆模式）
            control: 可选风格控制
            config: 生成参数

        Yields:
            音频块 numpy 数组，可用 np.concatenate 拼接
        """
        config = config or GenerateConfig()
        self._check_denoiser(config)
        full_text = self._wrap_control(text, control)
        kwargs = config.as_kwargs()
        if reference_wav:
            kwargs["reference_wav_path"] = str(reference_wav)
        for chunk in self.model.generate_streaming(text=full_text, **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def save(self, wav: np.ndarray, output_path: PathLike) -> None:
        """保存音频到文件（WAV/FLAC/MP3 等，取决于 soundfile 后端）。"""
        sf.write(str(output_path), wav, self.sample_rate)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VoxCPM 全功能声音合成与克隆工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 基础 TTS（无控制、无参考）
  python voxcpm_full.py --text "你好，世界！" -o out.wav

  # 声音设计
  python voxcpm_full.py --text "你好" --control "年轻女性，温柔甜美" -o out.wav

  # 可控克隆
  python voxcpm_full.py --text "你好" --reference speaker.wav -o out.wav
  python voxcpm_full.py --text "你好" --reference speaker.wav --control "cheerful" -o out.wav

  # 高保真克隆
  python voxcpm_full.py --text "你好" --prompt-wav speaker.wav --prompt-text "转写" -o out.wav

  # 音素输入
  python voxcpm_full.py --phoneme "{ni3}{hao3}{shi4}{jie4}" -o out.wav

  # 方言合成
  python voxcpm_full.py --text "伙計，唔該一個A餐" --dialect cantonese -o out.wav

  # 长文本分段合成
  python voxcpm_full.py --text "很长的文本..." --reference speaker.wav --long -o out.wav

  # 流式生成
  python voxcpm_full.py --text "你好" --reference speaker.wav --stream -o out.wav

  # 启用文本规范化（展开数字、日期）
  python voxcpm_full.py --text "总面积5640平方米" --normalize -o out.wav

  # 参考音频嘈杂时启用降噪
  python voxcpm_full.py --text "你好" --reference noisy.wav --denoise -o out.wav
""",
    )
    # 文本与模式
    p.add_argument("--text", "-t", help="目标文本（与 --phoneme 二选一）")
    p.add_argument("--phoneme", help="音素输入（如 '{ni3}{hao3}'，与 --text 二选一）")
    p.add_argument("--control", "-c", help="控制指令（如 '年轻女性，温柔甜美'）")
    p.add_argument("--reference", "-r", help="参考音频路径（克隆模式）")
    p.add_argument("--prompt-wav", help="prompt 音频路径（高保真克隆模式）")
    p.add_argument("--prompt-text", help="prompt 音频的逐字转写（高保真模式必填）")
    p.add_argument(
        "--dialect",
        choices=list(DIALECT_PRESETS.keys()),
        help="方言模式（自动套用对应控制指令）",
    )

    # 模型与输出
    p.add_argument("--model-path", help="模型路径或 ID（默认自动检测）")
    p.add_argument("--output", "-o", default="output.wav", help="输出音频路径（默认 output.wav）")
    p.add_argument("--device", help="设备（如 'cuda' 或 'cpu'）")

    # 生成参数
    p.add_argument("--cfg", type=float, default=2.0, help="CFG 强度（默认 2.0）")
    p.add_argument("--timesteps", type=int, default=10, help="推理步数（默认 10）")
    p.add_argument("--normalize", action="store_true", help="启用文本规范化（展开数字、日期）")
    p.add_argument("--denoise", action="store_true", help="对 prompt/参考音频降噪")
    p.add_argument("--no-retry", action="store_true", help="禁用异常长度重试")
    p.add_argument(
        "--load-denoiser",
        action="store_true",
        help="加载降噪器模型（--denoise 时自动启用）",
    )

    # 模式开关
    p.add_argument("--long", action="store_true", help="长文本分段合成模式")
    p.add_argument("--stream", action="store_true", help="流式生成模式")
    return p


def main() -> None:
    args = _build_cli().parse_args()

    # 参数校验
    if not args.text and not args.phoneme:
        raise SystemExit("必须提供 --text 或 --phoneme 之一")
    if args.text and args.phoneme:
        raise SystemExit("--text 和 --phoneme 不能同时使用")
    if args.prompt_wav and not args.prompt_text:
        raise SystemExit("高保真克隆模式需要同时提供 --prompt-wav 和 --prompt-text")
    if args.long and args.stream:
        raise SystemExit("--long 和 --stream 不能同时使用")
    if args.long and args.prompt_wav:
        raise SystemExit("--long 模式不支持高保真克隆")
    if args.long and args.phoneme:
        raise SystemExit("--long 模式不支持音素输入")
    if args.stream and args.phoneme:
        raise SystemExit("--stream 模式不支持音素输入")
    if args.dialect and args.control:
        raise SystemExit("--dialect 和 --control 不能同时使用（dialect 会自动设置控制指令）")

    config = GenerateConfig(
        cfg_value=args.cfg,
        inference_timesteps=args.timesteps,
        normalize=args.normalize,
        denoise=args.denoise,
        retry_badcase=not args.no_retry,
    )

    cloner = VoxCPMCloner(
        model_path=args.model_path,
        load_denoiser=args.load_denoiser or args.denoise,
        device=args.device,
    )

    print(f"模型: {cloner.model_path}")
    print(f"采样率: {cloner.sample_rate} Hz")

    # 控制指令：dialect 优先于显式 control
    effective_control = args.control
    if args.dialect:
        effective_control = DIALECT_PRESETS[args.dialect]["control"]
        print(f"方言模式: {DIALECT_PRESETS[args.dialect]['name']} (control={effective_control})")

    # ---------- 音素输入模式 ----------
    if args.phoneme:
        print("音素输入模式")
        wav = cloner.phoneme_tts(phonemes=args.phoneme, config=config)

    # ---------- 长文本模式 ----------
    elif args.long:
        segments = split_long_text(args.text)
        print(f"长文本模式：切分为 {len(segments)} 段")
        wav = cloner.synthesize_long(
            segments=segments,
            reference_wav=args.reference,
            control=effective_control,
            config=config,
        )

    # ---------- 流式模式 ----------
    elif args.stream:
        print("流式生成中...")
        chunks = list(
            cloner.stream(
                text=args.text,
                reference_wav=args.reference,
                control=effective_control,
                config=config,
            )
        )
        wav = np.concatenate(chunks)

    # ---------- 高保真克隆 ----------
    elif args.prompt_wav:
        print("高保真克隆模式")
        wav = cloner.hifi_clone(
            text=args.text,
            prompt_wav=args.prompt_wav,
            prompt_text=args.prompt_text,
            config=config,
        )

    # ---------- 可控克隆 ----------
    elif args.reference:
        print("可控克隆模式")
        wav = cloner.clone(
            text=args.text,
            reference_wav=args.reference,
            control=effective_control,
            config=config,
        )

    # ---------- 声音设计 / 基础 TTS ----------
    else:
        if effective_control:
            print("声音设计模式")
            wav = cloner.voice_design(
                text=args.text, control=effective_control, config=config
            )
        else:
            print("基础 TTS 模式")
            wav = cloner.basic_tts(text=args.text, config=config)

    cloner.save(wav, args.output)
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
