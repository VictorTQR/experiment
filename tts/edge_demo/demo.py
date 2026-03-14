import asyncio
import edge_tts
import os
import subprocess
from pathlib import Path

# ==========================================
# 1. 列出所有可用的声音
# ==========================================
async def list_available_voices():
    """
    获取并打印所有可用的 Edge TTS 声音。
    通常用于查找特定语言或性别的声音名称。
    """
    print("\n--- [Demo 1] 正在获取可用声音列表 ---")
    try:
        voices = await edge_tts.list_voices()
        print(f"共找到 {len(voices)} 种声音。\n")
        
        # 仅展示前3种声音的详细信息，避免刷屏
        print("示例数据 (前3个):")
        for i, voice in enumerate(voices[:3]):
            print(f"{i+1}. 名称: {voice['ShortName']}")
            print(f"   性别: {voice['Gender']}")
            print(f"   语言: {voice['Locale']}")
            print(f"   友好名称: {voice['FriendlyName']}")
            print("-" * 30)
            
        # 查找特定的中文声音示例
        print("正在查找中文(大陆)声音...")
        zh_voices = [v for v in voices if v['Locale'].startswith('zh-CN')]
        print(f"中文声音数量: {len(zh_voices)}")
        if zh_voices:
            print(f"推荐中文女声: {zh_voices[0]['ShortName']}") # 通常是 Xiaoxiao
            
    except Exception as e:
        print(f"获取声音列表失败: {e}")

# ==========================================
# 2. 基础文本转语音
# ==========================================
async def basic_text_to_speech(text, output_file):
    """
    最基础的 TTS 功能：将文本转换为 MP3 文件。
    """
    print(f"\n--- [Demo 2] 基础转换: '{text}' ---")
    try:
        # 使用默认声音 或指定声音
        # 这里的 communicate 是核心类
        communicate = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
        
        print(f"正在生成音频 -> {output_file} ...")
        await communicate.save(output_file)
        
        # 验证文件是否存在
        if os.path.exists(output_file):
            print(f"成功! 文件大小: {os.path.getsize(output_file)} bytes")
        else:
            print("失败: 文件未生成。")
            
    except Exception as e:
        print(f"TTS 转换失败: {e}")

# ==========================================
# 3. 带参数调节的文本转语音 (语速、音调、音量)
# ==========================================
async def text_to_speech_with_settings(text, output_file):
    """
    演示如何调整语速、音调和音量。
    参数格式:
    - rate: '-50%' (慢) 到 '+100%' (快), 默认 '+0%'
    - pitch: '-50Hz' (低) 到 '+50Hz' (高), 默认 '+0Hz'
    - volume: '+0%' 到 '+100%'
    """
    print(f"\n--- [Demo 3] 带参数转换 (语速快, 音调高) ---")
    try:
        communicate = edge_tts.Communicate(
            text, 
            voice="zh-CN-YunxiNeural",  # 使用男声云希
            rate="+50%",  # 语速加快 50%
            pitch="+10Hz", # 音调稍微提高
            volume="+50%"  # 音量增加
        )
        
        print(f"正在生成音频 -> {output_file} ...")
        await communicate.save(output_file)
        print("生成完毕!")
        
    except Exception as e:
        print(f"TTS 转换失败: {e}")

# ==========================================
# 4. 使用命令行工具生成字幕（备选方案）
# ==========================================
async def generate_audio_with_subtitles_cli(text, audio_file, subtitle_file):
    """
    使用 edge-tts 命令行工具生成音频和字幕。
    这是最可靠的方法，因为命令行工具已经内置了字幕生成。
    """
    print(f"\n--- [Demo 4] 使用命令行工具生成音频与字幕 ---")
    
    try:
        # 构造命令
        cmd = [
            "edge-tts",
            "--text", text,
            "--voice", "zh-CN-XiaoyiNeural",
            "--write-media", audio_file,
            "--write-subtitles", subtitle_file
        ]
        
        print(f"正在生成音频和字幕...")
        print(f"命令: {' '.join(cmd)}")
        
        # 运行命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print(f"音频已保存: {audio_file}")
        print(f"字幕已保存: {subtitle_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"命令行工具执行失败: {e}")
        print(f"错误输出: {e.stderr}")
    except Exception as e:
        print(f"生成失败: {e}")

# ==========================================
# 5. 使用 Python API 生成字幕（修正版）
# ==========================================
async def generate_audio_with_subtitles_python(text, audio_file, subtitle_file):
    """
    使用 Python API 生成字幕（修正版）。
    注意：此代码可能需要根据实际 edge-tts 版本调整。
    """
    print(f"\n--- [Demo 5] 使用 Python API 生成音频与字幕 ---")
    
    communicate = edge_tts.Communicate(text, voice="zh-CN-XiaoyiNeural")
    
    # 使用 SubMaker（注意：API 可能已更改）
    # 请根据实际安装的 edge-tts 版本调整方法名
    try:
        submaker = edge_tts.SubMaker()
        
        with open(audio_file, "wb") as audio, open(subtitle_file, "w", encoding='utf-8') as sub:
            print("正在流式处理数据...")
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # 写入音频二进制数据
                    audio.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    # 处理字幕时间轴
                    # 注意：这里需要根据实际 API 调整
                    # 可能的参数：text, offset, duration
                    # 或者使用不同的方法名
                    pass
            
            # 尝试不同的方法名生成字幕
            # 方法名可能是：generate_srt(), generate_subs(), write_srt() 等
            # 请根据实际 API 文档调整
            try:
                # 尝试旧方法名
                sub.write(submaker.generate_subs())
            except AttributeError:
                # 如果旧方法名不存在，尝试新方法名
                try:
                    sub.write(submaker.generate_srt())
                except AttributeError:
                    # 如果都没有，手动生成简单的 SRT 格式
                    print("警告：SubMaker API 不支持，使用手动生成")
                    # 手动生成简单字幕（不推荐，仅作为示例）
                    sub.write("1\n00:00:00,000 --> 00:00:05,000\n生成字幕失败，请使用命令行工具\n\n")
            
        print(f"音频已保存: {audio_file}")
        print(f"字幕已保存: {subtitle_file}")
        
    except Exception as e:
        print(f"生成失败: {e}")
        print("建议使用命令行工具方案（Demo 4）")

# ==========================================
# 主函数入口
# ==========================================
async def main():
    # 定义输出目录
    output_dir = "output_audio"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 1. 列出声音
    await list_available_voices()

    # 测试用的文本
    sample_text = "你好，这是一个 Edge TTS 的完整演示。它非常强大且易于使用。"

    # 2. 基础转换
    await basic_text_to_speech(sample_text, os.path.join(output_dir, "demo_basic.mp3"))

    # 3. 带参数转换
    await text_to_speech_with_settings(sample_text, os.path.join(output_dir, "demo_settings.mp3"))

    # 4. 使用命令行工具生成字幕（推荐）
    await generate_audio_with_subtitles_cli(
        sample_text, 
        os.path.join(output_dir, "demo_with_subs_cli.mp3"),
        os.path.join(output_dir, "demo_with_subs_cli.srt")
    )
    
    # 5. 使用 Python API 生成字幕（可能需要调整）
    await generate_audio_with_subtitles_python(
        sample_text, 
        os.path.join(output_dir, "demo_with_subs_python.mp3"),
        os.path.join(output_dir, "demo_with_subs_python.srt")
    )
    
    print("\n所有演示完成!")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())