import edge_tts
from io import BytesIO
from loguru import logger
import re
import time

def split_text_intelligently(text, max_chunk_size):
    """按标点符号智能分割文本，同时控制块的大小"""
    # 先按标点分割成小句
    sentences = re.split(r'([。！？.!?])', text)
    
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        # 重新加上标点
        if i + 1 < len(sentences):
            sentence += sentences[i+1]
            
        # 如果加上这句会超过限制，则保存当前块，开始新块
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

class EdgeTTS:
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", max_segment_length: int = 300):
        self.voice = voice
        self.max_segment_length = max_segment_length

    @staticmethod
    async def list_voices():
        """
        列出所有可用的语音。
        """
        voices = await edge_tts.list_voices()
        # 打印前5个语音的信息
        return voices
        
    async def _stream(self, text: str):
        """
        异步生成TTS音频流。
        """
        audio_buffer = BytesIO()
        communicate = edge_tts.Communicate(text, self.voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        return audio_buffer.getvalue()

    async def arun(self, text: str) -> bytes:
        """
        异步运行TTS，返回音频数据。
        """
        audio = BytesIO()

        try:
            # ... 原有代码
            if len(text) < 300:
                start_time = time.time()
                segment = await self._stream(text)
                audio.write(segment)
                end_time = time.time()
                logger.info(f"正在处理tts片段, 处理进度：1/1, 耗时: {end_time - start_time :.2f} 秒")
            else:
                # 智能分割文本
                segments = split_text_intelligently(text, self.max_segment_length)
                for idx, segment in enumerate(segments):
                    start_time = time.time()
                    audio.write(await self._stream(segment))
                    end_time = time.time()
                    logger.info(f"正在处理tts片段, 处理进度：{idx+1}/{len(segments)}, 耗时: {end_time - start_time :.2f} 秒")

        except Exception as e:
            logger.error(f"TTS处理失败: {str(e)}")
            raise

        return audio.getvalue()
