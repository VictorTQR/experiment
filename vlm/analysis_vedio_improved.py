"""
改进的视频分析脚本 - 使用 loguru 记录日志
Improved Video Analysis Script with Loguru Logging
"""

import base64
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import sys

# 配置日志
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
logger.add("logs/video_analysis_{time:YYYY-MM-DD}.log", rotation="500 MB", retention="10 days", encoding="utf-8")

load_dotenv()

from langchain_openai import ChatOpenAI
from extract_vedio_frames import extract_key_frames_histogram

# 视频路径配置
VIDEO_PATH = './data/dy.mp4'
OUTPUT_DIR = './data/frames'


def pic_to_base64(pic_path):
    """
    Convert image to base64 with proper data URI format
    将图片转换为 base64 格式，添加正确的 data URI 前缀
    """
    try:
        with open(pic_path, 'rb') as f:
            base64_str = base64.b64encode(f.read()).decode('utf-8')

        # 返回正确的 data URI 格式
        data_uri = f"data:image/jpeg;base64,{base64_str}"
        logger.debug(f"Converted {pic_path} to base64, length: {len(data_uri)}")
        return data_uri
    except Exception as e:
        logger.error(f"Failed to convert {pic_path} to base64: {str(e)}")
        raise


def extract_frames_if_needed(video_path, output_dir):
    """
    Extract key frames if output directory is empty
    如果输出目录为空，则提取关键帧
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_path.mkdir(parents=True, exist_ok=True)

    frames = list(output_path.glob('*.jpg'))

    if len(frames) == 0:
        logger.info("No frames found. Starting key frame extraction...")
        extract_key_frames_histogram(
            video_path,
            output_dir=output_dir,
            threshold=0.3,
            min_interval=15
        )
        frames = list(output_path.glob('*.jpg'))
        logger.success(f"Extracted {len(frames)} key frames")
    else:
        logger.info(f"Found {len(frames)} existing frames")

    return sorted(frames)


def analyze_frame(frame_path, llm_model="glm-4-flash"):
    """
    Analyze a single frame using VLM
    使用 VLM 分析单个帧
    """
    logger.info(f"Analyzing frame: {frame_path.name}")

    try:
        # Convert image to base64
        base64_uri = pic_to_base64(frame_path)

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_uri,
                            "detail": "high"  # 使用高清晰度
                        }
                    },
                    {
                        "type": "text",
                        "text": "请详细描述这张图片中的内容，包括：1. 画面中有什么人物或物体 2. 他们在做什么动作 3. 背景场景是什么样的 4. 整体画面的氛围如何"
                    }
                ]
            }
        ]

        logger.debug(f"Sending request to {llm_model}...")

        # Call OpenAI API
        llm = ChatOpenAI(model=llm_model)
        response = llm.invoke(messages)
        result = response.content

        logger.success(f"Received response: {len(result)} characters")

        return result

    except Exception as e:
        logger.error(f"Failed to analyze frame {frame_path}: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("Video Analysis with VLM")
    logger.info("=" * 60)

    # Step 1: Extract frames if needed
    logger.info("Step 1: Checking frames...")
    frames = extract_frames_if_needed(VIDEO_PATH, OUTPUT_DIR)

    # Step 2: Analyze frames
    logger.info(f"\nStep 2: Analyzing {len(frames)} frames...")
    results = []

    for idx, frame in enumerate(frames, 1):
        logger.info(f"\n[{idx}/{len(frames)}] Processing: {frame.name}")

        description = analyze_frame(frame)

        if description:
            results.append({
                'frame': str(frame),
                'frame_number': idx,
                'description': description
            })

            # Print result
            print(f"\n{'='*60}")
            print(f"Frame {idx}: {frame.name}")
            print(f"{'='*60}")
            print(description)
            print(f"{'='*60}\n")

        # Test mode: only process first frame
        if idx >= 1:
            logger.info("Test mode: stopping after first frame")
            break

    # Summary
    logger.info(f"\nAnalysis complete! Processed {len(results)} frames")

    # Optional: Save results to file
    if results:
        output_file = "logs/analysis_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"\n{'='*60}\n")
                f.write(f"Frame: {result['frame']}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{result['description']}\n\n")
        logger.success(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
