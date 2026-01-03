import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI



vedio_path = './data/dy.mp4'

# extract_key_frames_histogram(vedio_path, output_dir=output_dir)


def pic_to_base64(pic_path):
    """Convert vedio to base64 with proper data URI format"""
    with open(pic_path, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('utf-8')
    # Return proper data URI format
    return f"data:video/mp4;base64,{base64_str}"


llm = ChatOpenAI(model="zai-org/GLM-4.6V")

try:
    base64_str = pic_to_base64(vedio_path)
    print(f"Base64 length: {len(base64_str)} characters")
    print(f"Base64 prefix: {base64_str[:50]}...")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": base64_str,
                    }
                },
                {
                    "type": "text",
                    "text": "请详细描述这张视频中的内容，包含人物、场景、动作、画面文字等细节。"
                }
            ]
        }
    ]

    response = llm.invoke(messages)
    result = response.content
    print(f"Response: {result}")


except Exception as e:
    print(f"Error processing frame {frame}: {str(e)}")
    import traceback
    traceback.print_exc()
