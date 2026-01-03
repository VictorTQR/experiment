import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from extract_vedio_frames import extract_key_frames_histogram


vedio_path = './data/dy.mp4'
output_dir = './data/frames'

# extract_key_frames_histogram(vedio_path, output_dir=output_dir)


def pic_to_base64(pic_path):
    """Convert image to base64 with proper data URI format"""
    with open(pic_path, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('utf-8')
    # Return proper data URI format
    return f"data:image/jpeg;base64,{base64_str}"


llm = ChatOpenAI(model="zai-org/GLM-4.6V")
frames = sorted(Path(output_dir).glob('*.jpg'))
print(f"Found {len(frames)} frames to process")

for idx, frame in enumerate(frames, 1):
    print(f"\nProcessing frame {idx}/{len(frames)}: {frame.name}")

    try:
        base64_str = pic_to_base64(frame)
        print(f"Base64 length: {len(base64_str)} characters")
        print(f"Base64 prefix: {base64_str[:50]}...")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_str,
                        }
                    },
                    {
                        "type": "text",
                        "text": "请详细描述这张图片中的内容，包括人物、场景、动作等细节。"
                    }
                ]
            }
        ]

        response = llm.invoke(messages)
        result = response.content

        # import openai
        # response = openai.chat.completions.create(
        #     model="glm-4-flash",
        #     messages=messages,
        # )
        # result = response.choices[0].message.content
        print(f"Response: {result}")


    except Exception as e:
        print(f"Error processing frame {frame}: {str(e)}")
        import traceback
        traceback.print_exc()
