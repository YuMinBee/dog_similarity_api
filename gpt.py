# file name : gpt.py
# Date created : 2025-08-12
# Author : Yuminbee
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_dog_recommendation(user_text: str, image_path: str) -> str:
    base64_image = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "system",
            "content": "너는 반려견 전문가야. 사용자의 생활방식과 사진 속 정보를 분석해서 적합한 강아지 품종을 추천해줘."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"조건: {user_text}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return resp.choices[0].message.content.strip()

# test
if __name__ == "__main__":
    user_input = "사진과 닮고 털 빠짐이 적고 혼자 사는 여성에게 적합한 소형견 추천해줘"
    image_path = "C:/Users/youmi/OneDrive/Desktop/dog_server/test_img/dog8.jpg"
    print(get_dog_recommendation(user_input, image_path))
