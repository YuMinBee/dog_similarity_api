# file name : main.py
# Date created : 2025-08-12
# Author : Yuminbee

import os, io
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
from similarity import DogSearcher
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not in .env.")

client = OpenAI(api_key=OPENAI_API_KEY)

# data paths
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
EMB_PATH  = os.path.join(DATA_DIR, "dog_clip_embeddings.npy")
URLS_PATH = os.path.join(DATA_DIR, "dog_image_urls.json")

app = FastAPI(title="Dog Reco + Similarity API")

# load once when server starts
searcher = DogSearcher(emb_path=EMB_PATH, urls_path=URLS_PATH)

def gpt_recommend(user_text: str, pil_img: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    messages = [
        {"role": "system",
         "content": "너는 반려견 전문가야. 사용자의 생활방식과 사진 속 정보를 함께 고려해 적합한 품종을 추천하고, 이유와 주의점을 간결히 설명해줘."},
        {"role": "user",
         "content": [
             {"type": "text", "text": f"조건: {user_text}"},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}" }}
         ]}
    ]

    resp = client.chat.completions.create(model="gpt-4o", messages=messages)
    return resp.choices[0].message.content.strip()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recommend-and-search")
async def recommend_and_search(
    user_text: str = Form(...),
    image: UploadFile = File(...),
    top_k: int = Form(5)
):
    # 1) PIL conversion of upload images
    img_bytes = await image.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 2) Recommendation
    try:
        recommendation = gpt_recommend(user_text, pil)
    except Exception as e:
        return JSONResponse({"error": f"GPT error : {e}"}, status_code=500)

    # 3) Similar images Top‑K (auto skip dead URL)
    try:
        cands = searcher.search_by_pil(pil, top_k=top_k)
        top_alive = searcher.topk_alive(cands, top_k=top_k)
    except Exception as e:
        return JSONResponse({"error": f"similarity search error : {e}"}, status_code=500)

    return {
        "recommendation": recommendation,
        "similar": top_alive  # [{idx, sim, url}, ...]
    }
