# -*- coding: utf-8 -*-
"""
J.A.R.V.I.S. Backend  (FastAPI + Gemini)
=========================================
Mobil uygulama ses tanımayı telefonda yapar; bu sunucu yalnızca metni
alıp Gemini ile yanıtlar.

Uçlar:
    GET  /api/health  -> {"status": "ok"}
    POST /api/text    -> gövde {"text": "..."}
                         cevap {"type","command","params","speech"}

Yerelde çalıştırma:
    pip install -r requirements.txt
    export GEMINI_API_KEY="..."        # aistudio.google.com'dan al
    uvicorn main:app --reload --port 8000

Render / sunucuda (Start Command):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import json
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ---------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="JARVIS Backend", version="1.0")

# Netlify'deki arayüzün çapraz kaynaktan erişebilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Komut kütüphanesi
# ---------------------------------------------------------------
COMMAND_REGISTRY = {
    "play_music":         "Müzik çal/durdur. params: {action:'play'|'pause', query:'şarkı/sanatçı (ops.)'}",
    "send_sms":           "SMS gönder. params: {to:'kişi/numara', message:'içerik'}",
    "add_calendar_event": "Takvime etkinlik ekle. params: {title:'başlık', datetime:'ISO 8601', location:'(ops.)'}",
    "web_search":         "Web'de ara. params: {query:'sorgu'}",
    "set_alarm":          "Alarm kur. params: {hour:0-23, minute:0-59, label:'etiket'}",
    "set_timer":          "Sayaç kur. params: {seconds:int, label:'etiket'}",
    "open_app":           "Uygulama aç. params: {app_name:'ad'}",
    "get_weather":        "Hava durumu. params: {city:'şehir, varsayılan İzmir'}",
}


def build_system_prompt() -> str:
    cmds = "\n".join(f"- {k}: {v}" for k, v in COMMAND_REGISTRY.items())
    now = datetime.now().strftime("%Y-%m-%d %H:%M, %A")
    return f"""Sen JARVIS'sin — Türkçe konuşan kişisel ses asistanı.
Kısa, doğal ve hafif esprili konuş (filmdeki Jarvis: zarif, sakin, yardımsever). Kullanıcıya "efendim" diye hitap et.
Şu anki tarih/saat: {now}

Kullanıcının söylediğini analiz et ve YALNIZCA şu JSON formatında cevap ver, başka hiçbir şey yazma:

{{
  "type": "command" | "chat",
  "command": "komut_adi (yalnızca type=command ise)",
  "params": {{ ...parametreler... }},
  "speech": "kullanıcıya sesli okunacak kısa Türkçe cevap"
}}

Tanıdığın komutlar:
{cmds}

Kurallar:
- İstek bir komuta uyuyorsa type="command" kullan, parametreleri doldur, speech'te kibar bir onay ver.
- Sohbet/soru/bilgi isteğiyse type="chat" kullan, cevabı speech'e yaz.
- Tarih/saat ifadelerini ("yarın 3'te") ISO 8601'e çevir.
- Eksik parametre varsa speech içinde kibarca sor, type="chat" yap.
"""


# Model nesnesini tembel (lazy) kur — her istekte güncel sistem promptu ile
def get_model():
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=build_system_prompt(),
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.4,
            "max_output_tokens": 1024,
        },
    )


# Basit oturum belleği (production'da oturum bazlı yapılmalı)
_history = []
MAX_TURNS = 10


def ask_gemini(user_text: str) -> dict:
    global _history
    contents = _history + [{"role": "user", "parts": [user_text]}]

    model = get_model()
    resp = model.generate_content(contents)
    raw = (getattr(resp, "text", "") or "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"type": "chat", "speech": raw or "Bir hata oldu efendim, tekrar dener misiniz?"}

    _history.append({"role": "user", "parts": [user_text]})
    _history.append({"role": "model", "parts": [raw]})
    _history = _history[-MAX_TURNS * 2:]
    return result


# ---------------------------------------------------------------
# Şema
# ---------------------------------------------------------------
class TextIn(BaseModel):
    text: str


# ---------------------------------------------------------------
# Uçlar
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {"name": "JARVIS Backend", "model": GEMINI_MODEL, "endpoints": ["/api/health", "/api/text"]}


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Jarvis çevrimiçi, efendim."}


@app.post("/api/text")
def text_endpoint(body: TextIn):
    user_text = (body.text or "").strip()
    if not user_text:
        return {"error": "'text' alanı boş"}
    try:
        result = ask_gemini(user_text)
    except Exception as e:
        return {"type": "chat", "speech": "Beynimle bağlantı kuramadım efendim.", "error": str(e)}
    result["transcript"] = user_text
    return result


if __name__ == "__main__":
    import uvicorn
    if not GEMINI_API_KEY:
        print("[UYARI] GEMINI_API_KEY ayarlanmamış! aistudio.google.com'dan alıp ayarlayın.")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
