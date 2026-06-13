# -*- coding: utf-8 -*-
"""
J.A.R.V.I.S. Backend  (hafif sürüm — Gemini 3.1 Pro)
=====================================================
Mobil uygulama ses tanımayı telefonda (Web Speech API) yaptığı için
bu backend yalnızca metin alır ve Gemini ile yanıt üretir.

Uçlar:
    GET  /api/health  → { "status": "ok" }
    POST /api/text    → gövde {"text": "..."}
                        cevap {"type": "...", "command": "...", "params": {...}, "speech": "..."}

Yerelde çalıştırma:
    pip install -r requirements.txt
    export GEMINI_API_KEY="..."        # aistudio.google.com'dan al
    python main.py

Render / sunucuda çalıştırma (Start Command):
    gunicorn main:app
"""

import os
import json
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

# ---------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

app = Flask(__name__)
CORS(app)  # Netlify'deki arayüzün çapraz kaynaktan erişebilmesi için

# İstemci GEMINI_API_KEY / GOOGLE_API_KEY ortam değişkenini otomatik okur
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()

# ---------------------------------------------------------------
# Komut kütüphanesi
# Yeni komut eklemek için buraya bir kayıt ekle; uygulama tarafı
# dönen "command" + "params" değerine göre aksiyonu çalıştırır.
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


# Basit oturum belleği (production'da oturum bazlı yapılmalı)
_history = []
MAX_TURNS = 10


def ask_gemini(user_text: str) -> dict:
    global _history
    _history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    _history = _history[-MAX_TURNS * 2:]

    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_history,
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            response_mime_type="application/json",
            temperature=0.4,
            max_output_tokens=1024,
        ),
    )
    raw = (resp.text or "").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"type": "chat", "speech": raw or "Bir hata oldu efendim, tekrar dener misiniz?"}

    _history.append(types.Content(role="model", parts=[types.Part(text=raw)]))
    return result


# ---------------------------------------------------------------
# Uçlar
# ---------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Jarvis çevrimiçi, efendim."})


@app.route("/api/text", methods=["POST"])
def text():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"error": "'text' alanı boş"}), 400
    try:
        result = ask_gemini(user_text)
    except Exception as e:
        return jsonify({"type": "chat", "speech": "Beynimle bağlantı kuramadım efendim.", "error": str(e)}), 502
    result["transcript"] = user_text
    return jsonify(result)


@app.route("/", methods=["GET"])
def root():
    return jsonify({"name": "JARVIS Backend", "model": GEMINI_MODEL, "endpoints": ["/api/health", "/api/text"]})


if __name__ == "__main__":
    if not GEMINI_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
        print("[UYARI] GEMINI_API_KEY ayarlanmamış! aistudio.google.com'dan alıp ayarlayın.")
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
