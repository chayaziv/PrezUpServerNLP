import base64
from google import genai
from google.genai import types

from flask import Flask, request, jsonify
import os
from flask_cors import CORS  
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

prompt = """

נתח את קובץ ה-WAV המצורף, המכיל הקלטה של פרזנטציה. ספק משוב מפורט בצורת JSON בלבד, 
נתח את קובץ ה WAV המכיל הקלטה של פרנזטציה
ספק משוב מפורט בצורת JSON .בלבד ללא כל טקסט נוסף

המשוב צריך להתמקד בחמישה מדדים עיקריים .
1. בהירות (clarity) – עד כמה הרעיונות מובנים וברורים?
2.  שטף דיבור (fluency)  – האם הדובר מדבר באופן טבעי וללא היסוסים?
3.  ביטחון (confidence)  – האם הדובר משדר ביטחון?
4.  מעורבות (engagement)  – עד כמה הפרזנטציה מעניינת ומושכת?
5.  סגנון דיבור (speech style)  – איך הדובר משתמש בשפה? האם היא טבעית ומכובדת?
והסבר מפורט על הסיבות לציון זה

בנוסף, ספק טיפים לשיפור הפרזנטציה. הטיפים צריכים להיות ספציפיים ומעשיים, ולכלול גם משוב חיובי על נקודות החוזק של המציג.

הפלט צריך להיות בפורמט JSON הבא בדיוק:
{
    "scores": {
        "clarity": {
            "score": "",
            "reason": ""
        },
        "fluency": {
            "score": "",
            "reason": ""
        },
        "confidence": {
            "score": "",
            "reason": ""
        },
        "engagement": {
            "score": "",
            "reason": ""
        },
        "speech_style": {
            "score": "",
            "reason": ""
        }
    },
    "tips": ""
}

 כולל ציון מספרי ונימוק לכל מדד, וכן טיפים מפורטים לשיפור 
 הטיפים יהיו פסקה אחת באורך של 4 עד 6 שורות
 קפד למלא את כל השדות ב-JSON וליהות חיובי במדד 8 ומעלה
 תענה הכול בשפת העברית בלבד
 """


def encode_file_to_base64(file_path):
    """ ממיר קובץ לבסיס 64 """
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def analyze_presentation(audio_file_path):
    client = genai.Client(api_key=gemini_api_key)
    model = "gemini-2.0-flash"
    encoded_audio = encode_file_to_base64(audio_file_path)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="audio/wav",
                        data=encoded_audio,
                    )
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=generate_content_config
    ):
        response_text += chunk.text
       

    return response_text


app = Flask(__name__)
CORS(app) 

@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files["audio"]
    temp_path = "temp_audio.wav"
    audio_file.save(temp_path)
    
    try:
        analysis_result = analyze_presentation(temp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)
    return {"response":analysis_result}

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
