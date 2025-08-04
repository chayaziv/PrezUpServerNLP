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

def load_prompt_from_file():
    """טוען את הפרומפט מקובץ prompt.md"""
    try:
        with open('prompt.md', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("Warning: prompt.md file not found. Using default prompt.")
        return """
        נתח את קובץ הווידאו המצורף, המכיל הסרטה של פרזנטציה. 
        ספק משוב מפורט בצורת JSON בלבד ללא כל טקסט נוסף.
        """

# טעינת הפרומפט מהקובץ
prompt = load_prompt_from_file()


def encode_file_to_base64(file_path):
    """ ממיר קובץ לבסיס 64 """
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def analyze_presentation(video_file_path):
    client = genai.Client(api_key=gemini_api_key)
    model = "gemini-2.0-flash"
    encoded_video = encode_file_to_base64(video_file_path)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="video/mp4",
                        data=encoded_video,
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
        print(chunk.text)

    return response_text


app = Flask(__name__)
CORS(app) 
import requests


app = Flask(__name__)

@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    print("---------------------analyze_video---------------------")
    data = request.get_json()
    file_url = data.get("videoUrl")
    
    if not file_url:
        return jsonify({"error": "No video URL provided"}), 400
    
    temp_path = "temp_video.mp4"
    
    try:
        # הורדת הקובץ מ-S3
        response = requests.get(file_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download video file"}), 500
        
        with open(temp_path, "wb") as video_file:
            video_file.write(response.content)
        
        # ניתוח הווידאו
        analysis_result = analyze_presentation(temp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return analysis_result


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)