import os
from google import genai

os.environ["GEMINI_API_KEY"] = "AIzaSyCW5Gy6lFdVH8jHTkVLNTHJNNviy_LgZGo"
try:
    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='hello'
    )
    print("Success 2.5:", response.text)
except Exception as e:
    print("Error with 2.5:", e)

try:
    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents='hello'
    )
    print("Success 3.0:", response.text)
except Exception as e:
    print("Error with 3.0:", e)
