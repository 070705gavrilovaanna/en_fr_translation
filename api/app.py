from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
model.load_state_dict(torch.load('best_model_t5.pth', map_location=device))
model.eval()

def translate(text, max_len=32):
    input_text = text
    
    inputs = tokenizer(input_text, return_tensors='pt', max_length=32, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_len)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

app = FastAPI()

html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Переводчик EN → FR</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }}
        .container {{ background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); padding: 40px; max-width: 600px; width: 100%; }}
        h1 {{ color: #333; text-align: center; margin-bottom: 10px; }}
        .sub {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }}
        textarea {{ width: 100%; padding: 15px; font-size: 16px; border: 2px solid #e0e0e0; border-radius: 10px; resize: vertical; font-family: inherit; }}
        textarea:focus {{ outline: none; border-color: #667eea; }}
        button {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; font-size: 16px; border-radius: 25px; cursor: pointer; width: 100%; margin-top: 15px; transition: transform 0.2s; }}
        button:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }}
        .result {{ margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #667eea; }}
        .result-label {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
        .result-text {{ font-size: 18px; color: #667eea; word-wrap: break-word; }}
        hr {{ margin: 20px 0; border: none; height: 1px; background: #e0e0e0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Переводчик EN → FR</h1>
        <div class="sub">английский → французский (обученная модель)</div>
        <form method="post">
            <textarea name="text" rows="4" placeholder="Введите текст на английском..."></textarea>
            <button type="submit">Перевести</button>
        </form>
        {result}
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(html_form.format(result=""))

@app.post("/", response_class=HTMLResponse)
def predict(text: str = Form(...)):
    translated = translate(text)
    result_html = f"""
    <div class="result">
        <div class="result-label">Английский:</div>
        <div class="result-text">{text}</div>
        <hr>
        <div class="result-label">Французский:</div>
        <div class="result-text">{translated}</div>
    </div>
    """
    return HTMLResponse(html_form.format(result=result_html))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)