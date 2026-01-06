from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import base64
import ollama
from PIL import Image
import io

app = FastAPI()

# 挂载静态文件目录（存放前端页面）
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # 返回前端页面
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 读取上传的图片
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 将图片转成 base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # 构造提示词，要求模型识别环境
    prompt = (
        "请分析这张图片的环境，并列出以下内容：\n"
        "1. 主要地面类型（如草地、水泥地、沙地等）\n"
        "2. 主要物体（如墙体、车辆、树木、建筑物等）\n"
        "3. 整体场景类型（如城市、野外、森林、军事基地等）\n"
        "4. 可能存在的隐蔽位置（如墙角、树林、车辆后方等）\n"
        "请用中文回答，尽量简洁明了。"
    )

    # 调用 Ollama 本地模型
    try:
        response = ollama.chat(
            model="llava:7b",
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img_base64]
                }
            ]
        )
        result = response['message']['content']
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)