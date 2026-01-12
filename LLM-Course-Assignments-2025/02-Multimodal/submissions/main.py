from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import base64
import ollama
from PIL import Image
import io
import cv2  # 新增：视频处理库
import numpy as np
from datetime import datetime

app = FastAPI()

# 挂载静态文件目录（存放前端页面）
app.mount("/static", StaticFiles(directory="static"), name="static")


# 新增：视频抽帧函数（抽取关键帧，避免分析所有帧）
def extract_video_frames(video_data, frame_interval=10, max_frames=10):
    """
    从视频中抽取关键帧
    :param video_data: 视频二进制数据
    :param frame_interval: 每隔多少帧抽1帧
    :param max_frames: 最多抽取多少帧（避免卡顿）
    :return: 抽取的帧列表（PIL Image格式）
    """
    # 将视频数据写入临时内存
    temp_video = io.BytesIO(video_data)
    temp_video.seek(0)

    # 保存为临时文件（cv2需要文件路径）
    temp_filename = f"temp_video_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
    with open(temp_filename, 'wb') as f:
        f.write(temp_video.read())

    # 用cv2读取视频
    cap = cv2.VideoCapture(temp_filename)
    frames = []
    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔frame_interval帧抽取一帧
        if frame_count % frame_interval == 0:
            # 转换颜色空间（cv2默认BGR，PIL是RGB）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转PIL Image
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
            extracted_count += 1

        frame_count += 1

    # 释放资源
    cap.release()
    # 删除临时文件
    import os
    os.remove(temp_filename)

    return frames


# 新增：汇总多帧分析结果
def summarize_frame_results(frame_results):
    """汇总多个帧的分析结果，生成视频整体描述"""
    if not frame_results:
        return "未抽取到视频帧，无法分析"

    # 统计高频信息
    ground_types = []  # 地面类型
    main_objects = []  # 主要物体
    scene_types = []  # 场景类型
    hide_places = []  # 隐蔽位置

    for result in frame_results:
        # 解析每帧的结果（按行拆分）
        lines = result.strip().split('\n')
        for line in lines:
            if "地面类型" in line:
                ground_types.append(line.split('：')[-1].strip())
            elif "主要物体" in line:
                main_objects.append(line.split('：')[-1].strip())
            elif "场景类型" in line:
                scene_types.append(line.split('：')[-1].strip())
            elif "隐蔽位置" in line:
                hide_places.append(line.split('：')[-1].strip())

    # 取出现次数最多的信息（简单统计）
    def get_most_common(lst):
        if not lst:
            return "未识别"
        return max(set(lst), key=lst.count)

    # 生成汇总结果
    summary = (
            f"视频整体环境分析（基于{len(frame_results)}帧）：\n"
            f"1. 主要地面类型：{get_most_common(ground_types)}\n"
            f"2. 主要物体：{get_most_common(main_objects)}\n"
            f"3. 整体场景类型：{get_most_common(scene_types)}\n"
            f"4. 主要隐蔽位置：{get_most_common(hide_places)}\n"
            f"\n各帧细节：\n" + "\n---\n".join([f"帧{idx + 1}：{res}" for idx, res in enumerate(frame_results)])
    )
    return summary


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # 返回前端页面
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    # 读取上传的文件数据
    file_data = await file.read()
    file_type = file.content_type  # 获取文件类型（image/xxx 或 video/xxx）

    try:
        # 1. 处理图片
        if file_type.startswith("image"):
            image = Image.open(io.BytesIO(file_data))
            # 转base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # 构造提示词
            prompt = (
                "请分析这张图片的环境，并列出以下内容：\n"
                "1. 主要地面类型（如草地、水泥地、沙地等）\n"
                "2. 主要物体（如墙体、车辆、树木、建筑物等）\n"
                "3. 整体场景类型（如城市、野外、森林、军事基地等）\n"
                "4. 可能存在的隐蔽位置（如墙角、树林、车辆后方等）\n"
                "5. 是否发现穿迷彩服的人员或军事相关装备？\n"
                "请用中文回答，尽量简洁明了。"
            )

            # 调用LLaVA
            response = ollama.chat(
                model="llava:7b",
                messages=[{'role': 'user', 'content': prompt, 'images': [img_base64]}]
            )
            result = response['message']['content']
            return JSONResponse(content={"type": "image", "result": result})

        # 2. 处理视频
        elif file_type.startswith("video"):
            # 抽取视频关键帧
            frames = extract_video_frames(file_data, frame_interval=10, max_frames=10)
            if not frames:
                return JSONResponse(content={"error": "无法抽取视频帧"}, status_code=400)

            # 分析每帧
            frame_results = []
            for frame in frames:
                # 帧转base64
                buffered = io.BytesIO()
                frame.save(buffered, format="PNG")
                frame_base64 = base64.b64encode(buffered.getvalue()).decode()

                # 调用LLaVA分析单帧
                prompt = (
                    "请分析这一视频帧的环境，并列出以下内容：\n"
                    "1. 主要地面类型（如草地、水泥地、沙地等）\n"
                    "2. 主要物体（如墙体、车辆、树木、建筑物等）\n"
                    "3. 整体场景类型（如城市、野外、森林、军事基地等）\n"
                    "4. 可能存在的隐蔽位置（如墙角、树林、车辆后方等）\n"
                    "5. 是否发现穿迷彩服的人员或军事相关装备？\n"
                    "请用中文回答，尽量简洁明了。"
                )
                response = ollama.chat(
                    model="llava:7b",
                    messages=[{'role': 'user', 'content': prompt, 'images': [frame_base64]}]
                )
                frame_results.append(response['message']['content'])

            # 汇总结果
            summary = summarize_frame_results(frame_results)
            return JSONResponse(content={"type": "video", "result": summary})

        # 3. 不支持的文件类型
        else:
            return JSONResponse(content={"error": "仅支持图片和视频文件"}, status_code=400)

    except Exception as e:
        return JSONResponse(content={"error": f"处理失败：{str(e)}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    # 改端口（避免8000被占用，用8001）
    uvicorn.run(app, host="0.0.0.0", port=8001)