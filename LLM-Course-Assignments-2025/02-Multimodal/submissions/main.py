from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import base64
import ollama
import utils
import os
import io
import logging
import uuid
import asyncio
import json

# ==================== 基础配置 ====================
# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局进度缓存（key: file_id，value: 进度信息）
progress_cache = {}
# 全局结果缓存（key: file_id，value: 分析结果+标注图片）
result_cache = {}

# 实验参数（大模型优化配置）
EXPERIMENT_CONFIG = {
    "max_frames": 10,
    "frame_diff_threshold": 30,
    "use_yolo": True,
    "llava_model": "llava:7b",  # 轻量化模型，速度更快
    "llava_params": {
        "temperature": 0.5,
        "top_k": 30,
        "max_tokens": 500
    }
}

# FastAPI实例
app = FastAPI(title="军事场景识别Demo（大模型优化版）")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== 异步处理文件（整合所有优化） ====================
async def process_file_async(file_id: str, file_data: bytes, file_name: str):
    """异步处理文件（更新进度+大模型优化）"""
    try:
        # 1. 初始化进度
        progress_cache[file_id] = {
            "step": "开始处理",
            "progress": 0,
            "status": "running"
        }

        # 2. 判断文件类型
        file_ext = os.path.splitext(file_name)[1].lower()
        is_image = file_ext in {'.jpg', '.jpeg', '.png'}
        is_video = file_ext in {'.mp4', '.avi', '.mov'}

        # ========== 处理图片 ==========
        if is_image:
            progress_cache[file_id] = {"step": "分析图片", "progress": 30, "status": "running"}
            # 读取图片
            image = utils.Image.open(io.BytesIO(file_data))
            # YOLO检测
            progress_cache[file_id] = {"step": "检测士兵", "progress": 50, "status": "running"}
            yolo_result = utils.detect_soldier_with_yolo(image)
            # LLaVA分析（带缓存+重试+标准化）
            progress_cache[file_id] = {"step": "分析军事场景", "progress": 80, "status": "running"}
            try:  # 新增：单独捕获推理异常
                llava_parsed = await utils.infer_frame_with_cache(image, yolo_result, EXPERIMENT_CONFIG["llava_model"])
            except Exception as e:
                logger.error(f"LLaVA推理失败：{e}")
                # 推理失败时填充兜底结果
                llava_parsed = {
                    "ground_type": "草地",
                    "scene": "草地隐蔽区",
                    "hide_place": "树林深处",
                    "soldier_state": "移动",
                    "risk": "高"
                }
                # 立即更新进度为失败
                progress_cache[file_id] = {
                    "step": f"推理失败：{str(e)[:50]}",
                    "progress": 80,
                    "status": "failed",
                    "error": f"LLaVA推理失败：{str(e)}"
                }
                return  # 终止处理

            # 可视化标注
            annotated_img = utils.draw_annotation(image.copy(), yolo_result, llava_parsed)
            # 生成最终结果
            final_result = (
                f"【YOLO士兵检测】\n{yolo_result}\n\n【军事场景分析】\n"
                f"地面类型：{llava_parsed['ground_type']}\n"
                f"场景细分：{llava_parsed['scene']}\n"
                f"隐蔽点：{llava_parsed['hide_place']}\n"
                f"士兵状态：{llava_parsed['soldier_state']}\n"
                f"风险等级：{llava_parsed['risk']}"
            )
            # 缓存结果
            result_cache[file_id] = {
                "type": "image",
                "result": final_result,
                "annotated_img": annotated_img,
                "llava_parsed": llava_parsed
            }
            # 更新进度
            progress_cache[file_id] = {"step": "分析完成", "progress": 100, "status": "success"}

        # ========== 处理视频 ==========
        elif is_video:
            progress_cache[file_id] = {"step": "智能抽帧", "progress": 20, "status": "running"}
            # 智能抽帧
            frames = utils.extract_smart_frames(
                file_data,
                max_frames=EXPERIMENT_CONFIG["max_frames"],
                threshold=EXPERIMENT_CONFIG["frame_diff_threshold"]
            )
            progress_cache[file_id] = {"step": f"抽帧完成（共{len(frames)}帧）", "progress": 30, "status": "running"}

            # YOLO检测所有帧
            progress_cache[file_id] = {"step": "批量检测士兵", "progress": 40, "status": "running"}
            yolo_results = [utils.detect_soldier_with_yolo(frame) for frame in frames]

            # 批量异步推理（大模型优化）
            progress_cache[file_id] = {"step": "批量分析军事场景", "progress": 50, "status": "running"}
            try:  # 新增：单独捕获视频推理异常
                llava_results = await utils.batch_llava_inference(frames, yolo_results,
                                                                  EXPERIMENT_CONFIG["llava_model"])
            except Exception as e:
                logger.error(f"视频LLaVA推理失败：{e}")
                # 填充兜底结果
                llava_results = [{
                    "ground_type": "草地",
                    "scene": "草地隐蔽区",
                    "hide_place": "树林深处",
                    "soldier_state": "移动",
                    "risk": "高"
                } for _ in frames]
                # 更新进度为失败
                progress_cache[file_id] = {
                    "step": f"推理失败：{str(e)[:50]}",
                    "progress": 50,
                    "status": "failed",
                    "error": f"LLaVA推理失败：{str(e)}"
                }
                return  # 终止处理

            # 生成帧结果+标注
            frame_results = []
            for idx, (frame, yolo_res, llava_res) in enumerate(zip(frames, yolo_results, llava_results)):
                progress = 50 + (idx / len(frames)) * 30
                progress_cache[file_id] = {
                    "step": f"标注帧{idx + 1}/{len(frames)}",
                    "progress": int(progress),
                    "status": "running"
                }
                # 标注图片
                annotated_frame = utils.draw_annotation(frame.copy(), yolo_res, llava_res)
                frame_results.append({
                    "yolo": yolo_res,
                    "llava_parsed": llava_res,
                    "frame": frame,
                    "annotated_frame": annotated_frame
                })

            # 汇总结果
            progress_cache[file_id] = {"step": "汇总结果", "progress": 90, "status": "running"}
            final_summary = utils.summarize_frame_results(frame_results)
            # 缓存结果（取第一帧标注图作为预览）
            result_cache[file_id] = {
                "type": "video",
                "result": final_summary,
                "annotated_img": frame_results[0]["annotated_frame"] if frame_results else None,
                "frame_results": frame_results
            }
            # 更新进度
            progress_cache[file_id] = {"step": "分析完成", "progress": 100, "status": "success"}

        # ========== 不支持的文件类型 ==========
        else:
            progress_cache[file_id] = {
                "step": "不支持的文件类型",
                "progress": 0,
                "status": "failed",
                "error": f"仅支持jpg/png/mp4/avi/mov，当前文件后缀：{file_ext}"
            }

    except Exception as e:
        logger.error(f"处理文件失败：{e}", exc_info=True)
        progress_cache[file_id] = {
            "step": "处理失败",
            "progress": 0,
            "status": "failed",
            "error": str(e)
        }

# ==================== 核心接口 ====================
# 根路径：返回前端页面
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取前端页面失败：{e}")
        return HTMLResponse(content="<h1>页面加载失败，请检查static/index.html是否存在</h1>")


# 上传文件：启动异步处理+返回file_id
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_data = await file.read()
    file_id = str(uuid.uuid4())  # 生成唯一文件ID
    # 启动异步处理
    asyncio.create_task(process_file_async(file_id, file_data, file.filename or "未知文件"))
    return JSONResponse(content={
        "code": 200,
        "msg": "开始分析，可轮询进度",
        "file_id": file_id
    })


# 查询进度接口
@app.get("/progress/{file_id}")
async def get_progress(file_id: str):
    return JSONResponse(content=progress_cache.get(file_id, {
        "step": "未找到任务",
        "progress": 0,
        "status": "not_found"
    }))


# 获取分析结果接口
@app.get("/result/{file_id}")
async def get_result(file_id: str):
    if file_id not in result_cache:
        return JSONResponse(content={"code": 404, "msg": "结果未生成"}, status_code=404)
    result = result_cache[file_id]
    # 标注图转Base64（前端预览）
    annotated_base64 = ""
    if result.get("annotated_img"):
        buffered = io.BytesIO()
        result["annotated_img"].save(buffered, format="PNG")
        annotated_base64 = base64.b64encode(buffered.getvalue()).decode()
    return JSONResponse(content={
        "code": 200,
        "type": result["type"],
        "result_text": result["result"],
        "annotated_img": annotated_base64
    })


# 导出结果接口（支持txt/图片）
@app.get("/export/{file_id}")
async def export_result(file_id: str, format: str = "txt"):
    if file_id not in result_cache:
        return JSONResponse(content={"code": 404, "msg": "结果未生成"}, status_code=404)
    result = result_cache[file_id]

    # 导出TXT
    if format == "txt":
        return Response(
            content=result["result"],
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={file_id}_result.txt"}
        )
    # 导出标注图片
    elif format == "image":
        if not result.get("annotated_img"):
            return JSONResponse(content={"code": 400, "msg": "无标注图片"}, status_code=400)
        buffered = io.BytesIO()
        result["annotated_img"].save(buffered, format="PNG")
        return Response(
            content=buffered.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={file_id}_annotated.png"}
        )
    else:
        return JSONResponse(content={"code": 400, "msg": "仅支持txt/image格式"}, status_code=400)


# ==================== 启动服务 ====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")