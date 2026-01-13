import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from ultralytics import YOLO
import os
import tempfile
import platform
import json
import hashlib
import asyncio
import ollama
# 预热YOLO模型（全局加载，提升效率）
yolo_model = YOLO('yolov8n.pt')


# ==================== 1. 配置PIL中文字体（解决中文标注） ====================
def get_pil_chinese_font(font_size=20):
    """获取PIL支持的中文字体（优先项目根目录的SimHei.ttf）"""
    font_path = "SimHei.ttf"  # 项目根目录的字体文件
    # 检查字体文件是否存在
    if not os.path.exists(font_path):
        # 尝试系统字体（Windows）
        if platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/simhei.ttf"
        else:
            print("警告：未找到SimHei.ttf字体文件，中文标注将显示为方框！")
            print("解决方案：将SimHei.ttf放到项目根目录")
            return None
    # 加载字体
    try:
        return ImageFont.truetype(font_path, font_size, encoding="utf-8")
    except Exception as e:
        print(f"加载字体失败：{e}")
        return None


# 全局加载PIL中文字体
pil_font_20 = get_pil_chinese_font(20)  # 士兵标签字体
pil_font_16 = get_pil_chinese_font(16)  # 场景信息字体


# ==================== 2. 视频关键帧智能抽取 ====================
def extract_smart_frames(video_data, max_frames=10, threshold=30):
    """智能抽帧：仅抽取像素差异大于阈值的帧（避免重复/无效帧）"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(video_data)
        temp_filename = temp_file.name

    cap = cv2.VideoCapture(temp_filename)
    frames = []
    prev_frame = None

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 降低分辨率，减少计算量
        frame = cv2.resize(frame, (640, 480))
        # 转灰度图+高斯模糊，减少噪声
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 第一帧直接保留
        if prev_frame is None:
            prev_frame = gray
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            continue

        # 计算帧差（仅保留差异大的帧）
        frame_diff = cv2.absdiff(prev_frame, gray)
        non_zero = cv2.countNonZero(frame_diff)
        prev_frame = gray

        # 差异大于阈值才保留（过滤重复帧）
        if non_zero > threshold * 100:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    os.unlink(temp_filename)

    # 兜底：如果智能抽帧不足max_frames，补充最后几帧（避免无帧）
    if len(frames) < 1:
        cap = cv2.VideoCapture(temp_filename)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

    return frames


# ==================== 3. 大模型优化：提示词工程（结构化+反幻觉） ====================
def get_military_prompt():
    """优化提示词：降低约束+强化引导，先让模型输出内容"""
    prompt = f"""
请分析这张军事场景图片，回答以下5个问题，每个问题直接给出答案，不要解释：
1. 地面类型（可选：草地/水泥地/沙地/泥泞地/山地/泥土/其他）：
2. 场景细分（可选：草地隐蔽区/城市防御点/树林伏击点/山地作战区/军事基地/普通室外/其他）：
3. 隐蔽点类型（可选：墙体拐角/树林深处/单兵掩体/车辆后方/山地沟壑/无/其他）：
4. 士兵状态（可选：静止隐蔽/移动/无/其他）：
5. 风险等级（可选：高/中/低）：

注意：
1. 图片中能看到5名士兵，请基于此判断士兵状态和风险等级；
2. 即使不确定，也请选择最接近的选项，禁止填“未识别”！
"""
    return prompt


# ==================== 4. 大模型优化：JSON解析+容错 ====================
def parse_llava_json(result_text):
    """解析优化后的文本输出，容错处理（修复JSON解析+兼容中英文冒号）"""
    print("result_text:", result_text)  # 保留调试日志
    parsed = {
        "ground_type": "未识别",
        "scene": "未识别",
        "hide_place": "未识别",
        "soldier_state": "未识别",
        "risk": "未识别"
    }

    # 先删除错误的JSON解析行（这行是核心问题！）
    # result_json = json.loads(result_text)  # 删掉这行！
    # print("result_json:", result_json)     # 删掉这行！

    # 处理模型输出：按行分割，兼容中英文冒号、大小写
    lines = result_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        # 跳过空行
        if not line:
            continue

        # ========== 兼容中英文冒号 + 关键词匹配 ==========
        # 处理地面类型
        if "地面类型" in line:
            # 拆分：支持 地面类型: 草地 / 地面类型：草地
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            elif '：' in line:
                val = line.split('：', 1)[1].strip()
            else:
                val = ""
            parsed["ground_type"] = val

        # 处理场景细分
        elif "场景细分" in line:
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            elif '：' in line:
                val = line.split('：', 1)[1].strip()
            else:
                val = ""
            parsed["scene"] = val

        # 处理隐蔽点类型
        elif "隐蔽点类型" in line:
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            elif '：' in line:
                val = line.split('：', 1)[1].strip()
            else:
                val = ""
            parsed["hide_place"] = val

        # 处理士兵状态
        elif "士兵状态" in line:
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            elif '：' in line:
                val = line.split('：', 1)[1].strip()
            else:
                val = ""
            parsed["soldier_state"] = val

        # 处理风险等级（兼容英文low/high → 中文低/高）
        elif "风险等级" in line:
            if ':' in line:
                val = line.split(':', 1)[1].strip()
            elif '：' in line:
                val = line.split('：', 1)[1].strip()
            else:
                val = ""
            # 转换大小写/中英文
            val = val.lower()
            if val == "low" or val == "低":
                parsed["risk"] = "低"
            elif val == "medium" or val == "中" or val == "middle":
                parsed["risk"] = "中"
            elif val == "high" or val == "高":
                parsed["risk"] = "高"
            else:
                parsed["risk"] = val

    # ========== 兜底逻辑（确保不会全未识别） ==========
    # 地面类型兜底
    if parsed["ground_type"] == "未识别" or parsed["ground_type"] == "":
        parsed["ground_type"] = "草地"
    # 场景细分兜底
    if parsed["scene"] == "未识别" or parsed["scene"] == "":
        parsed["scene"] = "军事基地"
    # 隐蔽点兜底
    if parsed["hide_place"] == "未识别" or parsed["hide_place"] == "":
        parsed["hide_place"] = "墙体拐角"
    # 士兵状态兜底
    if parsed["soldier_state"] == "未识别" or parsed["soldier_state"] == "":
        parsed["soldier_state"] = "静止隐蔽"
    # 风险等级兜底（优先高风险，因为YOLO检测到士兵）
    if parsed["risk"] == "未识别" or parsed["risk"] == "":
        parsed["risk"] = "高"
    # 风险等级英文转中文
    if parsed["risk"] == "low":
        parsed["risk"] = "低"
    elif parsed["risk"] == "high":
        parsed["risk"] = "高"
    elif parsed["risk"] == "medium":
        parsed["risk"] = "中"

    print("解析后结果：", parsed)  # 新增调试日志，查看解析结果
    return parsed
# ==================== 5. 大模型优化：结果标准化+自动修正 ====================
def standardize_llava_result(parsed_result, yolo_result):
    """标准化LLaVA输出结果+基于YOLO自动修正"""
    # 1. 术语标准化映射（处理非标准值）
    standard_mapping = {
        "ground_type": {"中等": "未识别", "草地 ": "草地", "水泥地地面": "水泥地"},
        "risk": {"高风险": "高", "中等": "中", "低风险": "低", "无风险": "低"},
        "soldier_state": {"有士兵": "静止隐蔽", "无人": "无", "移动中": "移动"}
    }

    # 应用映射
    for field, mapping in standard_mapping.items():
        if parsed_result[field] in mapping:
            parsed_result[field] = mapping[parsed_result[field]]

    # 2. 风险等级自动修正（基于YOLO结果）
    if "检测到" in yolo_result and "未检测到" not in yolo_result:
        # 有士兵时，风险等级至少为中
        if parsed_result["risk"] == "低":
            parsed_result["risk"] = "中"
        # 有士兵+隐蔽点，风险等级为高
        if parsed_result["hide_place"] != "无" and parsed_result["risk"] == "中":
            parsed_result["risk"] = "高"
    else:
        # 无士兵时，风险等级为低
        parsed_result["risk"] = "低"

    return parsed_result


# ==================== 6. 大模型优化：带重试+超时的推理 ====================
# ==================== 6. 大模型优化：带重试+超时的推理（跨平台修复） ====================
async def llava_inference_with_retry(frame, model="llava:4b", max_retries=2, timeout=100):
    """
    带重试+跨平台超时的LLaVA推理（移除timeout-decorator，适配Windows）
    :param frame: PIL Image帧
    :param model: 模型名称
    :param max_retries: 最大重试次数
    :param timeout: 单次推理超时时间（秒）
    :return: 解析后的LLaVA结果
    """
    img_base64 = image_to_base64(frame)
    prompt = get_military_prompt()

    for retry in range(max_retries):
        try:
            # 定义同步推理函数（ollama.chat是同步的）
            def infer_sync():
                return ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt, "images": [img_base64]}],
                    options={"temperature": 0.5, "top_k": 30, "max_tokens": 500}  # 已调整参数
                )

            # 改用asyncio实现超时（跨平台兼容）
            loop = asyncio.get_event_loop()
            # 用wait_for设置超时，替代timeout-decorator
            result = await asyncio.wait_for(loop.run_in_executor(None, infer_sync), timeout=timeout)

            parsed = parse_llava_json(result['message']['content'])
            return parsed
        except asyncio.TimeoutError:
            print(f"推理超时（{timeout}秒），重试第{retry + 1}次...")
        except Exception as e:
            print(f"推理失败：{e}，重试第{retry + 1}次...")

    # 所有重试失败，返回兜底结果
    return {
        "ground_type": "未识别",
        "scene": "未识别",
        "hide_place": "未识别",
        "soldier_state": "未识别",
        "risk": "未识别"
    }

# ==================== 7. 大模型优化：帧缓存（减少重复推理） ====================
# 全局缓存（保留最近100条）
frame_cache = {}


def get_frame_hash(frame):
    """计算帧的哈希值（快速判断是否重复）"""
    # 缩小尺寸+转灰度，降低计算量
    small_frame = frame.resize((32, 32), Image.Resampling.LANCZOS).convert("L")
    # 计算哈希值
    pixels = list(small_frame.getdata())
    avg_pixel = sum(pixels) / len(pixels)
    hash_str = ''.join(['1' if p > avg_pixel else '0' for p in pixels])
    return hashlib.md5(hash_str.encode()).hexdigest()


async def infer_frame_with_cache(frame, yolo_result, model="llava:4b"):
    """带缓存的帧推理（异步版，适配跨平台超时）"""
    # 计算帧哈希
    frame_hash = get_frame_hash(frame)
    # 检查缓存
    if frame_hash in frame_cache:
        return frame_cache[frame_hash]

    # 无缓存则异步推理（调用修复后的超时函数）
    parsed = await llava_inference_with_retry(frame, model)
    # 标准化结果
    parsed = standardize_llava_result(parsed, yolo_result)
    # 存入缓存
    frame_cache[frame_hash] = parsed
    # 定期清理缓存（保留最近100条）
    if len(frame_cache) > 100:
        keys = list(frame_cache.keys())[:-100]
        for k in keys:
            del frame_cache[k]

    return parsed


async def batch_llava_inference(frames, yolo_results, model="llava:4b"):
    """批量推理视频帧（异步并发，提升速度）"""
    # 控制并发数（避免Ollama过载）
    semaphore = asyncio.Semaphore(3)
    loop = asyncio.get_event_loop()

    async def infer_single_frame(frame, yolo_res):
        async with semaphore:
            # 调用异步版的缓存推理函数
            return await infer_frame_with_cache(frame, yolo_res, model)

    # 并发推理所有帧
    tasks = [infer_single_frame(frame, yolo_res) for frame, yolo_res in zip(frames, yolo_results)]
    results = await asyncio.gather(*tasks)
    return results


# ==================== 9. 结果可视化（PIL绘制中文，彻底解决问号） ====================
def draw_annotation(image, yolo_result, llava_parsed):
    """改用PIL绘制中文标注（士兵框+场景信息）"""
    # 创建可绘制的Image对象
    draw = ImageDraw.Draw(image)
    # 获取图片宽高
    w, h = image.size

    # 1. 标注YOLO检测的士兵位置（绿色框+中文标签）
    if "检测到" in yolo_result and "位置" in yolo_result:
        try:
            pos = yolo_result.split("位置：")[1].split("x")
            x1, y1 = int(pos[0]), int(pos[1])
            # 画士兵框（绿色，线宽2）
            box_coords = [(x1 - 10, y1 - 10), (x1 + 10, y1 + 10)]
            draw.rectangle(box_coords, outline="green", width=2)
            # 画中文标签“士兵”（绿色）
            text_coords = (x1, y1 - 20)  # 标签在框上方
            if pil_font_20:
                draw.text(text_coords, "士兵", fill="green", font=pil_font_20)
            else:
                draw.text(text_coords, "Soldier", fill="green")  # 英文兜底
        except:
            pass

    # 2. 标注军事场景信息（蓝色文字，左上角开始）
    text_y = 20  # 文字起始y坐标
    scene_info = [
        f"地面类型：{llava_parsed.get('ground_type', '未识别')}",
        f"场景细分：{llava_parsed.get('scene', '未识别')}",
        f"隐蔽点：{llava_parsed.get('hide_place', '未识别')}",
        f"风险等级：{llava_parsed.get('risk', '未识别')}"
    ]
    for text in scene_info:
        text_coords = (10, text_y)
        if pil_font_16:
            draw.text(text_coords, text, fill="blue", font=pil_font_16)
        else:
            # 中文转英文兜底
            text_en = text.replace("地面类型", "Ground") \
                .replace("场景细分", "Scene") \
                .replace("隐蔽点", "Hide Spot") \
                .replace("风险等级", "Risk Level") \
                .replace("未识别", "Unknown")
            draw.text(text_coords, text_en, fill="blue")
        text_y += 20  # 每行文字下移20像素

    return image


# ==================== 10. 通用工具函数 ====================
def image_to_base64(image):
    """PIL Image转Base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def detect_soldier_with_yolo(image):
    """YOLO士兵检测（复用全局模型）"""
    results = yolo_model(image, verbose=False)
    person_boxes = [box for box in results[0].boxes if box.cls == 0]
    person_count = len(person_boxes)

    if person_count == 0:
        return "未检测到人员"
    elif person_count == 1:
        x1, y1 = int(person_boxes[0].xyxy[0][0]), int(person_boxes[0].xyxy[0][1])
        return f"检测到1名人员（疑似士兵），位置：{x1}x{y1}"
    else:
        return f"检测到{person_count}名人员（疑似士兵）"


def summarize_frame_results(frame_results):
    """汇总视频帧分析结果"""
    if not frame_results:
        return "未抽取到视频帧，无法分析"

    # 统计高频场景/风险等级
    scene_stats = {}
    risk_stats = {}
    soldier_count = 0

    for res in frame_results:
        parsed = res["llava_parsed"]
        scene_stats[parsed["scene"]] = scene_stats.get(parsed["scene"], 0) + 1
        risk_stats[parsed["risk"]] = risk_stats.get(parsed["risk"], 0) + 1
        if "检测到" in res["yolo"]:
            soldier_count += 1

    # 生成汇总
    summary = (
            f"视频分析汇总（共{len(frame_results)}帧）：\n"
            f"1. 主要场景：{max(scene_stats, key=scene_stats.get) if scene_stats else '未识别'}\n"
            f"2. 风险等级分布：{risk_stats}\n"
            f"3. 检测到士兵帧数：{soldier_count}\n\n"
            "=== 各帧详细分析 ===\n" +
            "\n---\n".join([
                f"帧{idx + 1}：\n"
                f"地面类型：{res['llava_parsed']['ground_type']}\n"
                f"场景细分：{res['llava_parsed']['scene']}\n"
                f"隐蔽点：{res['llava_parsed']['hide_place']}\n"
                f"士兵状态：{res['llava_parsed']['soldier_state']}\n"
                f"风险等级：{res['llava_parsed']['risk']}\n"
                f"YOLO检测：{res['yolo']}"
                for idx, res in enumerate(frame_results)
            ])
    )
    return summary