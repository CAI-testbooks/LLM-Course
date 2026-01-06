import cv2
import time
import airsim
import imageio
import numpy as np


def record_uav_swarm_video(output_path="uav_swarm_demo_hd.mp4",
                           duration=150,  # 录制时长（秒）
                           follow_demo=False,  # 自动跟随巡检结束
                           demo_end_signal=None):
    """
    适配旧版AirSim的高清视频录制（移除width/height参数，兼容所有版本）
    """
    # 初始化AirSim客户端（仅读取画面，不控制无人机）
    client = airsim.MultirotorClient()
    client.confirmConnection()  # 验证连接

    # ========== 适配旧版AirSim：移除width/height参数 ==========
    camera_id = 0  # 数字ID兼容所有版本
    camera_type = airsim.ImageType.Scene
    img_request = airsim.ImageRequest(
        camera_id,
        camera_type,
        pixels_as_float=False,
        compress=False  # 移除width/height，使用默认分辨率
    )

    # ========== 先获取一次画面，确定原始分辨率（关键！）==========
    print("📷 正在检测相机分辨率...")
    responses = client.simGetImages([img_request], vehicle_name="UAV0")
    if not responses:
        raise RuntimeError("❌ 无法获取AirSim相机画面，请检查AirSim是否启动")

    response = responses[0]
    # 动态获取原始画面的宽高（避免拉伸模糊）
    ORIGINAL_WIDTH = response.width
    ORIGINAL_HEIGHT = response.height
    print(f"✅ 检测到相机分辨率：{ORIGINAL_WIDTH}×{ORIGINAL_HEIGHT}")

    # ========== 视频编码参数（匹配原始分辨率）==========
    frame_width = ORIGINAL_WIDTH
    frame_height = ORIGINAL_HEIGHT
    # 兼容旧版OpenCV的H.264编码（若报错可换mp4v）
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # 若不行则改为 *'mp4v'
    fps = 30  # 旧版AirSim建议降为30fps，避免画面卡顿
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"✅ 开始录制（{frame_width}×{frame_height}，{fps}fps）")
    print(f"⏱️  预计录制时长：{duration}秒（按q键提前终止）")
    start_time = time.time()

    try:
        while True:
            # 1. 获取UAV0的相机画面
            responses = client.simGetImages([img_request], vehicle_name="UAV0")
            if not responses:
                print("⚠️  未获取到画面，重试...")
                time.sleep(0.1)
                continue

            # 2. 转换为OpenCV格式（无拉伸，保证清晰）
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            # 直接使用原始分辨率reshape，避免缩放模糊
            img_rgb = img1d.reshape(ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3)

            # 3. 写入视频（尺寸完全匹配，无拉伸）
            out.write(img_rgb)

            # 4. 显示预览窗口（可选）
            cv2.imshow('UAV Swarm Record', img_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 用户手动终止录制")
                break

            # 5. 终止条件
            elapsed_time = time.time() - start_time
            if follow_demo:
                if demo_end_signal and demo_end_signal[0]:
                    print("✅ 巡检结束，自动终止录制")
                    break
            else:
                if elapsed_time >= duration:
                    print("✅ 录制时长达标，自动终止")
                    break

    except Exception as e:
        print(f"❌ 录制出错：{str(e)}")
    finally:
        # 强制释放资源，避免视频损坏
        out.release()
        cv2.destroyAllWindows()
        total_time = time.time() - start_time
        print(f"\n📽️  录制完成！")
        print(f"📂 文件：{output_path}")
        print(f"⏱️  实际时长：{total_time:.1f}秒")


def video_to_gif_hd(video_path, gif_path="uav_swarm_demo_hd.gif", fps=10):
    """
    视频转GIF（适配旧版AirSim录制的视频）
    """
    try:
        reader = imageio.get_reader(video_path)
        # 高质量GIF设置
        writer = imageio.get_writer(
            gif_path,
            fps=fps,
            quality=10,
            macro_block_size=1
        )
        # 逐帧写入，减小体积
        for i, frame in enumerate(reader):
            if i % 2 == 0:  # 每2帧取1帧
                writer.append_data(frame)
        writer.close()
        print(f"✅ GIF转换完成：{gif_path}")
    except Exception as e:
        print(f"❌ GIF转换失败：{str(e)}")


# 主函数（直接运行）
if __name__ == "__main__":
    # 录制视频（150秒，覆盖完整巡检）
    record_uav_swarm_video(duration=400)
    # 转换为GIF（可选）
    video_to_gif_hd("uav_swarm_demo_hd.mp4")