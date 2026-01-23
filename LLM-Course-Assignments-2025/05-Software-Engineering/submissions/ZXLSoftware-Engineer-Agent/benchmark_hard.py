import os

def save_log(message):
    # 目标：将日志写入 logs/app.log
    # BUG：如果 logs 文件夹不存在，open() 会报错
    file_path = "logs/app.log"
    
    # 修复：确保父目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w") as f:
        f.write(message)
    print(f"✅ 日志已写入 {file_path}")

if __name__ == "__main__":
    try:
        # 确保环境干净（删除 logs 文件夹）
        if os.path.exists("logs"):
            import shutil
            shutil.rmtree("logs")
            
        save_log("System init...")
    except FileNotFoundError:
        print("❌ FileNotFoundError: 目录不存在")
        exit(1)