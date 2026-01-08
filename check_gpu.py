import torch

# 检查显卡
print("--- 显卡检查开始 ---")
if torch.cuda.is_available():
    print("✅ 成功！Pytorch 识别到了显卡！")
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} GB")
    print("任务 01：环境搭建 --- [完成]")
else:
    print("❌ 失败：使用的是CPU，请检查环境设置！")