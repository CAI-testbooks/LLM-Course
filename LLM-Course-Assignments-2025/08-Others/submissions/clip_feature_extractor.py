import os
import torch
import numpy as np
from PIL import Image
import open_clip
from skimage.segmentation import slic
from skimage.util import img_as_float
from tqdm import tqdm
import cv2

# 配置参数
class FeatureConfig:
    model_type = 'ViT-H-14'
    pretrained = 'laion2b_s32b_b79k'
    
    # 根据500x500图像尺寸优化的超像素级别
    slic_levels = {
        'large': 2500,   # 大量超像素块 - 保留精细细节
        'middle': 500,  # 中量超像素块 - 平衡细节和结构
        'few': 100       # 少量超像素块 - 高度抽象
    }
    
    # 改进的纹理增强参数
    texture_scale = {
        'large': 0.5,   # 适度纹理增强
        'middle': 0.3,  # 中等纹理增强
        'few': 0.1      # 轻微纹理增强
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
vlmodel, preprocess_train, _ = open_clip.create_model_and_transforms(
    model_name=FeatureConfig.model_type,
    pretrained=FeatureConfig.pretrained,
    precision='fp32',
    device=FeatureConfig.device
)
vlmodel = vlmodel.to(FeatureConfig.device).eval()

def process_dataset(input_root, output_root):
    """处理整个数据集，生成三个级别的超像素图和特征"""
    # 创建输出目录
    output_dirs = {}
    for level in FeatureConfig.slic_levels.keys():
        level_dir = os.path.join(output_root, f"{os.path.basename(input_root)}_{level}")
        os.makedirs(level_dir, exist_ok=True)
        output_dirs[level] = level_dir
    
    # 初始化特征字典（三个级别）
    features = {level: {} for level in FeatureConfig.slic_levels.keys()}
    
    # 遍历所有图像
    for root, dirs, files in os.walk(input_root):
        for file in tqdm(files, desc=f"Processing {os.path.basename(input_root)}"):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(img_path, input_root)
            feature_key = os.path.basename(img_path)
            
            # 处理每个级别的超像素
            for level, n_segments in FeatureConfig.slic_levels.items():
                level_path = os.path.join(output_dirs[level], rel_path)
                
                # 创建子目录
                os.makedirs(os.path.dirname(level_path), exist_ok=True)
                
                # 生成该级别的超像素图（如果不存在）
                if not os.path.exists(level_path):
                    slic_img = _generate_leveled_slic(img_path, level, n_segments)
                    slic_img.save(level_path)
                
                # 提取该级别的CLIP特征
                if feature_key not in features[level]:
                    img = Image.open(level_path)
                    feat = _get_clip_embedding(img)
                    features[level][feature_key] = feat.cpu()
    
    # 保存特征（每个级别单独保存）
    for level in features.keys():
        torch.save(
            features[level], 
            os.path.join(output_root, f"{os.path.basename(input_root)}_{level}_slic.pt")
        )
    
    return features

def _generate_leveled_slic(img_path, level, n_segments):
    """生成指定级别的超像素图"""
    img = Image.open(img_path).convert("RGB")
    np_img = np.array(img)
    image = img_as_float(np_img)
    
    # 优化SLIC参数（根据级别调整紧凑度）
    compactness = 10 if level == 'large' else 20 if level == 'middle' else 30
    
    # SLIC分割
    segments = slic(
        image, 
        n_segments=n_segments, 
        compactness=compactness,
        sigma=1,
        min_size_factor=0.1,
        max_size_factor=2
    )
    
    # 创建均值化图像
    mean_image = np.zeros_like(image)
    for label in np.unique(segments):
        mask = (segments == label)
        mean_color = image[mask].mean(axis=0)
        mean_image[mask] = mean_color
    
    # 纹理增强
    texture_weight = FeatureConfig.texture_scale[level]
    if texture_weight > 0:
        # 转换为Lab空间进行纹理分析
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2Lab)
        _, a, b = cv2.split(lab)
        
        # 计算纹理图
        sobel_a_x = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
        sobel_a_y = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
        sobel_b_x = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
        sobel_b_y = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
        
        texture_map = np.sqrt(sobel_a_x**2 + sobel_a_y**2 + sobel_b_x**2 + sobel_b_y**2)
        texture_map = cv2.normalize(texture_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # 将纹理映射扩展到RGB
        texture_map_rgb = np.stack([texture_map]*3, axis=-1)
        
        # 纹理作为增强因子
        blended_image = mean_image * (1 + texture_weight * texture_map_rgb)
        blended_image = np.clip(blended_image, 0, 1)
    else:
        blended_image = mean_image
    
    # 转换为PIL图像
    result_image = (blended_image * 255).astype(np.uint8)
    return Image.fromarray(result_image)

def _get_clip_embedding(image):
    """获取CLIP嵌入"""
    with torch.no_grad(), torch.cuda.amp.autocast():
        tensor = preprocess_train(image).unsqueeze(0).to(FeatureConfig.device)
        feature = vlmodel.encode_image(tensor)
        return feature / feature.norm(dim=-1, keepdim=True)

if __name__ == "__main__":
    # 处理训练集
    train_features = process_dataset(
        input_root="./Data/training_images",
        output_root="./Data/processed_images"
    )
    
    # 处理测试集
    test_features = process_dataset(
        input_root="./Data/test_images",
        output_root="./Data/processed_images"
    )
    
    print("特征提取完成！")
    print(f"训练集特征: {len(train_features['large'])} 图像")
    print(f"测试集特征: {len(test_features['large'])} 图像")