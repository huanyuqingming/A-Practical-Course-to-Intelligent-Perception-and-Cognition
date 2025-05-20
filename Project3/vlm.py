import base64
import os
import json
from volcenginesdkarkruntime import Ark

def image_to_data_url(image_path):
    """将本地图片转换为Base64 Data URL"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        file_extension = image_path.split(".")[-1].lower()
        mime_type = (
            f"image/{file_extension}" 
            if file_extension in ["jpeg", "jpg", "png", "gif", "webp"]
            else "image/jpeg"
        )
        return f"data:{mime_type};base64,{encoded_image}"
    
    except FileNotFoundError:
        raise ValueError(f"图片文件不存在: {image_path}")
    except Exception as e:
        raise RuntimeError(f"图片处理失败: {str(e)}")

# 初始化客户端
client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)

# 读取图片路径列表
data_path = "data/flickr8k/test_imgs.txt"
try:
    with open(data_path, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"{data_path}不存在")
    exit()

results = []

# 遍历处理每张图片
for idx, image_path in enumerate(image_paths, 1):
    try:
        print(f"正在处理第 {idx}/{len(image_paths)} 张图片: {image_path}")
        
        # 生成Data URL
        image_path = os.path.join("data/flickr8/image", image_path)
        data_url = image_to_data_url(image_path)
        
        # API请求
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_TYPE"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "请用英文描述这张图片的内容，只需要输出描述，输出长度控制在20个单词以内，如：man in a red shirt and a blue shirt is standing in front of a city street ."}
                    ]
                }
            ],
            extra_headers={"x-is-encrypted": "true"},
            temperature=0.5,
            top_p=0.8,
            max_tokens=1024,
        )
        
        # 提取结果
        img_id = os.path.basename(image_path)
        prediction = [response.choices[0].message.content.strip()]
        # prediction = ["This is a placeholder response."]  # Placeholder for actual API response
        
        results.append({
            "img_id": img_id,
            "prediction": prediction
        })
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        results.append({
            "img_id": os.path.basename(image_path),
            "prediction": [f"ERROR: {str(e)}"]
        })

# 保存结果
with open("experiments/Doubao-1.5-vision-pro/predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)