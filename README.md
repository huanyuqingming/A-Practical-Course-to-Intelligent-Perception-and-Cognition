# 智能感知认知实践

## Project1：语言模型
在只更改模型结构的情况下，最优结果的复现方式为：
```bash
cd Project1
python main.py --cuda --model=LSTM --nlayers=4 --nhid=512
```

在更改模型结构与嵌入维度的情况下，最优结果的复现方式为：
```bash
cd Project1
python main.py --cuda --model=LSTM --nlayers=4 --nhid=512 --emsize=256
```

在更改模型结构与dropout rate的情况下，最优结果的复现方式为：
```bash
cd Project1
python main.py --cuda --model=LSTM --nlayers=4 --nhid=512 --dropout=0.3
```

在更改模型结构、嵌入维度、dropout rate、序列长度的情况下，最优结果的复现方式为：
```bash
cd Project1
python main.py --cuda --model=Transformer --nlayers=8 --nhid=1024 --emsize=512 --dropout=0.1 --bptt=200 --lr=5
```

## Project2：图像生成
最小重建误差的复现方式为：
```bash
cd Project2
python main.py --z_dimension=32
```

可视化方式为：
```bash
cd Project2
python plot.py --z_dimension=<NEEDED_DIMENSION>
```

## Project3：图片摘要生成

将数据集按照以下路径格式配置：
```
└── data
    └── flickr8
        ├── caption.txt
        ├── test_imgs.txt
        ├── train_imgs.txt
        ├── val_ings.txt
        └── image
            └── *.jpg
        
```

训练CNN-LSTM with attention：
```bash
cd Project3
python main.py train_evaluate --config_file configs/<CONFIG_NAME>.yaml
```

使用Doubao-1.5-vision-pro大模型推理：
```bash
cd Project3
export ARK_API_KEY=<YOUR_API_KEY>
export MODEL_TYPE=<YOUR_MODEL_TYPE>
python vlm.py
```

计算指标：
```bash
cd Project3

# 对CNN-LSTM with attention模型
python evaluate.py --prediction_file experiments/resnet101_attention/<CONFIG_NAME>/<CONFIG_NAME>_predictions.json \
                   --reference_file data/flickr8k/caption.txt \
                   --output_file experiments/resnet101_attention/<CONFIG_NAME>/<CONFIG_NAME>_coco_scores.txt

# 对Doubao-1.5-vision-pro大模型
python evaluate.py --prediction_file experiments/Doubao-1.5-vision-pro/predictions.json \
                   --reference_file data/flickr8k/caption.txt \
                   --output_file experiments/Doubao-1.5-vision-pro/coco_scores.txt
```