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