### 项目说明
- 数据集CORA 图数据集
- 任务：多分类
- 使用模型GCN SVM FNN
- 包括构图、数据预处理及feature encoding

### 依赖库安装
```
pip install requirements.txt
# 以下4个库可能会安装失败
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
# 如果出现安装失败的提示，按照如下操作解决
# 1.获取cuda版本
# 2.使用如下命令安装 将${cuda}替换为自己的cuda版本即可，如果没有cuda环境，则将其替换为空字符串，安装cpu版本
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${cuda}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${cuda}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${cuda}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${cuda}.html
```
### 程序运行
```
移动到main.py所在目录，执行命令：
python main.py
等待命令执行完毕，大概需要运行3分钟左右
```

