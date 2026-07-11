# Qwen2.5-0.5B 知识蒸馏工程
基于通义千问Qwen2.5-0.5B底座完成小规模知识蒸馏，针对《卡拉马佐夫兄弟》问答场景构建专用轻量模型。

## 项目简介
1. 任务：通用大模型轻量化蒸馏，构建文学问答专用小模型
2. 底座：Qwen2.5-0.5B
3. 数据集：卡拉马佐夫兄弟问答对话语料
4. 技术栈：Python 3.10 + PyTorch + Transformers + LoRA微调 + 知识蒸馏
5. 说明：436MB权重文件 `model.safetensors` 因GitHub单文件100MB限制，**仓库不存储完整权重**，仅保留数据集、配置、训练脚本；权重可自行合并或从外部存储获取。
6. ## 完整模型权重获取（外部存储一键下载）
本仓库仅存放模型配置、分词器与数据集，不含436MB权重文件，完整权重一键下载：

1. HuggingFace 主仓库：
[https://huggingface.co/luweijie151/Qwen2.5-0.5B-Karamazov-Distill](https://huggingface.co/luweijie151/Qwen2.5-0.5B-Karamazov-Distill)

2. ModelScope 国内高速镜像：
[https://modelscope.cn/models/luweijie151/Qwen2.5-0.5B-Karamazov-Distill](https://modelscope.cn/models/luweijie151/Qwen2.5-0.5B-Karamazov-Distill)

3. 百度网盘备用：
[https://pan.baidu.com/s/xxxxxxx](https://pan.baidu.com/s/xxxxxxx)
提取码：abcd

下载完成后将 `model.safetensors` 放到路径 `融合文件/qwen0.5b_merged_full/` 下即可正常加载模型。



