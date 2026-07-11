# Qwen2.5-0.5B 知识蒸馏工程
基于通义千问Qwen2.5-0.5B底座完成小规模知识蒸馏，针对《卡拉马佐夫兄弟》问答场景构建专用轻量模型。

## 项目简介
1. 任务：通用大模型轻量化蒸馏，构建文学问答专用小模型
2. 底座：Qwen2.5-0.5B
3. 数据集：卡拉马佐夫兄弟问答对话语料
4. 技术栈：Python 3.10 + PyTorch + Transformers + LoRA微调 + 知识蒸馏
5. 说明：436MB权重文件 `model.safetensors` 因GitHub单文件100MB限制，**仓库不存储完整权重**，仅保留数据集、配置、训练脚本；权重可自行合并或从外部存储获取。
## Qwen2.5-0.5B获取
可以通过[huggingface](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)获取
@misc{qwen2.5,
    title = {Qwen2.5: A Party of Foundation Models},
    url = {https://qwenlm.github.io/blog/qwen2.5/},
    author = {Qwen Team},
    month = {September},
    year = {2024}
}

@article{qwen2,
      title={Qwen2 Technical Report}, 
      author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
      journal={arXiv preprint arXiv:2407.10671},
      year={2024}
}

