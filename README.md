# miniChatGPT
Mini ChatGPT基于Huggingface GPT2，支持有监督微调(SFT)、奖励模型(Reward Model)和PPO全流程ChatGPT训练。

## 如何使用
1. 使用prepare_sft_dataset.py准备训练数据
2. 使用train_sft.py进行有监督微调(SFT)
3. 使用train_rm.py训练奖励模型(Reward Model)
4. 基于奖励模型，使用train_ppo.py对SFT模型进行进一步PPO训练

## 致谢
本项目是对[minChatGPT](https://github.com/ethanyanjiali/minChatGPT)的修改和完善，感谢minChatGPT的辛勤工作。
