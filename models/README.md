# 模型来源与加载

自产合成游戏轨迹上的B2教师监督模仿模型，MIT许可；无外部模型或玩家数据。模型均为929参数NumPy网络。

- L_initial.npz：SHA256 `b1510e3e4d19650dd47fca4fc97e395bdc541a7075aa26a82506a9eda311693b`，验证最优epoch 45；真实优化日志见对应_training.json。
- L_final.npz：SHA256 `0a3649d98d8ff3905b0629d8b5286e127dd96e8f5af25602e8feb0d49631807d`，验证最优epoch 20；真实优化日志见对应_training.json。
- L_no_opponent_initial.npz：SHA256 `22a73c41d6a2979647b78fb173ac6b347e5f00b5734bd6430ac331ee78ef24b7`，验证最优epoch 45；真实优化日志见对应_training.json。
- L_no_opponent.npz：SHA256 `acd34d00cde8dc3657f593f2d277f871022c75da76f0a4f5e8693e7d46c0c544`，验证最优epoch 20；真实优化日志见对应_training.json。

默认L加载L_final.npz，缺权重明确报错，不退回规则策略冒充学习。只有一个训练随机种子。训练与胜率负结果见../docs/experiments.md。
