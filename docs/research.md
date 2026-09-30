# 规则与算法来源核验

访问日期：2026-09-30。以下是短调研；没有复制第三方游戏代码、图像或音频，也没有操作公开比赛或排行榜。

| 来源 | 核验内容 | 许可与借鉴范围 |
|---|---|---|
| [Minesweeper Online 基本规则](https://minesweeper.online/cn/help/gameplay) | 实际可玩的经典扫雷；八邻域数字、揭格、插旗、零区展开，揭完安全格获胜 | 网站内容未作为开放素材复用；仅引用规则事实。其无猜模式/3BV筛选不沿用 |
| [Minesweeper Online PvP](https://minesweeper.online/help/pvp) | 已有多人竞争、匹配与私人大厅 | 未复用代码或排名系统；该页面不足以证明与本项目相同的共享移动规则 |
| [Minesweeper Co-op 玩法](https://www.minesweepercoop.com/how-to-play) · [当前首页](https://www.minesweepercoop.com/) · [更新记录](https://www.minesweepercoop.com/changelog) | 玩法页明确区分共享一张棋盘的合作模式与各自操作相同布局的 1v1 竞速；共享线索并非本项目首创 | 未发现可供复用的源码／素材许可证，只引用结构事实，不复制其程序。玩法页的单雷团灭描述与首页及 2026-09-08 更新的三次共享生命存在版本差异，不据此推断所有模式统一规则 |
| [Xbox 官方 Minesweeper Flags 介绍](https://news.xbox.com/en-us/2009/02/11/arcade-minesweeper-flags/) | 官方确认已有多人 Flags 模式；因此不能声称首次多人扫雷 | 商业游戏，不复制代码或素材。本项目的安全钻石与逐格移动须单独说明 |
| [DavidNHill/JSMinesweeper](https://github.com/DavidNHill/JSMinesweeper) · [LICENSE](https://github.com/DavidNHill/JSMinesweeper/blob/master/LICENSE) · [可玩页面](https://davidnhill.github.io/JSMinesweeper/) | 求解器将环境、GUI与Solver分开；概率引擎与行动策略不同，状态风险最低不等价于全局获胜最优 | MIT，许可证署名为 Copyright (c) 2022 David N Hill；仅参考架构与概率计数思想，本项目独立实现，无源码复制 |

经典验证模式继承数字、固定雷图、揭格、人工旗标、零区展开及安全格全部揭开胜利；未实现数字双击连开，首点可能失败，界面明确说明。主模式新增双方位置、公开安全钻石、四邻接轮流移动、到达计分、踩雷停止行动及步数上限；不自动展开零区。共享线索也会帮助对手。完整规则及工程解释见 rules.md。

现有网页的首击保护、无猜生成和多人生命机制各不相同，不能把任一产品的全部细则等同于唯一的经典标准。本项目保留小型经典规则验证模式，并明确披露固定雷图、首击可能失败的实现选择。主模式采用未筛选的均匀固定雷数先验；不沿用上述合作网站的可解性筛选，因此其精确概率计算与本项目的先验条件也不能混用。

技术问题是固定资源位置和公开线索下的风险与争夺权衡。约束计数不是本项目原创；风险路径代价为启发式，并非精确联合路径成功概率。轻量监督模仿的训练标签是教师动作，不是最优动作证明，也不是单次雷位作为精确概率标签。

调研未进行系统新颖性检索；现有变体的存在足以排除“首次多人扫雷”这一未经证明的主张。本项目为课程实践，不宣称获得专利。
