# 规则与算法来源核验

访问日期：2026-09-30。以下是短调研；没有复制第三方游戏代码、图像或音频，也没有操作公开比赛或排行榜。

| 来源 | 核验内容 | 许可与借鉴范围 |
|---|---|---|
| [Minesweeper Online 基本规则](https://minesweeper.online/cn/help/gameplay) | 实际可玩的经典扫雷；八邻域数字、揭格、插旗、零区展开，揭完安全格获胜 | 网站内容未作为开放素材复用；仅引用规则事实。其无猜模式/3BV筛选不沿用 |
| [Minesweeper Online PvP](https://minesweeper.online/help/pvp) | 已有多人竞争、匹配与私人大厅 | 未复用代码或排名系统；该页面不足以证明与本项目相同的共享移动规则 |
| [Minesweeper Co-op 玩法](https://www.minesweepercoop.com/how-to-play) · [当前首页](https://www.minesweepercoop.com/) · [更新记录](https://www.minesweepercoop.com/changelog) | 玩法页明确区分共享一张棋盘的合作模式与各自操作相同布局的 1v1 竞速；共享线索并非本项目首创 | 未发现可供复用的源码／素材许可证，只引用结构事实，不复制其程序。玩法页的单雷团灭描述与首页及 2026-09-08 更新的三次共享生命存在版本差异，不据此推断所有模式统一规则 |
| [Xbox 官方 Minesweeper Flags 介绍](https://news.xbox.com/en-us/2009/02/11/arcade-minesweeper-flags/) | 官方确认已有多人 Flags 模式；因此不能声称首次多人扫雷 | 商业游戏，不复制代码或素材。本项目的安全钻石与逐格移动须单独说明 |
| [DavidNHill/JSMinesweeper](https://github.com/DavidNHill/JSMinesweeper) · [LICENSE](https://github.com/DavidNHill/JSMinesweeper/blob/master/LICENSE) · [可玩页面](https://davidnhill.github.io/JSMinesweeper/) | 求解器将环境、GUI与Solver分开；概率引擎与行动策略不同，状态风险最低不等价于全局获胜最优 | MIT，许可证署名为 Copyright (c) 2022 David N Hill；仅参考架构与概率计数思想，本项目独立实现，无源码复制 |
| [ReactOS WinMine main.c](https://github.com/reactos/reactos/blob/master/base/applications/games/winmine/main.c) · [main.h](https://github.com/reactos/reactos/blob/master/base/applications/games/winmine/main.h) | Windows 风格的可编译经典扫雷实现。头文件定义9×9/10、16×16/40、30列×16行/99；源码可核对首击保护、零区展开和按旗数连开 | 所核验main.c/main.h文件头为 **LGPL-2.1-or-later**，署名Copyright 2000 Joshua Thielen；不能用ReactOS仓库首页的GPL标签代替组件许可判断。未复制代码、位图、图标或音频 |
| [Wine WineMine main.c](https://github.com/wine-mirror/wine/blob/master/programs/winemine/main.c) · [main.h](https://github.com/wine-mirror/wine/blob/master/programs/winemine/main.h) · [LICENSE](https://github.com/wine-mirror/wine/blob/master/LICENSE) | Wine项目的经典Windows程序实现；同样提供三档尺寸、首击格安全与chord。ReactOS所核验源文件保留WineMine名称及同一早期作者署名，不能当作两个相互独立的规则创新 | main.c/main.h文件头及Wine项目LICENSE均明确 **LGPL-2.1-or-later**；文件署名Joshua Thielen，项目LICENSE署名Wine project authors。仅核对规则事实，不将其代码改标MIT |

## Windows风格与源码许可的边界

本次没有找到微软原版Winmine以开源许可证发布的官方依据。ReactOS/Wine的开源实现不意味着微软原版程序自动开源；界面截图只能作为外观参考，不能代替源文件和素材的授权。公开仓库或“复刻”名称也不能证明代码来源及再分发权。GitHub的[许可说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)明确区分可见仓库与开源许可。本项目采用独立编写的HTML/CSS界面，不嵌入原版截图、系统游戏位图或第三方程序。

从Wine/ReactOS源码可以核对：初次揭格时布雷，排除所选**一格**；这保证首击不直接触雷，但不等于首击周围3×3全部安全。chord检查周围旗标数量是否等于已揭数字，再揭开未插旗邻格；插错旗仍可能踩雷，旗标不能被AI推断器当作已验证雷位。中级16×16/40和高级30列×16行/99是可对照的难度预设，不代表这些棋盘一定可无猜求解。依据为[Wine布雷与连开实现](https://github.com/wine-mirror/wine/blob/master/programs/winemine/main.c)及[难度常量](https://github.com/wine-mirror/wine/blob/master/programs/winemine/main.h)。

## 本项目沿用与新增规则

v1的9×9资源争夺实验保持冻结：10雷、3公开安全钻石、四邻接轮流移动、无零区展开、踩雷淘汰并保留得分、200步上限。它的5,200局评测与模型属于`arena-v1.0`，不能改称大棋盘结果。旧版经典验证模式首击可能失败且没有chord，这只是v1的已披露实现边界，不是对Windows经典扫雷的完整复现。

大棋盘扩展另记v2：中级16×16/40雷/7资源，高级30列×16行/99雷/11资源；预设尺寸及雷数参考经典实现，公开资源、三条生命、重生与双方角落3×3安全区属于本项目新增的争夺规则。v2还增加零区展开；资源只能由实际到达取得，不能随展开远程计分。生命与开局安全区会改变风险暴露和轨迹长度，不能把v1/v2胜率直接横向比较或声称新版本自然更强。最终冻结参数、实现状态和验收以[rules.md](rules.md)及对应实验协议为准。

经典扩展与争夺扩展分开验证：经典模式的难度预设、首击保护、零区展开和chord服务于规则操作；本项目`classic-v2.0`排除首击格及其八邻格布雷，是比上述Wine/ReactOS首格安全更强的明确扩展，不能称为微软原版的逐项复刻。主模式仍是两人共享棋盘逐格争夺，不退回两张独立棋盘。已有网页的首击保护、无猜生成和多人生命机制各不相同，不能把任一产品的全部细则等同于唯一标准。公开安全区应进入合法先验；本项目不沿用第三方合作网站的可解性筛选，也不把对应概率模型混用。

技术问题是固定资源位置和公开线索下的风险与争夺权衡。约束计数不是本项目原创；风险路径代价为启发式，并非精确联合路径成功概率。轻量监督模仿的训练标签是教师动作，不是最优动作证明，也不是单次雷位作为精确概率标签。

大棋盘提供更多可能状态，但“局面更大”并不自动等于“训练数据质量更高”。本项目应以合法动作和标签核验、公开输入隔离、完整回放、来源标记、基础地图分区、重复状态处理、风险精确/近似标识及失败保留来描述数据质量；真实训练、留出评测和相应日志完成后才报告结果。启发式教师标签即使可重算，仍只是教师行为标签。

调研未进行系统新颖性检索；现有变体的存在足以排除“首次多人扫雷”这一未经证明的主张。本项目为课程实践，不宣称获得专利。
