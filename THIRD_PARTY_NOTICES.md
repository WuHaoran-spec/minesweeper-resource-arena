# 依赖与许可

本仓库的Python、HTML、CSS、JavaScript均为本项目编写，采用MIT。没有打包系统字体、商业游戏素材或教师原材料。UI仅使用系统字体、文本与CSS。

- Python：运行时依赖，[PSF License](https://docs.python.org/3/license.html)；不在本仓库重新分发 Python 运行时。
- NumPy：BSD-3-Clause，见[官方许可证](https://numpy.org/doc/stable/license.html)。仅通过 requirements 安装，不在仓库内打包其代码；本项目的 MIT 不替代 NumPy 的许可条款。
- GitHub Actions checkout/setup-python：CI 引用各官方 Action，不将第三方代码拷入本仓库。运行时仍分别受[checkout 许可证](https://github.com/actions/checkout/blob/main/LICENSE)与[setup-python 许可证](https://github.com/actions/setup-python/blob/main/LICENSE)约束。
- [JSMinesweeper](https://github.com/DavidNHill/JSMinesweeper)：[MIT](https://github.com/DavidNHill/JSMinesweeper/blob/master/LICENSE)，Copyright (c) 2022 David N Hill。仅作思想与架构参考，未复制源文件或素材，详见[来源调研](docs/research.md)。若后续复制其实质代码，须保留该项目完整 MIT 版权与许可声明。
- Minesweeper Online、Xbox Minesweeper Flags、Minesweeper Co-op：只引用官方／运营方的规则说明，不分发网页、商业程序、游戏图片、音频或其他未获许可素材。网站可访问并不表示其内容使用 MIT。
- 自生成地图/机器对战记录与本项目训练权重：不含外部数据，按本仓库MIT许可提供；详情见数据说明。
- 人工操作：默认标为未经核验的交互；未将浏览器自动化标成人类数据，没有公开任何未经同意的玩家记录。

私人课程文件和教师转写、模板未获得公开许可，完全排除于本仓库。

本地截图属于本程序实际运行输出，仅展示自行绘制的 UI 与合成游戏状态。附图 SVG 为本项目绘制。仓库不含字体文件、外部预训练模型或真人玩家数据。许可链接于 2026-09-30 核验；公开发布前的文件与压缩内容检查使用 `scripts/privacy_audit.py`，报告保存在公开仓库之外。
