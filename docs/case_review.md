# 真实案例复核

|类型|真实对局 / 记录步号|动作|目的地风险|状态|最终胜者|
|---|---|---|---:|---|---|
|explicit_target_switch|retest-m000-p0-s0-f0 / 7|right|0.00000|exact_global_model_count|1|
|learned_failure|retest-m000-p1-s1-f0 / 2|down|0.66667|exact_global_model_count|0|
|algorithm_disagreement|retest-m001-p0-s0-f0 / 10|left|0.00000|exact_global_model_count|1|
|explicit_contest|retest-m001-p0-s0-f0 / 14|left|0.00000|exact_global_model_count|1|
|approximate_risk|retest-m071-p0-s0-f0 / 18|left|0.24403|approximate_cutoff|1|

## explicit_target_switch

同一行动者目标从[3, 5]转为[3, 7]，两个钻石当时均未被领取。 记录步号7，行动者1。核验方式：logical predicate checked against actual logged public state; full replay verified during evaluation。人工审核：否。

## learned_failure

最终L输局，且该步踩雷；踩雷前风险只是后验/估计，不是读取真值。 记录步号2，行动者1。核验方式：logical predicate checked against actual logged public state; full replay verified during evaluation。人工审核：否。

## algorithm_disagreement

同一公开观察：B1=up，B2=left。 记录步号10，行动者1。核验方式：logical predicate checked against actual logged public state; full replay verified during evaluation。人工审核：否。

## explicit_contest

当前规则策略目标与另一方上次显式输出目标相同，双方仍存活。 记录步号14，行动者1。核验方式：logical predicate checked against actual logged public state; full replay verified during evaluation。人工审核：否。

## approximate_risk

模型计数达到预算，明确标记近似；不声称精确概率。 记录步号18，行动者1。核验方式：logical predicate checked against actual logged public state; full replay verified during evaluation。人工审核：否。

导入对应完整公共回放，可检查原观察、动作、风险和终局。网络没有显式目标头；自动案例不是意图解释。见[实验报告](experiments.md)。
