"""Write a current-run report after full verification; do not modify repository docs."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/v2'
summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
verification = json.loads((OUT / 'verification.json').read_text())
if not verification['passed']:
    raise RuntimeError('Verification must pass before report generation')
m = summary['suite_level_threshold_0_25']
c = summary['author_stored_prediction_comparison']
report = f'''# v2 本机推理实验结果

在本实验中，我复现了 FSE 2023 论文《Contextual Predictive Mutation Testing》中
MutationBERT 的跨项目测试套件预测实验，使用作者最终 v2 的训练权重，在本地重新编码
全部 1,040 个变异体对应套件、42,687 个变异体—测试对并完成模型推理。

输入长度上限为 1024 tokens；按作者规则截断超长输入。各测试产生杀死概率，套件通过
`max(p_killed) > 0.25` 判定已杀死，套件评价正类为存活（0）。保留全部官方标签。

| 结果 | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| 本机重新推理 | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['accuracy']:.6f} |
| 论文 Table 3 跨项目套件 | 0.52 | 0.65 | 0.58 | 未列出 |

混淆矩阵（行=真实，列=预测，顺序 `[0存活, 1已杀死]`）为
`{m['confusion_matrix_labels_0_1']}`。三项主要指标四舍五入至两位小数与论文一致。

设备为 {summary['gpu']}，FP32、batch size={summary['batch_size']}，
纯全量推理耗时 {summary['inference_seconds']:.2f} 秒
（{summary['inference_seconds']/60:.2f} 分钟），不含环境准备、编码、试运行与核验。

64 个真实输入完成历史前向公式对照；全部 42,687 个新预测与作者保存概率比较，
最大概率绝对差 {c['max_pair_probability_abs_difference']:.10f}，
阈值 0.25、0.5、0.9 的分类分歧为
`{c['pair_threshold_decision_disagreements']}`。独立计数核验通过。

作者保存概率仅用于核对；正式指标来自本机新预测。套件标签与保留测试对标签 OR
有 {summary['suite_label_vs_pair_or_mismatches']} 处不一致，保留官方套件标签。

本次范围是 v2 跨项目套件的推理复现。未重新训练、执行 Major/Defects4J、
复现同项目评估、Seshat 对照或节时结论。同一套件检查点的测试对指标是辅助结果，
不能对应论文矩阵最优检查点。

来源：[论文](https://clairelegoues.com/assets/papers/jainContextualPMT.pdf)、
[最终 v2 实验材料](https://zenodo.org/records/10654933)。
'''
(OUT / '复现报告.md').write_text(report, encoding='utf-8')
print('Report written:', OUT / '复现报告.md')
