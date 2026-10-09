# Contextual Predictive Mutation Testing — v2 复现

FSE 2023 论文 **Contextual Predictive Mutation Testing** 的小规模复现工程：
使用作者最终 **v2 跨项目套件检查点**，在本地重新编码、部署模型并推理 Csv 测试集的全部
**1,040 个变异体对应套件、42,687 个变异体—测试对**。

| 跨项目套件预测 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 本机 v2 推理 | 0.518519 | 0.652542 | 0.577861 |
| 论文 Table 3 | 0.52 | 0.65 | 0.58 |

评价正类为存活（0），套件判定规则为 `max(p_killed) > 0.25`。
三项主要指标保留两位小数后与论文一致。正式指标来自本机新推理

## 查看与核验结果

- [实验报告](docs/实验报告.md)：实验目的、设置、步骤、结果及局限。
- [本次运行报告](results/v2/复现报告.md)、[机器可读汇总](results/v2/summary.json)。
- [测试对预测](results/v2/test_pair_predictions.csv)、[套件预测](results/v2/test_suite_predictions.csv)。
- [前向计算对照](results/v2/forward_parity.json)、[分片对齐证据](results/v2/author_alignment.json)。
- [独立核验记录](results/v2/verification.json)、[数据与模型来源](docs/数据与模型.md)。

只检查仓库中已保存的结果，**无需 GPU、PyTorch 或下载数据**：

```powershell
python verify_v2.py --results-only
python -m unittest discover -s tests -v
```

该命令重新计算 CSV 的指标、套件聚合与参考差异；它不会运行模型。
GitHub Actions 同样执行这些检查。

## 从新克隆的仓库重跑模型

推荐 Windows、Python 3.13、支持 CUDA 的 NVIDIA GPU，显存 8 GB，批大小 4。
本次实验使用 RTX 4060 Laptop GPU、FP32、PyTorch 2.11.0+cu128、Transformers 4.57.1。
纯全量推理约 12.2 分钟；首次下载、环境准备、输入编码与核验另计。
为避免占用 C 盘，请将仓库克隆到 D 盘，再在仓库目录执行：

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe prepare_assets.py
.\run_v2.ps1 -BatchSize 4
```

`setup.ps1` 将虚拟环境、临时目录和依赖缓存设在仓库内。
`prepare_assets.py` 从官方最终 v2 TAR 按字节范围提取约 500 MB 的模型张量及约 33 MB 的
源码、原始测试数据和参考预测，同时获取分词器。不会下载整个约 63.8 GB 的实验包。
下载可续传，校验固定 TAR 大小、内容范围、SHA-256 和原模型张量 CRC32。

若 Google Drive 无法访问，可先按[官方 v2 说明](https://zenodo.org/records/10654933)
下载原始实验 TAR 到 D 盘，再执行：

```powershell
.\.venv\Scripts\python.exe prepare_assets.py --local-archive 'D:\datasets\contextual-pmt-artifact.tgz'
```

原包虽然可能以 `.tgz` 命名，实际是未压缩 TAR；脚本只接受本实验固定版本。
资产准备完毕后的推理可离线运行。其他平台可以在安装合适的 PyTorch CUDA 版本后执行
`python run_all_v2.py --batch-size 4`；本仓库的完整 GPU 验证环境是 Windows。

## 运行流程

`run_all_v2.py` 在一个进程中完成：

1. 核验作者预处理源码 SHA-256，直接执行原始 `tokenize_str`、`subsample_mutants`，生成 1024-token 输入。
2. 提取作者保存概率作为参考，按整分片真实套件标签及有序测试对标签唯一匹配分片顺序。
3. 加载最终 v2 套件权重，按真实位置嵌入尺寸重建模型，检查 64 个输入的历史前向公式对照。
4. 执行跨分片试运行、全部模型推理和套件聚合，输出 CSV、JSON。
5. 独立计数核验、源文件与模型哈希检查，生成本次运行报告。

运行会更新 `results/v2/`，原始公开实验结果可通过 Git 历史恢复。
`provenance/` 保存原实验材料的哈希、字节索引和模型张量来源信息。
新提取检查点的 ZIP 元数据可能与首次本地提取不同，容器 SHA-256 会相应变化；
模型张量和模型对象的原始校验值均固定。


## 来源与许可

- Jain et al., *Contextual Predictive Mutation Testing*, FSE 2023，DOI：10.1145/3611643.3616289。
- [论文 PDF](https://clairelegoues.com/assets/papers/jainContextualPMT.pdf)。
- [作者最终 v2 实验材料](https://zenodo.org/records/10654933)。
- [CodeBERT](https://huggingface.co/microsoft/codebert-base)。
- [历史 Transformers 4.23.1 编码器](https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/roberta/modeling_roberta.py)。

本仓库新增复现代码采用 [MIT 许可](LICENSE)。下载的第三方材料仍遵循各自许可；
