<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner-light.svg" width="760" alt="Glance-X：一个模型，三种看法">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <b>简体中文</b></p>

<p align="center">
  <a href="https://github.com/Vision-Decision/Glance-X/actions/workflows/ci.yml"><img src="https://github.com/Vision-Decision/Glance-X/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.4+">
  <a href="https://huggingface.co/Qwen/Qwen3.5-4B-Base"><img src="https://img.shields.io/badge/base-Qwen3.5--4B--Base-7C3AED" alt="基座：Qwen3.5-4B-Base"></a>
  <a href="https://huggingface.co/Vision-Decision/Glance-X-4B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20model-Glance--X--4B-FFD21E?logo=huggingface&logoColor=000" alt="Hugging Face 模型：Glance-X-4B"></a>
  <a href="https://github.com/Vision-Decision/Glance-Data"><img src="https://img.shields.io/badge/data-Glance--Data-181717?logo=github&logoColor=white" alt="Glance-Data (GitHub)"></a>
  <a href="https://huggingface.co/datasets/Vision-Decision/Glance-Data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-Glance--Data-FFD21E?logo=huggingface&logoColor=000" alt="Glance-Data (Hugging Face)"></a>
  <img src="https://img.shields.io/badge/weights-CC%20BY--NC--SA%204.0-6B7280" alt="Weights: CC BY-NC-SA 4.0">
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-Apache%202.0-2F80ED?logo=apache&logoColor=white" alt="代码许可：Apache-2.0"></a>
</p>

<p align="center">
  <a href="#为什么要三个方向">为什么</a> ·
  <a href="#工作原理">工作原理</a> ·
  <a href="#快速上手">快速上手</a> ·
  <a href="#结果">结果</a> ·
  <a href="#自己训练">训练</a> ·
  <a href="docs/ARCHITECTURE.md">架构</a> ·
  <a href="docs/RESULTS.md">完整结果</a>
</p>

---

**Glance-X** 根据一句文字需求在图里找东西，比如 *"中间那个戴眼镜的孩子"* 或 *"boy on the left raising his racket"*。
它会给每个符合的物体画一个**框**，或者回答**图里没有这个东西**。

它的不同之处在于：**同一份权重可以按三个方向运行**，每次调用由你来选：

| 方向 | 只算一次、可共享的部分 | 适合 |
|---|---|---|
| `image_first` &nbsp;先图后文 | 图片 | 一张图，很多句需求，并行回答 |
| `text_first` &nbsp;先文后图 | 需求 | 一句需求，在很多张图或视频帧里找 |
| `joint` &nbsp;图文双向 | 无：图和需求互相读取 | 单个（图, 需求）对上最准的答案 |

<p align="center">
  <img src="assets/demo.gif" width="860" alt="演示：Glance-X 在一张图上回答六句需求、找出多个目标、拒绝相似但不对的描述，并在六张图里搜索同一句需求">
  <br>
  <sub>Glance-X-4B（K=8）在留出图片上的真实输出。<a href="assets/demo.mp4">MP4 版本</a> ·
  由 <a href="demo/"><code>demo/</code></a> 生成 · 图片来自 MS COCO。</sub>
</p>

## 为什么要三个方向

根据文字找物体的模型，通常把读输入的顺序写死了，而这个顺序决定了哪类任务便宜：

- **先图后文**（大多数视觉语言模型）：图只读一次，之后可以对它问任意多句。但要拿*一句*需求在*很多*张图里找就贵了，
  因为模型读每张图时并不知道要找什么。
- **先文后图**：需求只读一次，每张图都是带着"要找什么"去读的。适合在相册或视频帧里搜索，但对同一张图问很多句时，
  图的计算会被重复。
- **双向**：图和文字互相注意（类似 Grounding DINO 的融合），单次回答最准，但没有任何部分可以共享或缓存。

Glance-X 把这三种做进了**一个模型**。缓存和精度之间的取舍，变成推理时由你做的选择，而不是训练时就定死的。

## 工作原理

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg">
    <img src="assets/architecture-light.svg" width="900" alt="Glance-X 架构：图片流和文字流分别运行的单模态段、三种排布的融合段、以及四个输出头">
  </picture>
</p>

基座是 **Qwen3.5-4B-Base**：32 层（24 层 Gated DeltaNet + 8 层 gated 全注意力）。语言模型的所有线性层加
LoRA r=64，视觉塔冻结。

1. **单模态段**（第 1…K 层，默认 K=8）：图片 token 和需求 token 作为两条独立的流运行，互相看不见，所以都可以只算一次、
   缓存复用。需求流从固定位置开始，因此缓存下来的文字特征与之后配哪张图无关。
2. **融合段**（第 K+1…32 层）：一条流作为共享的*前缀*，另一条作为读取它的*块*，很多个块可以在一次前向里共享同一个前缀：
   - `image_first`：前缀 = 图片，块 = [需求, R]
   - `text_first`：前缀 = 需求，块 = [图片, R]
   - `joint`：没有前缀，块 = [图片, 需求, R]，块内注意力是双向的
3. **输出头**读取读出 token R 和图片 token：
   - **presence**：P(图里有符合需求的东西)
   - **box**：先选一个 patch 中心，再为四条边各预测一个 64-bin 分布
   - **IoU**：P(框与目标的 IoU ≥ 0.5)
   - **multi-box**：逐 patch 的 objectness，为每个目标各出一个框

单框的做法（存在门、patch 中心 softmax、四边分布、IoU 头）沿用 [Glance](https://github.com/Vision-Decision/Glance)；
三方向骨干和多框头是 Glance-X 新增的。

前缀/块的计算是精确的：在 K=0、块内因果掩码的设置下，每个方向都与把对应单条序列（[图, 需求] 或 [需求, 图]）
直接送进 Hugging Face 前向的结果一致。
[`scripts/test_equivalence.py`](scripts/test_equivalence.py) 检查这一点，并检查共享同一前缀的不同需求之间互不影响。
详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 快速上手

```bash
git clone https://github.com/Vision-Decision/Glance-X && cd Glance-X
pip install -e ".[fla]"        # 没有 CUDA GPU 的机器去掉 [fla]
```

权重已发布在 Hugging Face：**[Vision-Decision/Glance-X-4B](https://huggingface.co/Vision-Decision/Glance-X-4B)**
（LoRA + 输出头，547 MB；9 GB 的基座模型在首次使用时下载）。

```python
from glancex import GlanceX

m = GlanceX.from_pretrained("Vision-Decision/Glance-X-4B")   # 也可以是自己导出的目录或训练得到的 .pt

# 一张图，多句需求：图只算一次
m.ask("tennis.jpg", ["the kid with glasses in the middle", "right racket", "左边举起球拍的男孩",
                     "boy on the left raising a blue racket", "a frisbee"])
# Result('the kid with glasses in the middle', present_prob=1.000, box=[249, 165, 381, 424], confidence=0.997, n_boxes=1)
# Result('right racket', present_prob=1.000, box=[528, 72, 632, 242], confidence=0.984, n_boxes=1)
# Result('左边举起球拍的男孩', present_prob=0.992, box=[65, 143, 253, 422], confidence=0.987, n_boxes=1)
# Result('boy on the left raising a blue racket', present_prob=0.040, box=[64, 120, 249, 422], confidence=0.038, n_boxes=0)
# Result('a frisbee', present_prob=0.000, box=[387, 93, 587, 330], confidence=0.000, n_boxes=0)

# 单个（图, 需求）对，最准；`boxes` 给出所有目标
r = m.find("giraffes.jpg", "the three giraffes in the field", mode="joint")
r.boxes   # [[47, 62, 242, 316], [333, 69, 515, 312], [522, 101, 638, 314]]

# 一句需求，很多张图：需求只算一次，结果按置信度排序
m.search("a giraffe eating from someone's hand", photos)[0]
# Result("a giraffe eating from someone's hand", present_prob=0.994, box=[307, 22, 585, 324], confidence=0.753, n_boxes=0)
```

以上输出是真实结果（CUDA，bf16），与演示视频里的预测相同。

| `Result` 字段 | 含义 |
|---|---|
| `present_prob` | P(图里有符合需求的东西)，用它来拒答 |
| `box` | 最可能的目标，`[x1, y1, x2, y2]`，输入图的像素坐标（目标不存在时也会填，请用 `present_prob` 把关） |
| `iou_prob`, `confidence` | P(IoU ≥ 0.5)，以及 `present_prob × iou_prob`，可直接用于设阈值或排序 |
| `boxes`, `box_scores` | 多框头找到的所有目标及其 objectness 分数 |
| `image_index` | 结果属于哪张输入图（`search` 用） |

每个模式也接受别名：`i2t` 对应 `image_first`，`t2i` 对应 `text_first`，`bi` 对应 `joint`。

## 结果

以下都是 **Glance-X-4B（K=8）** 在**完整**测试集上的结果，同一个 checkpoint，三个方向分别测，分两组报告。

### 1. 原始评测集（人工标注）

RefCOCO / RefCOCO+ / RefCOCOg，Acc@0.5（%）：

| 划分 | Glance-4B-preview<br><sub>（仅先图后文）</sub> | Glance-X<br>`image_first` | Glance-X<br>`text_first` | Glance-X<br>`joint` |
|---|:-:|:-:|:-:|:-:|
| RefCOCO val | **93.2** | 92.9 | 92.1 | 93.1 |
| RefCOCO testA | 94.5 | 94.8 | 93.9 | **94.9** |
| RefCOCO testB | **90.7** | 90.4 | 89.5 | 90.5 |
| RefCOCO+ val | **89.7** | 89.2 | 87.6 | 89.6 |
| RefCOCO+ testA | **93.3** | 93.0 | 91.6 | 93.1 |
| RefCOCO+ testB | 84.6 | 84.7 | 82.0 | **84.9** |
| RefCOCOg val | 90.0 | 90.0 | 89.3 | **90.3** |
| RefCOCOg test | 90.4 | **90.5** | 89.0 | **90.5** |
| **8 个划分平均** | 90.8 | 90.7 | 89.4 | **90.9** |

Glance-4B-preview 的数字来自 [Glance README](https://github.com/Vision-Decision/Glance#benchmarks)。

gRefCOCO（无目标、单目标、多目标表达），%：

| 划分 | 指标 | `image_first` | `text_first` | `joint` |
|---|---|:-:|:-:|:-:|
| val | P(present) 的 AUROC | 98.1 | 97.3 | **98.2** |
| val | 无目标表达被拒答的比例 | **79.2** | 75.9 | 78.9 |
| val | 多目标：框集合 F1 | 77.3 | 78.1 | **82.7** |
| testA | 多目标：框集合 F1 | 72.7 | 73.4 | **78.3** |
| testB | 多目标：框集合 F1 | 63.8 | 66.5 | **72.2** |

这里的 val 是 gRefCOCO val 与同一批图片上的 RefCOCO val 合并而成（这个 gRefCOCO 版本的 val 没有单目标表达）。

### 2. Glance-Data 评测集（我们自己的 benchmark）

来自 [Glance-Data](https://github.com/Vision-Decision/Glance-Data)，在 300 张留出的 RefCOCO-val 图片上共有
23,586 句需求，覆盖短、长、口语和中文需求、相似但不对的负例、以及多目标需求。部分题型：

| 题型 | 指标 | `image_first` | `text_first` | `joint` |
|---|---|:-:|:-:|:-:|
| 短 / 中 / 长描述 | Acc@0.5 | 86.7 / 90.8 / 91.3 | 86.4 / 90.5 / 92.0 | **87.6 / 91.0 / 92.0** |
| 只说类别名 | Acc@0.5 | 85.0 | 86.5 | **86.7** |
| 相似负例：属性不对 | 拒答率 | 94.9 | 93.8 | **95.0** |
| 相似负例：关系不对 | 拒答率 | 94.2 | 94.2 | **94.4** |
| 一组目标 | 至少命中 1 个 | 78.5 | 85.1 | **87.1** |
| 某类的全部实例 | 至少命中 1 个 | 61.4 | 65.0 | **66.8** |
| 全部多目标题 | F1 / 数量完全正确 | 62.6 / 39.1 | 66.7 / 43.0 | **69.7 / 46.7** |
| 英文 / 中文 | Acc@0.5 | 88.1 / 86.8 | 87.2 / 85.9 | 87.8 / 86.8 |

> [!IMPORTANT]
> Glance-Data 评测集与一部分训练数据同分布，所以这些数字衡量的是模型把新题型学得怎么样，不是零样本 benchmark。
> 标签由 pipeline 生成，抽检精度约 90%。

### 结论

- **`joint` 最准。** 在 RefCOCO 系列上与单方向的 Glance-4B-preview 持平（平均 90.9 vs 90.8），多目标明显更好
  （gRefCOCO val 上 F1 比 `image_first` 高 5.4）。
- **`text_first` 在 RefCOCO 上比 `image_first` 低约 1.3 个点**，但能找到一组目标中的更多个（Glance-Data 的"一组目标"上
  至少命中率高 6.6 个点）：带着需求去读图，有助于召回。
- **每个方向都能拒答。** gRefCOCO 上 P(present) 的 AUROC ≥ 97.3；相似、数量、位置、相近类别等负例的拒答率在 94–100%。
- **还有差距。** 我们内部一个单方向模型（从指令版 Qwen3.5-4B 出发，只用原始数据训练）在 RefCOCO 系列上高 1.5–3.8 个点。
  可能的原因（Base 与指令版起点不同、新数据占比更大、K=8 的分段）还没有拆开验证，见 [docs/RESULTS.md](docs/RESULTS.md)。

## 自己训练

整个训练在**一张 GPU** 上就能跑完：6,100 步（2 个 epoch），在**一张 H200 上约 3 小时**，显存峰值 22 GB。

```bash
python scripts/prepare_refcoco.py --ann data/ann --out data/prep            # RefCOCO / + / g、gRefCOCO -> 记录
python scripts/merge_data.py --orig data/prep/train.jsonl \
    --aug data/glance-data/aug_train_all.jsonl --out data/train.jsonl       # 加入 Glance-Data，每张图一条记录
python -m glancex.train --cfg configs/glance-x-4b-k8.json                   # K=8（默认）；K=0 用 configs/glance-x-4b-k0.json
```

| 设置 | 值 |
|---|---|
| 数据 | 35.5 万句原始需求（RefCOCO 系列 + gRefCOCO 负例）+ 183 万句 Glance-Data 需求；每张图抽到的需求中至少 1/3 来自原始数据 |
| 每个 batch 的方向 | `image_first` 50%，`text_first` 25%，`joint` 25% |
| 每张图的需求数 | 24 句（`image_first`），3 对（`text_first`、`joint`） |
| 优化 | 每步 8 张图，AdamW，学习率 2e-4（LoRA）/ 5e-4（输出头），余弦退火，bf16 + fp32 LoRA |
| 损失 | presence BCE、patch 中心 CE、四边的 Cramér + GIoU、IoU 头 BCE、objectness focal loss |

评测某个方向：

```bash
python -m glancex.evaluate --ckpt outputs/glance-x-4b-k8/ckpt_final.pt --mode joint \
    --sets refcoco_val,refcoco_testA,refcoco_testB,refcocog_val,grefcoco_val_merged \
    --glance_data data/glance-data/aug_val.jsonl --out results/k8_joint.json
```

数据目录结构、硬件说明和全部配置项见 [docs/TRAINING.md](docs/TRAINING.md)。

## 仓库结构

```
glancex/
  inference.py     GlanceX：from_pretrained / ask / find / search
  backbone.py      三方向 Qwen3.5 骨干（单模态段 + 融合段）
  _layers.py       Gated DeltaNet 与注意力层的前缀/块计算
  heads.py         presence / box / IoU / multi-box 输出头和损失
  data.py          记录 -> 图片流和文字流
  train.py         训练循环
  evaluate.py      按方向评测：原始评测集 + Glance-Data
scripts/           数据准备、导出到 Hugging Face、精确性测试
demo/              predict.py + render.py：演示视频的生成方式
configs/           K=8、K=0 和冒烟测试配置
tests/             CPU 单元测试（CI 中运行）
```

## 局限

- 只针对自然图片：训练数据是 COCO 风格的照片，不是截图或文档。
- `text_first` 和 `joint` 每个块只算一个（需求, 图）对；对一张图问很多句时，`image_first` 更高效。
- 各部分许可不同：代码为 Apache-2.0；权重为 CC BY-NC-SA 4.0（非商业），因为 gRefCOCO 与 Glance-Data 仅限非商业研究。

## 引用

```bibtex
@software{glancex2026,
  title  = {Glance-X: one model, three ways to look},
  author = {Vision-Decision},
  year   = {2026},
  url    = {https://github.com/Vision-Decision/Glance-X}
}
```

## 致谢

- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B-Base)：基座模型。
- [Glance](https://github.com/Vision-Decision/Glance)：框的做法和训练设置。
- Glance-Data（[GitHub](https://github.com/Vision-Decision/Glance-Data) ·
  [Hugging Face](https://huggingface.co/datasets/Vision-Decision/Glance-Data)）：新的需求类型。
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)：Gated DeltaNet 内核。
- 数据集：RefCOCO / RefCOCO+ / RefCOCOg、gRefCOCO、MS COCO。
- 思路来源：SAM 3（presence token）、Grounding DINO（双向融合）、VLMo / BEiT-3（可切换模式的骨干）。
