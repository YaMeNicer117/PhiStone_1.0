# PhiStone_1.0

PhiStone_1.0 是面向蛋白结合口袋的三维分子生成项目。项目通过 PhiSSeparator 对蛋白与配体进行片段化表示，以片段嵌入、三维位置和局部参考系构建训练数据，再使用 PhiSSE3TD 等变扩散模型学习并生成候选配体。

项目支持两类生成任务：

- **Creative**：以蛋白口袋为条件，从头生成候选分子，适用于没有固定配体片段的分子设计。
- **Survival**：以蛋白口袋和保留的先导化合物片段为条件进行片段生长与连接，适用于先导化合物的优化修饰。

训练、推理和实验中使用的权重、固定节点设置及词表组合应与相应任务保持一致。

## 1. 项目结构

```text
PhiStone_1.0/
├── CHEMCUT/
│   ├── datas/
│   │   ├── train_datas/
│   │   │   ├── BioLiP/                         # 数据下载、整理、清理与口袋处理
│   │   │   └── PDBbind/                        # 数据清理与口袋处理
│   │   └── work_datas/SE3TD/
│   │       ├── TASK_A/                         # 带参考配体的实验复合物 PDB
│   │       └── TASK_B/                         # Creative 条件输入 PDB
│   ├── processed_data/
│   │   ├── train_projector.py                  # 768 → 64 维片段嵌入投影训练
│   │   ├── Add_hac.py                          # 数据嵌入替换与重原子数补充
│   │   ├── fragment_vocabulary_768.json
│   │   ├── fragment_vocabulary_64_dynamic_margin.json
│   │   ├── normalization_stats_2.9w.json
│   │   ├── SE3TD_Pre_datas_64_dynamic/          # 下载、解压后的预训练数据
│   │   ├── SE3TD_fine_datas_64_dynamic/         # 下载、解压后的微调数据
│   │   ├── TASK_A/                             # Survival 推理条件
│   │   └── TASK_B/                             # Creative 推理条件
│   ├── PhiSSeparator SE3TD PDBbind 多线程.ipynb
│   ├── PhiSSeparator SE3TD BioLiP 多线程.ipynb
│   ├── Tools_vocabulary_separator.ipynb
│   └── PhiSSeparator Generation TASKAB MEB.ipynb
├── SE3TD/
│   ├── PhiSSE3TD_config.py                     # 数据、训练、模式和权重路径配置
│   ├── PhiSSE3TD_train_pretrain.py             # 预训练
│   ├── PhiSSE3TD_train_Finetune_Survival.py     # 微调
│   ├── PhiSSE3TD_generate_ddp.py               # 多进程、多 GPU 并行生成
│   └── processed_data/                        # 权重、训练日志及生成结果
└── GATv2/
```

上图中的两个训练数据目录需要下载解压或通过预处理生成。本文路径以项目根目录为起点，统一使用 `/` 分隔符。

## 2. 运行环境

预处理使用 Jupyter Notebook；训练脚本使用 PyTorch 分布式训练和 NCCL 后端，训练命令面向具备 NVIDIA GPU 的 Linux 服务器。

主要依赖包括：

- PyTorch、PyTorch Geometric、torch-scatter、e3nn。
- RDKit、Open Babel、NumPy、SciPy、pandas。
- transformers、tqdm、TensorBoard。
- Jupyter、IPython、ipywidgets、py3Dmol、jupyter-ui-poll，用于预处理与交互式生成条件设置。

PyTorch、CUDA 和 PyTorch Geometric 扩展需要在服务器中配套安装。预处理 Notebook 使用 `CHEMCUT/ChemBERTa_Local/` 中的本地 ChemBERTa 模型，可从第 6.2 节提供的网盘下载模型包并解压至该目录；部分数据预处理 Notebook 在该目录不存在时会尝试下载模型。重新运行 BioLiP 下载脚本还需要其调用的 AWS CLI。

## 3. 数据来源与下载

### 3.1 原始三维复合物数据

训练使用的三维复合物 PDB 与配体 SDF 数据来源于 **BioLiP** 和 **PDBbind**。数据下载、整理、清理及去重相关流程可在以下目录中查阅：

- `CHEMCUT/datas/train_datas/BioLiP/`：包含 `Download_organize.py`、`Organize_dataset.py`、`Clean_biolip.py`、`Crop_pockets.py` 及相关 Notebook。
- `CHEMCUT/datas/train_datas/PDBbind/`：包含 `clen_PDBbind.py` 及相关 Notebook。

复现原始数据处理时，应按照相应数据库的数据使用要求获取数据，并检查脚本中的输入目录与输出目录。部分清理脚本会重建输出目录，运行前应确认其指向待处理数据的位置。

### 3.2 预处理后的训练数据

可直接下载已加工的 64 维训练数据，以进入后续训练流程：

| 数据文件 | 用途 | 下载链接 | 提取码 |
|---|---|---|---|
| `SE3TD_Pre_datas_64_dynamic.tar.gz` | 预训练数据 | [百度网盘](https://pan.baidu.com/s/1aI-7Yq3RYrs8pGMFlEapQQ?pwd=q6gw) | `q6gw` |
| `SE3TD_fine_datas_64_dynamic.tar.gz` | 微调数据 | [百度网盘](https://pan.baidu.com/s/1LaLLbbvY8gvgOrggZFKn0Q?pwd=4f8j) | `4f8j` |

解压后，将对应数据目录放在 `CHEMCUT/processed_data/` 下，并在训练前将 `SE3TD/PhiSSE3TD_config.py` 中的 `PYG_DATA_DIR` 指向本阶段使用的数据目录。若压缩包带有额外的父目录，应以解压后的实际目录层级为准。

## 4. 数据预处理与词表加工

直接使用上述已处理数据时，可跳过原始训练数据的重新加工；推理条件仍需按第 7 节进行预处理。

### 4.1 三维复合物预处理

以下两个 Notebook 用于原始数据的片段化、特征加工、词表收集和 PyTorch Geometric 数据构建：

- `CHEMCUT/PhiSSeparator SE3TD PDBbind 多线程.ipynb`
- `CHEMCUT/PhiSSeparator SE3TD BioLiP 多线程.ipynb`

建议以 `CHEMCUT/` 为 Notebook 工作目录，先检查 `DATA_ROOT`、输出目录、嵌入维度及并行工作进程数，再执行预处理单元。Notebook 中部分原始数据路径沿用了实验环境的 `datas/tain_datas/SE3TD/...` 结构，应按实际下载和整理后的路径修改。

本项目的 64 维动态嵌入加工流程以 **768 维原始片段嵌入**为输入。当前 PDBbind Notebook 输出 768 维数据，而 BioLiP Notebook 的默认输出为 128 维；从原始数据重新构建本流程时，应先将 BioLiP Notebook 相应变量的原定义统一为：

```python
TARGET_EMBEDDING_DIM = 768
PYG_DATA_DIR = os.path.join(output_dir, 'SE3TD_Pre_datas_768')
VOCAB_FILE = os.path.join(output_dir, 'fragment_vocabulary_768.json')
```

两个来源的数据应使用统一的原始片段词表，并保留后续推理所需的归一化统计文件。

### 4.2 训练 64 维片段嵌入投影

`CHEMCUT/processed_data/train_projector.py` 使用已有的 768 维词嵌入训练投影网络，输出：

- `fragment_vocabulary_64_dynamic_margin.json`：64 维片段词表。
- `fragment_projector.pth`：投影网络权重。

投影训练同时使用基于重原子数的动态边界损失与嵌入相似度保持约束。当前文件定义了训练函数，但没有在文件末尾自动调用；在 `CHEMCUT/processed_data/` 中可按以下方式启动：

```bash
python -c "from train_projector import train_dynamic_projection_space; train_dynamic_projection_space()"
```

输入词表为同目录下的 `fragment_vocabulary_768.json`。重新加工时，训练数据与推理条件应使用同一套片段嵌入表示。

### 4.3 替换训练数据嵌入并补充 HAC

`CHEMCUT/processed_data/Add_hac.py` 根据原始词表匹配数据中的片段，将 `frag_embeds` 替换为对应的 64 维嵌入，并增加重原子数属性 `hac`：

| 输入目录 | 输出目录 |
|---|---|
| `SE3TD_Pre_datas_768/` | `SE3TD_Pre_datas_64_dynamic/` |
| `SE3TD_fine_datas_768/` | `SE3TD_fine_datas_64_dynamic/` |

该脚本以当前工作目录构建路径。从 `CHEMCUT/` 执行时，`PROCESSED_DIR` 会指向 `CHEMCUT/processed_data/`；应同时将两个词表路径的原定义改为该目录下的文件：

```python
VOCAB_768_FILE = os.path.join(PROCESSED_DIR, 'fragment_vocabulary_768.json')
VOCAB_64_FILE = os.path.join(PROCESSED_DIR, 'fragment_vocabulary_64_dynamic_margin.json')
```

随后在 `CHEMCUT/` 中执行：

```bash
python processed_data/Add_hac.py
```

### 4.4 划分推理词表

`CHEMCUT/Tools_vocabulary_separator.ipynb` 根据片段的化学组成、环结构、重原子数及力场参数支持情况，将 64 维词表划分为 `frame`、`linker`、`puppy`、`filter` 和 `valid` 等集合。

推理时通过配置文件中的 `VOCAB_FILES_MAP` 和 `MODE_VOCAB_SETTINGS` 选择词表。当前主配置中，Survival 使用 `frame` 与 `linker`，Creative 使用 `valid`；固定锚点使用完整的 `fragment_vocabulary_64_dynamic_margin.json`。若实验使用其他词表组合，应同步调整配置指向实际文件。

## 5. 模型训练

### 5.1 预训练

训练脚本为 `SE3TD/PhiSSE3TD_train_pretrain.py`。在 `SE3TD/PhiSSE3TD_config.py` 中，将训练数据路径设置为：

```python
PYG_DATA_DIR = os.path.join(CHEMCUT_DATA_DIR, 'SE3TD_Pre_datas_64_dynamic')
```

随后在 `SE3TD/` 中启动分布式训练。以下为使用两张 GPU 的示例，`--nproc_per_node` 应按实际使用的 GPU 数量设置：

```bash
torchrun --standalone --nproc_per_node=2 PhiSSE3TD_train_pretrain.py
```

预训练的学习率、轮数、预热策略和保存路径由 `PRETRAIN_*`、`CHECKPOINT_DIR_PRETRAIN`、`LOG_DIR_PRETRAIN` 及 `BEST_MODEL_PATH_PRETRAIN` 等配置项控制。

### 5.2 Survival 微调

训练脚本为 `SE3TD/PhiSSE3TD_train_Finetune_Survival.py`。将 `PYG_DATA_DIR` 改为微调数据目录：

```python
PYG_DATA_DIR = os.path.join(CHEMCUT_DATA_DIR, 'SE3TD_fine_datas_64_dynamic')
```

当前主配置文件尚未定义该脚本使用的 `LOG_DIR_FINETUNE_S` 和 `BEST_MODEL_PATH_FINETUNE_S`。运行前应在已有保存路径配置中补充这两个变量，例如：

```python
LOG_DIR_FINETUNE_S = os.path.join(SAVE_ROOT, 'logs_finetuned_64_Survival')
BEST_MODEL_PATH_FINETUNE_S = os.path.join(
    CHECKPOINT_DIR_FINETUNE, 'best_finetuned_S_model_64.pt'
)
```

将预训练基座权重放在 `BEST_MODEL_PATH_PRETRAIN` 指定的位置，然后在 `SE3TD/` 中启动：

```bash
torchrun --standalone --nproc_per_node=2 PhiSSE3TD_train_Finetune_Survival.py
```

微调脚本优先加载已有的 `BEST_MODEL_PATH_FINETUNE_S` 断点；没有微调断点时加载 `BEST_MODEL_PATH_PRETRAIN`。首次从预训练基座开始微调时只加载模型参数，微调轮次从 0 开始。两类权重都不存在时，脚本会随机初始化训练。

### 5.3 固定节点数量与训练任务设置

实验中通过手动调整固定节点数量及其采样比例开展训练。相应参数位于两个训练脚本的 `train_epoch()` 和 `validate_epoch()` 中，包括无锚点分支概率、`keep_ratio` 和 `num_fixed`。这里的节点指片段节点。

当前源码中的实际设置为：

| 脚本 | 训练掩码 | 验证掩码 |
|---|---|---|
| `PhiSSE3TD_train_pretrain.py` | 40% 无固定节点；60% 按约 10%～60% 的比例保留节点 | 与训练采用相同的混合设置 |
| `PhiSSE3TD_train_Finetune_Survival.py` | 80% 无固定节点；20% 保留一个固定节点 | 全部不保留固定节点 |

上述概率针对节点数大于 1 的配体；单节点配体不保留锚点。微调脚本部分注释与实际数值不同，实验设置应以执行代码为准。调整固定节点策略时，应同时确定训练与验证的任务设置；生成配置中的 `DEFAULT_GENERATION_MODE` 不会改变这些训练掩码。

两个脚本使用相同的特征、位置、参考系噪声预测损失，以及边拓扑、配体内部距离、配体与蛋白距离和碰撞约束。固定节点作为干净条件参与训练，基础去噪损失仅作用于自由节点。模型按验证集总损失保存最佳权重，日志可通过 TensorBoard 查看。

## 6. 权重与配套模型下载

### 6.1 生成模型权重

| 分享内容 | 用途 | 下载链接 | 提取码 |
|---|---|---|---|
| `survival` | Survival 模式实验权重 | [百度网盘](https://pan.baidu.com/s/1dkbTiHOhCDBKsAZgt-rrlw?pwd=9jja) | `9jja` |
| `creative` | Creative 模式实验权重 | [百度网盘](https://pan.baidu.com/s/1ZuzbC5XLWsu4EnBXwx62NQ?pwd=mj32) | `mj32` |
| `best_finetuned_S_model_64-100.tar.gz` | 用于后续微调的预训练基座权重 | [百度网盘](https://pan.baidu.com/s/1ihmYDLxsZWgZif3FZDMGdA?pwd=nmf4) | `nmf4` |

下载压缩包后，先解压取得实际的 `.pt` 权重文件，再根据训练或推理配置设置其路径。`best_finetuned_S_model_64-100.pt` 在本次实验中作为预训练基座使用。

### 6.2 配套模型

以下两个配套模型包通过网盘单独提供，使用时下载解压并恢复到对应目录：

| 模型包 | 用途 | 解压后放置目录 | 下载链接 | 提取码 |
|---|---|---|---|---|
| `size_predictor_64.tar` | 节点数量预测器，可选使用 | `SE3TD/processed_data/size_predictor_64/` | [百度网盘](https://pan.baidu.com/s/1QKx6WifN1hjFMQtSzNxP8w?pwd=3gxi) | `3gxi` |
| `ChemBERTa_Local.tar` | 数据与生成条件预处理使用的 ChemBERTa 模型 | `CHEMCUT/ChemBERTa_Local/` | [百度网盘](https://pan.baidu.com/s/1B0NBfxy_Ej-FsbgKtLyZZQ?pwd=jppg) | `jppg` |

启用节点数量预测器时，默认加载文件为 `SE3TD/processed_data/size_predictor_64/pocket_size_predictor_64.pt`。若生成模块部署在其他目录，可通过 `SIZE_PREDICTOR_MODEL_PATH` 指向实际解压位置。

**节点数量预测器不是必需组件。** 可以不下载该模型包，在配置文件中关闭预测器，并手动指定待生成的自由片段节点数量，具体设置见第 7.2 节。

## 7. 模型推理

### 7.1 预处理生成条件

使用 `CHEMCUT/PhiSSeparator Generation TASKAB MEB.ipynb` 预处理生成条件输入。以 `CHEMCUT/` 为工作目录，并使用与训练配套的 64 维词表和归一化统计文件。

Notebook 的 `TASK_MODE` 控制输入来源：

- `TASK_MODE = 'TASK_A'`：读取 `datas/work_datas/SE3TD/TASK_A/` 中带参考配体的复合物 PDB，通过交互界面设置需要保留或删除的配体片段。
- `TASK_MODE = 'TASK_B'`：读取 `datas/work_datas/SE3TD/TASK_B/` 中的蛋白 PDB，确认生成中心与口袋范围，构建 Creative 条件输入。

按 Notebook 的交互提示完成任务处理，输出条件应具有以下结构：

```text
CHEMCUT/processed_data/
├── TASK_A/
│   └── <task_id>/
│       ├── complex.pt         # Survival：口袋与固定配体片段
│       └── pocket.pdb
└── TASK_B/
    └── <task_id>/
        ├── <task_id>.pt       # Creative：口袋条件
        └── pocket.pdb
```

生成脚本会按所选模式扫描对应任务目录，每个任务需要同时包含 `.pt` 条件文件和 `pocket.pdb`。

### 7.2 设置模式与权重

在 `SE3TD/PhiSSE3TD_config.py` 中设置生成模式：

```python
DEFAULT_GENERATION_MODE = 'SURVIVAL'
# 或将上面的值设为 'CREATIVE'
```

模式选择与权重选择分别配置。按默认权重路由运行时，设置：

```python
DEFAULT_MODEL_CHOICE = 'PRETRAIN'
```

将本次推理选用的实际 `.pt` 权重文件复制到以下位置，并命名为 `best_pretrained_model_64.pt`：

```text
SE3TD/processed_data/checkpoints_pretrained_64/best_pretrained_model_64.pt
```

若服务器将生成模块部署为 `SE3TD_generation/`，则对应位置为：

```text
SE3TD_generation/processed_data/checkpoints_pretrained_64/best_pretrained_model_64.pt
```

路径由生成模块内的配置文件位置决定；当前仓库目录名为 `SE3TD`。该默认路由也可加载本次选用的微调权重，具体内容由放入的权重文件决定。

也可以保留下载后的权重文件名，在配置文件中修改 `BEST_MODEL_PATH_PRETRAIN` 的原定义，并确保 `MODEL_WEIGHT_ROUTES['PRETRAIN']` 指向该路径。例如，使用推荐的 Survival 权重时：

```python
BEST_MODEL_PATH_PRETRAIN = os.path.join(
    CHECKPOINT_DIR_PRETRAIN, 'best_finetuned_S_model_64-100+58.pt'
)
```

节点数量可以由预测器确定，也可以手动指定。

使用第 6.2 节提供的节点数量预测器时，准备 `SIZE_PREDICTOR_MODEL_PATH` 指定的权重，并设置：

```python
USE_SIZE_PREDICTOR = True
```

手动指定节点数量时，关闭预测器并设置 `INPUT_NOISE_NODES`，例如：

```python
USE_SIZE_PREDICTOR = False
INPUT_NOISE_NODES = 6
```

`INPUT_NOISE_NODES` 表示待生成的自由片段节点数。Creative 模式下，该示例生成 6 个片段节点；Survival 模式下，保留的固定片段节点另计，再生成 6 个自由片段节点。该参数控制生成起始节点数，经过连通性筛选后，最终保留的片段数可能减少。

### 7.3 启动并行生成

在 `SE3TD/` 中执行：

```bash
python PhiSSE3TD_generate_ddp.py
```

脚本自行创建多进程工作池，并为每个进程加载模型。默认使用可用的 GPU 0～6，按每张卡的总显存分配工作进程；可在脚本入口调整 `TARGET_GPUS` 和每卡进程分配方式，以适配服务器资源。

生成任务数量、每个口袋的候选数量、最大尝试次数与采样步数分别由 `NUM_POCKETS_TO_PROCESS`、`LIGANDS_PER_POCKET`、`MAX_GENERATION_ROUNDS` 和 `DDIM_STEPS` 控制。多个进程共享每个口袋的成功数及尝试数，在达到目标数量或尝试上限时结束该任务。

### 7.4 输出结果

结果保存到 `OUTPUT_DIR_GENERATION` 指定的位置，默认位于 `SE3TD/processed_data/generated_molecules/`。并行生成时，每个结果目录位于对应的 `worker_<id>/` 下：

```text
generated_molecules/
└── worker_<id>/
    └── <task_id>_run<index>/
        ├── 1_original_all_fragments.sdf
        ├── 1.5_translated_fragments.sdf
        ├── 2_mmff94_optimized_ligand.sdf
        └── 3_mmff94_complex.pdb
```

这些文件分别用于查看原始解码片段、处理后保留的片段、组装并优化后的配体及其与蛋白口袋组成的复合物。生成过程包含解码、连通性、成键规划及分子构建筛选，最终候选数取决于各任务的成功情况。

## 8. 本次实验与推荐权重

`CHEMCUT/datas/work_datas/SE3TD/` 提供了本次实验使用的先导化合物复合物 PDB。实验以白皮杉醇为先导化合物，在生成条件预处理阶段删除其中间的乙烯基，将保留片段作为生成起点，并使用 Survival 模式下的多个权重分别生成候选分子。

本次实验推荐的权重如下：

| 使用场景 | 推荐权重 |
|---|---|
| Survival：先导化合物优化修饰 | `best_finetuned_S_model_64-100+58.pt`、`best_finetuned_S_model_64-100+75.pt` |
| Creative：从头生成分子 | `best_finetuned_S_model_64-100+72.pt`、`best_finetuned_S_model_64-100+40.pt` |
| 预训练基座：用于后续微调 | `best_finetuned_S_model_64-100.pt` |

复现实验时，先在生成条件 Notebook 中完成乙烯基删除与固定片段确认，再将 `DEFAULT_GENERATION_MODE` 设为 `'SURVIVAL'`，逐一选择相应权重运行生成。比较不同权重时，建议为各次运行设置独立的 `OUTPUT_DIR_GENERATION`，并记录权重文件名、词表组合和生成参数，便于对应实验结果。
