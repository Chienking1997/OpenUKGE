# OpenUKGE  
An Open‑Source Python Library for Uncertain Knowledge Graph Embedding

[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9D%A4-red?logo=pytorch)](https://pytorch.org/)

## 🌍 Overview  
Uncertain knowledge graphs (UKGs), which associate each triple with a confidence score, enable more reliable knowledge completion and uncertainty-aware reasoning compared to deterministic knowledge graphs. However, heterogeneous implementations, diverse programming languages, and inconsistent evaluation settings of existing uncertain knowledge graph embedding (UKGE) methods hinder fair comparison and practical adoption.

**OpenUKGE** addresses these challenges by providing a unified, reproducible Python framework for UKGE research, supporting standardized model implementation, dataset integration, and evaluation protocols.

## 🧩 Key Features  
- **Native UKG Support**: Natively processes quadruples of the form `(head, relation, tail, confidence)`  
- **Unified Model Interface**: Multiple UKGE models with consistent APIs for easy comparison  
- **Benchmark Datasets**: Integrated with standard UKG datasets (cn15k, nl27k, ppi5k, onet20k) and few-shot variants, with automatic download support via `download_dataset` utility  
- **Comprehensive Evaluation**: Supports confidence prediction (MSE, MAE, ECE), link prediction (MR, MRR, Hit@k), and ranking quality (nDCG) metrics  
- **Advanced Training Utilities**: Includes specialized trainers with semi-supervised learning and few-shot learning support  
- **High Reproducibility**: Standardized experimental pipelines to ensure consistent and comparable results  
- **Extensible Architecture**: Easy to extend with new models, datasets, or evaluation metrics

## 📦 Installation
### 1. Install PyTorch for your system 
⚙️OpenUKGE depends on PyTorch, but it is not included in the default dependency list. The reason is: 

1. Different hardware configurations require different PyTorch builds.
2. Official recommendation: PyTorch developers advise users to install it manually using the appropriate wheel for their environment.
3. Avoiding compatibility issues: Including PyTorch directly in install_requires often causes failed installations or large downloads on systems without matching CUDA libraries.

Example for torch 2.3.0 + CUDA 12.1:
```bash
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```
Example (CPU-only):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
### 2. Install OpenUKGE package
+ Install from source
```bash
git clone https://github.com/Chienking1997/OpenUKGE.git  
cd OpenUKGE  
pip install .
```
+ Install by pypi
```bash
pip install OpenUKGE
```
---

## 🚀 Quick Start
Take training the UKGE model using the NL27K dataset as an example.  
### 1️⃣ Import the required modules
```python
from openukge.data import nl27k
from openukge.models import UKGE
from openukge.training import UKGETrainer, OptimBuilder, EarlyStop
from openukge.loss import UKGELoss
from openukge.utils import seed_everything
```
### 2️⃣ Fix seeds, load data, define model, loss, optimizer & early stopping for modules
```python
seed_everything()
data = nl27k.load_data('data', num_neg=10, batch_size=512)
model = UKGE(data.num_ent, data.num_rel, emb_dim=128)
loss = UKGELoss()
opt = {'optimizer': {'type': 'Adam', 'lr': 0.001}}
optimizer = OptimBuilder(opt)
early_stop = EarlyStop(patience=2, min_delta=0, 
                       monitor="mse", mode="min", 
                       monitor_mode="tail")
```
### 3️⃣ Train and valid the model
```python
trainer = UKGETrainer(data, model, loss, 
                      optimizer, early_stop, 
                      save_path="best_nl27k-mse.pt")
trainer.fit(epochs=100, eval_freq=2)
```
### 4️⃣ Output the comprehensive evaluation results of the model
```python
trainer.test()
```

## 📂 Project Structure (overview)  
- `openukge/` – main source code directory
- `nl27k_UKGE_demo.py` – example demonstration script  
- `requirements.txt` – python dependencies  
- `LICENSE` – GPL‑3.0 Open Source License  

## 🪪 License  
This project is licensed under the **GNU General Public License v3.0 (GPL‑3.0)**.

See [LICENSE](LICENSE) for details.

## 🤝 Acknowledgements

OpenUKGE builds upon:
- [PyTorch](https://pytorch.org/) 
- [UKGE (Chen et al., 2019)](https://github.com/stasl0217/UKGE)
- [UKGsE (Yang et al., 2020)](https://github.com/ShihanYang/UKGsE)
- [PASSLEAF (Chen et al., 2021)](https://github.com/Franklyncc/PASSLEAF)
- [BEUrRE (Chen et al., 2021)](https://github.com/stasl0217/beurre)
- [unKR (Wang et al., 2024)](https://github.com/seucoin/unKR)
---

## 💡 Contributing  
Contributions are very welcome. You can:  
- Submit issues to report bugs or request features  
- Fork the repository and create pull requests  
- Extend the framework by adding new models, new datasets, or new evaluation metrics  

---
## 📬 Contact

For questions, suggestions, or collaborations:  
📧 **chienking1997@outlook.com**

We hope **OpenUKGE** becomes your go‑to tool for uncertain knowledge graph representation learning and drives advancement in this research area.
