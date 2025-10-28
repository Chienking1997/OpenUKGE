# OpenUKGE  
An Open‑Source Python Library for Uncertain Knowledge Graph Embedding

[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9D%A4-red?logo=pytorch)](https://pytorch.org/)

## Overview  
Uncertain knowledge graphs (UKGs), which associate each triple with a confidence score, enable more reliable knowledge completion and uncertainty-aware reasoning compared to deterministic knowledge graphs. However, heterogeneous implementations, diverse programming languages, and inconsistent evaluation settings of existing uncertain knowledge graph embedding (UKGE) methods hinder fair comparison and practical adoption.

**OpenUKGE** addresses these challenges by providing a unified, reproducible Python framework for UKGE research, supporting standardized model implementation, dataset integration, and evaluation protocols.

## Key Features  
- Support embedding of uncertain knowledge graphs: triples of the form (h, r, t, confidence)  
- Supports multiple embedding models implemented in Python under one unified API  
- Integrates benchmark UKG datasets for knowledge completion and confidence estimation tasks  
- Provides standard and novel evaluation protocols for ranking, classification, and confidence prediction  
- Offers reproducibility and extensibility: you can reproduce published results, plug in new models, datasets or metrics  
- Licensed under GPL‑3.0 and designed for research and open‑source usage  

## Installation  
```bash
git clone https://github.com/Chienking1997/OpenUKGE.git  
cd OpenUKGE  
pip install -r requirements.txt  
```

## Quick Start  
1. Prepare your uncertain knowledge graph data: entities, relations, triples and confidence scores.  
2. Modify the example script (`nl27k_UKGE_demo.py`) for your data paths and hyper‑parameters.  
3. Run the demo:  
   ```bash
   python nl27k_UKGE_demo.py
   ```
4. Inspect the outputs: training log, embedding results, evaluation metrics for tasks such as confidence prediction or ranking.  

## Project Structure (overview)  
- `openukge/` – main source code directory  
- `datasets/` – UKG benchmark datasets included or interface to load external UKG data  
- `experiments/` – configuration scripts, example runs, evaluation pipelines  
- `nl27k_UKGE_demo.py` – example demonstration script  
- `requirements.txt` – python dependencies  
- `LICENSE` – GPL‑3.0 Open Source License  

## 🪪 License  
This project is licensed under the **GNU General Public License v3.0 (GPL‑3.0)**.
See [LICENSE](LICENSE) for details.

## Citation  
If you use this library in your research, please cite the associated work:  

## 🤝 Acknowledgements

OpenUKGE builds upon:
- [PyTorch](https://pytorch.org/) 
- [UKGE (Chen et al., 2019)](https://github.com/stasl0217/UKGE)
- [UKGsE (Yang et al., 2020)](https://github.com/ShihanYang/UKGsE)
- [PASSLEAF (Chen et al., 2021)](https://github.com/Franklyncc/PASSLEAF)
- [BEUrRE (Chen et al., 2021)](https://github.com/stasl0217/beurre)
- [unKR (Wang et al., 2024)](https://github.com/seucoin/unKR)


---

## Contributing  
Contributions are very welcome. You can:  
- Submit issues to report bugs or request features  
- Fork the repository and create pull requests  
- Extend the framework by adding new models, new datasets, or new evaluation metrics  

---

We hope **OpenUKGE** becomes your go‑to tool for uncertain knowledge graph representation learning and drives advancement in this research area.
