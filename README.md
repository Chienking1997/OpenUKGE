# OpenUKGE  
An Open‑Source Python Library for Uncertain Knowledge Graph Embedding

## Overview  
Uncertain knowledge graphs (UKGs), which associate each triple with a confidence score, have recently attracted growing attention for their ability to represent and reason under uncertainty. Compared with deterministic knowledge graphs, UKGs enable more reliable knowledge completion and confidence estimation, but they also bring new challenges to embedding and evaluation.  
Despite the rapid development of uncertain knowledge graph embedding (UKGE) methods, their fair comparison and practical adoption remain difficult due to heterogeneous implementations, programming languages, and evaluation settings.

In this paper, we introduce **OpenUKGE**, an open‑source Python library for unified and reproducible representation learning on UKGs. OpenUKGE integrates a wide range of embedding models, benchmark datasets, and both standard and newly designed evaluation protocols within a single framework. This allows researchers to perform fair and comprehensive evaluations, efficiently reproduce existing methods, and easily extend the framework with their own models. We hope that OpenUKGE can serve as a standardized and extensible platform to accelerate research on uncertain knowledge graphs.  

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

## License  
This project is licensed under the **GNU General Public License v3.0 (GPL‑3.0)**.

## Citation  
If you use this library in your research, please cite the associated work:  
> Chen, X., Chen, M., Shi, W., Sun, Y., Zaniolo, C. *Embedding Uncertain Knowledge Graphs*. AAAI 2019.

## Contributing  
Contributions are very welcome. You can:  
- Submit issues to report bugs or request features  
- Fork the repository and create pull requests  
- Extend the framework by adding new models, new datasets, or new evaluation metrics  

---

We hope **OpenUKGE** becomes your go‑to tool for uncertain knowledge graph representation learning and drives advancement in this research area.
