# Installation Instructions (OpenUKGE)
## 1. Install PyTorch (Required)
Visit https://pytorch.org/ and select the installation command corresponding to your system and CUDA version.

Example (CPU-only):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Example (GPU+CUDA 12.1):
```bash
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. Install OpenUKGE package
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