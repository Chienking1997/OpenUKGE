# OpenUKGE
An Open Toolkit for Uncertain Knowledge Embedding

### Install

Make sure your local environment has the following installed:

```
python >=3.6
pytorch >= 1.5(GPU version)
numpy
pandas
sklearn
scipy
```

Quick Guide for Anaconda users:

```
conda create --name OpenUKGE python=3.7 cudatoolkit=10.1 cudnn
```

```
conda activate OpenUKGE 
```

Install Pytorch

```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Run the experiments:

```
python run.py --data ppi5k --model uanalogy --function rect --batch_size 1024 --dim 128 --epoch 1000 --reg_scale 0.0001
```

You can use `--model urotate` to switch to the U RotatE model.

And you can use `--function logi` to switch to the logi  mapping function.

![](https://github.com/Chienking1997/OpenUKGE/blob/master/docs/framework.png)

