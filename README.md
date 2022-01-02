# Metapath2vec  

`metapath2vec` is the algorithm that enables scalable representation learning for heterogeneous networks.  

This repository contains training code of `metapath2vec`.  

This implementation is based on **Pytorch API**.  

논문을 한국어로 리뷰한 글은 [이 곳](https://greeksharifa.github.io/machine_learning/2021/12/11/metapath2vec/) 에서 확인할 수 있습니다.  

---
## Overview  
The most important files in this project are as follows:  

- analyzing torch_geometric metapath2vec class: [notebook](https://github.com/hoopoes/metapath2vec/blob/main/analyzing_metapath2vec_class.ipynb)  
- hyperparameter tuning with optuna: [notebook](https://github.com/hoopoes/metapath2vec/blob/main/tuning_hyperparameters_with_optuna.ipynb)  
- training metapath2vec model with wandb: [python file](https://github.com/hoopoes/metapath2vec/blob/main/train.py)  

## Setup  
To run the code, you need the following dependencies:  

- matplotlib, seaborn  
- pandas, numpy, scikit-learn  
- optuna, nbformat, plotly  
- tqdm, wandb  
- torch==1.9.0  
- torch-geometric==2.0.3  
- torch-scatter==2.0.9  
- torch-sparse==0.6.12  

You might not need the exact version.  

## Usage  
Execute the following scripts to train `metapath2vec`:  

```python
python train.py -rn run_name -e num_epochs
```

**run_name** represents the wandb run name. Default value is "new run".  
**num_epochs** represents the number of epochs. Default value is 5.  


---
## Results  
My experiments were conducted on a machine with a **NVIDIA GeForce RTX 3070 Ti** (8GB memory), **6-Core AMD Ryzen 5 5600X CPU** (3.70 GHz) and 24 GB of RAM.  

After running 15 epochs, the results are as follows:  

**Train Loss Plot**  
<center><img src="/img/train_loss.png" width="80%"></center>  

**Test Accuracy Plot**  
<center><img src="/img/train_loss.png" width="80%"></center>  

**Visualization of venue embedding vector**  
Embedding vectors are compressed using `tsne` algorithm.  

If venues have same label, it means that they are similar venues. For instance, SIGIR and RecSys have same label and CVPR and ICCV have same label.  

After 1 epoch, you can see that embedding vectors that have same label are not that close.  

<center><img src="/img/e1.png" width="80%"></center>  

But after 15 epochs they do have similar values, which means that training process is finished well.  

<center><img src="/img/e15.png" width="80%"></center>  


---
## Acknowledgements  
- original paper: [link](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)  
- torch geometric metapath2vec class: [link](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/metapath2vec.html)
- AMiner dataset source code: [link](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/aminer.html#AMiner)  

