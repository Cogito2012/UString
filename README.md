# VGRNN_AA
Variational Graph RNN for Accident Anticipation

### Install Dependencies

For Linux System with CUDA-9.2:

```shell
conda install pytorch==1.4.0 torchvision cudatoolkit=9.2 -c pytorch
```

Note: use the following command to install torch_sparse
```
pip install https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.4.0/torch_sparse-latest%2Bcu92-cp36-cp36m-linux_x86_64.whl
pip install https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.4.0/torch_scatter-latest%2Bcu92-cp36-cp36m-linux_x86_64.whl
pip install https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.4.0/torch_cluster-latest%2Bcu92-cp36-cp36m-linux_x86_64.whl
```


### Note Update
For this branch, we remove the variational inference part with KL loss, and just use the simplest GCN + GRU method