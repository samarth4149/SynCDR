# SynCDR : Training Cross Domain Retrieval Models with Synthetic Data

This code contains code for training and evaluation of SynCDR models. 
It is based upon code from the [CDS repository](https://github.com/VisionLearningGroup/CDS).

### Environment setup

Using anaconda/miniconda, run the following.
```bash
conda create -n syncdr python=3.10
conda activate syncdr
pip install -r requirements.txt
``` 

### Dataset setup
Download and extract the datasets to a location of your choice. To download domainnet, you can run the following script :
```bash
bash download_domainnet.sh /path/to/data_dir
```

Filelists for different splits are available under `data/`. 
These assume that datasets are available under `/home/ubuntu/data`.
To update the filelists to point to `/path/to/data_dir` of your choosing, run :

```bash
cd data/
python replace_path_roots.py /path/to/data_dir
cd ..
```

### Training

For training SynCDR on domainnet painting and clipart, run the following command. 
Please provide `/path/to/synthetic_data_root` (default = `../ELITE/synthetic_data`), 
and `/path/to/real_data_root` (same as `/path/to/data_dir` above). The following 
trains SynCDR where training data has the first set of classes in paintings (as 
indicated by `--train_split`) and the second set (non-overlapping with the first) 
in clipart (as indicated by `--train_split2`). 

```bash
python train.py --dataset domainnet \
                --source painting \
                --target clipart \
                --ppp \
                --syn_root /path/to/synthetic_data_root \
                --real_root /path/to/real_data_root \
                --train_split train_cls_disjoint1 \
                --train_split2 train_cls_disjoint2 \
                --add_test_eval
```
