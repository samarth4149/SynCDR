
# Contrastive Unpaired Translation (CUT)

This folder contains code to train constrastive unpaired translation (CUT) models given unlabeled data from a pair of domains and generating synthetic data using these trained models. This builds upon code in [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation).

### Environment Setup
Create a new conda environment using `conda env create -f environment.yml`.


### CUT Training

For training a CUT model for translation from paintings to cliparts in the domainnet dataset run the following.

```bash
python train.py --filelist_root path/to/filelist_root \
                --dataset domainnet \
                --source painting \
                --target clipart \
                --train_split train_cls_disjoint1 \
                --train_split2 train_cls_disjoint2 \
                --batch_size 4 \
                --name domainnet_p2c_12
```

For the above command, please provide the path to the filelist root `path/to/filelist_root` (default = `../SynCDR/data`). 
The above command trains the model for 5000 training steps (default) using images from `train_cls_disjoint1` split of 
the painting domain and `train_cls_disjoint2` split of the clipart domain (Note that these are the images available for 
training SynCDR in a given run. The splits have no category overlap). Checkpoints for the above training run are saved in 
`checkpoints/domainnet_p2c_12/`.

### Synthetic Data Generation 

For synthetic data generation using the CUT model trained above, run the following.

```bash
python inference.py --filelist_root path/to/filelist_root \
                    --dataset domainnet \
                    --source painting \
                    --target clipart \
                    --train_split train_cls_disjoint1 \
                    --train_split2 train_cls_disjoint2 \
                    --root_dir path/to/synthetic_data_output_dir \
                    --name domainnet_p2c_12
```

Provide the path to synthetic data output and filelist root above. Note that 
the name `domainnet_p2c_12` is used to load the latest checkpoint from the directory 
`checkpoints/domainnet_p2c_12`. 
