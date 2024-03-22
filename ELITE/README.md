# ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation

This folder contains code for generating synthetic data using ELITE. 
This builds upon code in the [ELITE github repository](https://github.com/csyxwei/ELITE). 

### Environment Setup

```shell
git clone https://github.com/csyxwei/ELITE.git
cd ELITE
conda create -n elite python=3.9
conda activate elite
pip install -r requirements.txt
```

### Pretrained Checkpoint

The pretrained ELITE checkpoints are in [Google Drive](https://drive.google.com/drive/folders/1VkiVZzA_i9gbfuzvHaLH2VYh7kOTzE0x?usp=sharing). 
Please download the checkpoint `global_mapper.pt` and place it in `checkpoints`.

### Synthetic Data Generation

For generating synthetic data with ELITE, use the script `gen_data.py` as follows: 

```bash

python gen_data.py --root_dir path/to/synthetic_data_output_dir \
                   --dataset domainnet \
                   --source painting \
                   --target clipart \
                   --batch_size 4 \
                   --filelist_root path/to/filelist_root 
```

The above command will produce clipart images for each domainnet painting image. 
Please provide the output path to the synthetic data to replace `path/to/synthetic_data_output_dir` 
and the root path for data filelist `path/to/filelist_root`. (this by default is the path to `CDS/data`).
