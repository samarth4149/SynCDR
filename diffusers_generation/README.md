# Synthetic Data Generation with Diffusers

This folder contains code for synthetic data generation with the Img2Img and InstructPix2Pix models available in the [diffusers library](https://github.com/huggingface/diffusers.git).

### Environment Setup

Using anaconda/miniconda, run the following. 

```bash
conda create -n diff_gen python=3.10
conda activate diff_gen
pip install -r requirements.txt
```


## Data Generation

### Img2Img

Use the following command to generate clipart domain synthetic data from paintings in the DomainNet dataset. 
Please provide `path/to/synthetic_data_output_dir` and `path/to/filelist_root` (default = `../SynCDR/data`) appropriately.

```bash
python generate_img2img.py --root_dir path/to/synthetic_data_output_dir \
                           --dataset domainnet \
                           --source painting \
                           --target clipart \
                           --batch_size 4 \
                           --split train \
                           --filelist_root path/to/filelist_root \
                           --show_progress
```

### InstructPix2Pix

Use the following command to generate clipart domain synthetic data from paintings in the DomainNet dataset. 
Please provide `path/to/synthetic_data_output_dir` and `path/to/filelist_root` (default = `../SynCDR/data`) appropriately.

```bash
python generate_instructpix2pix.py --root_dir path/to/synthetic_data_output_dir \
                                   --dataset domainnet \
                                   --source painting \
                                   --target clipart \
                                   --batch_size 4 \
                                   --split train \
                                   --filelist_root path/to/filelist_root \
                                   --show_progress
```