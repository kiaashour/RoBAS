# Robust Bayes-Assisted Conformal Prediction

This repository contains code for reproducing experiments from the paper:
[Robust Bayes-Assisted Conformal Prediction](https://icml.cc/virtual/2026/poster/62631).


## Download the code and environment

Clone the repository:

```bash
git clone <YOUR_REPO_URL>
cd robas
```

Create the Conda environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate robas
```

If you use `mamba`, you can do:

```bash
mamba env create -f environment.yaml
mamba activate robas
```

## Data 

The datasets used for the experiments are provided in this [link](https://drive.google.com/drive/folders/1gx4Gp-wPddenNI22mEIQoaoM1b1yu_4z?usp=drive_link). Download the datasets and place them in your data folder. 

The `concrete` and `facebook` datasets are taken from [1], the airfoil dataset is taken from [2], and the UTKFaces [3] and VVolume [4] datasets were obtained by passing the datasets and their provided sets through ImageNet-pretrained ResNet34 from `torchvision` and the ViT/Ti-8 encoder from the `timm` library.


## Run one experiment with `run_experiment.sh`

List available launcher targets:

```bash
bash run_experiment.sh --list
```

General form:

```bash
bash run_experiment.sh <target> [extra_python_args...]
```

### Example A: synthetic experiment

```bash
bash run_experiment.sh synthetic \
  --data_var 1 \
  --n_cal 10 \
  --num_trials 300 \
  --methods dto dta hoff hs_eb hs_plugin # hs_eb refers to RoBAS-EB and hs_plugin to RoBAS-Plugin \
  --save_results true \
  --results_dir ./results/synthetic \
  --exp_name final
```

### Example B: UCI parallel main-method run

```bash
bash run_experiment.sh uci_parallel/main_methods \
  --dataset_name airfoil \
  --data_dir /path/to/airfoil_data \
  --results_dir /path/to/results/uci \
  --model_name rf \
  --n_cal 10 \
  --num_trials 300 \
  --methods dto dta hoff hs_eb hs_plugin # hs_eb refers to RoBAS-EB and hs_plugin to RoBAS-Plugin \
  --num_workers 6 \
  --save_results true \
  --exp_name final
```

### Example C: image parallel run

```bash
bash run_experiment.sh image_datasets_parallel/main_methods \
  --dataset_name vvolume \
  --embeddings_path /path/to/subsets_embeddings.pkl \
  --results_dir /path/to/results/image \
  --model_name rf \
  --n_cal 10 \
  --num_trials 300 \
  --methods dto dta hoff hs_eb hs_plugin # hs_eb refers to RoBAS-EB and hs_plugin to RoBAS-Plugin \
  --num_workers 6 \
  --save_results true \
  --exp_name final
```



## Run the full batch with `run_all.sh`

`run_all.sh` is a large list of commands for synthetic, UCI, and image experiments.

Run it from repo root:

```bash
bash run_all.sh
```

Recommended for long runs:

```bash
mkdir -p logs
nohup bash run_all.sh > logs/run_all.log 2>&1 &
```


## Parallelization note (paper reproducibility)

The experiments reported in the paper were run with **parallelization enabled** for computational efficiency.

In practice, this means using the `*_parallel/*` targets and setting `--num_workers` to values greater than 1 (for example `--num_workers 6`, as used in `run_all.sh` commands).

## References

[1] Romano, Y., Patterson, E., and Candes, E. Conformalized quantile regression. Advances in Neural Information Processing Systems, 32, 2019.

[2] Brooks, T., Pope, D., & Marcolini, M. (1989). Airfoil Self-Noise [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW2C.

[3] Zhang, Z., Song, Y., & Qi, H. (2017). Age progression/regression by conditional adversarial autoencoder. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

[4] Gustafsson, F. K., Danelljan, M., and Schön, T. B. How reliable is your regression model’s uncertainty under real-world distribution shifts? Transactions on Machine Learning Research, 2023.


## Citation

If you use this code for your research, please cite the paper:

```
@inproceedings{ashouritaklimi2024robas,
  title={Robust Bayes-Assisted Conformal Prediction},
  author={Ashouritaklimi*, Kianoosh and Cortinovis*, Stefano and Caron, François},
  booktitle={International Conference on Machine Learning},
  year={2026},
  publisher={PMLR}
}