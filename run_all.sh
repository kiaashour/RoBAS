#!/usr/bin/env bash
set -e          # stop immediately if any command fails
set -u          # error on undefined variables

####################################################################################
# Synthetic
####################################################################################
# Var. 0.1, n cal 10
bash run_experiment.sh synthetic --data_var 0.1 --n_cal 10  --use_grid false --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --save_results true  --exp_name final  --results_dir ./results/synthetic
# Var. 0.5, n cal 10
bash run_experiment.sh synthetic --data_var 0.5 --n_cal 10  --use_grid false --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --save_results true  --exp_name final  --results_dir ./results/synthetic
# Var. 1, n cal 10
bash run_experiment.sh synthetic --data_var 1 --n_cal 10  --use_grid false --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --save_results true  --exp_name final  --results_dir ./results/synthetic
# Var. 2, n cal 10
bash run_experiment.sh synthetic --data_var 2 --n_cal 10  --use_grid false --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --save_results true  --exp_name final  --results_dir ./results/synthetic
# Var. 5, n cal 10
bash run_experiment.sh synthetic --data_var 5 --n_cal 10  --use_grid false --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --save_results true  --exp_name final  --results_dir ./results/synthetic
# Var. 1, n cal 5
bash run_experiment.sh synthetic --data_var 1 --n_cal 5  --use_grid false --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --save_results true  --exp_name final  --results_dir ./results/synthetic
# Var. 1, n cal 25
bash run_experiment.sh synthetic --data_var 1 --n_cal 25  --use_grid false --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --save_results true  --exp_name final  --results_dir ./results/synthetic
# Var. 1, n cal 50
bash run_experiment.sh synthetic --data_var 1 --n_cal 50  --use_grid false --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --save_results true  --exp_name final  --results_dir ./results/synthetic


####################################################################################
# UCI
####################################################################################
#######################
# Airfoil
#######################
# DTO, DTA, HOFF, HS_EB, HS_PLUGIN  - RF
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal 5 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal 10 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal 25 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal 50 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal standard --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci

# DTO, DTA, HOFF, HS_EB, HS_PLUGIN  - BLR PREDS
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal 5 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/airfoil --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal 10 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/airfoil --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal 25 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/airfoil --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal 50 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/airfoil --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name airfoil --n_cal standard --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/airfoil --results_dir ./results/uci  --data_dir ./data/airfoil_uci

# CQR  - RF
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name airfoil --n_cal 5 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name airfoil --n_cal 10 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name airfoil --n_cal 25 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name airfoil --n_cal 50 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name airfoil --n_cal standard --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci

# LOCAL  - RF
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name airfoil --n_cal 5 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name airfoil --n_cal 10 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name airfoil --n_cal 25 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name airfoil --n_cal 50 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name airfoil --n_cal standard --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/airfoil_uci

# CB (model results dir needs to include dataset name)
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name airfoil --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name airfoil --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name airfoil --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name airfoil --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name airfoil --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci

# CBMA
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name airfoil --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name airfoil --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name airfoil --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name airfoil --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name airfoil --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/airfoil  --save_results true  --results_dir ./results/uci  --data_dir ./data/airfoil_uci


# #######################
# Concrete
# #######################
# DTO, DTA, HOFF, HS_EB, HS_PLUGIN  - RF
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal 5 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal 10 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal 25 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal 50 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal standard --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr

# DTO, DTA, HOFF, HS_EB, HS_PLUGIN  - BLR preds
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal 5 --num_trials 300  --methods hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/concrete --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal 10 --num_trials 300  --methods hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/concrete --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal 25 --num_trials 300  --methods hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/concrete --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal 50 --num_trials 300  --methods hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/concrete --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/main_methods --num_workers 6 --dataset_name concrete --n_cal standard --num_trials 300  --methods hs_eb hs_plugin  --model_name blr_preds  --save_results true  --exp_name final_blr  --blr_model_results_dir ./results/cb_models/concrete --results_dir ./results/uci  --data_dir ./data/concrete_cqr


# # CQR  - RF
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name concrete --n_cal 5 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name concrete --n_cal 10 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name concrete --n_cal 25 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name concrete --n_cal 50 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cqr --num_workers 6 --dataset_name concrete --n_cal standard --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr

# LOCAL  - RF
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name concrete --n_cal 5 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name concrete --n_cal 10 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name concrete --n_cal 25 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name concrete --n_cal 50 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/local --num_workers 6 --dataset_name concrete --n_cal standard --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/concrete_cqr

# CB (model results dir needs to include dataset name)
# NOTE: a) model results dir needs to include dataset name  b) if the experiment has been run once with fit_posterior as true, then you can set it to false to avoid the cost of refitting the posterior
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name concrete --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name concrete --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name concrete --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name concrete --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cb --num_workers 6 --dataset_name concrete --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr

# CBMA
# NOTE: a) model results dir needs to include dataset name  b) if the experiment has been run once with fit_posterior as true, then you can set it to false to avoid the cost of refitting the posterior
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name concrete --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name concrete --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name concrete --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name concrete --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr
bash run_experiment.sh uci_parallel/cbma --num_workers 6 --dataset_name concrete --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/concrete  --save_results true  --results_dir ./results/uci  --data_dir ./data/concrete_cqr


# #######################
# Facebook
# #######################
# DTO, DTA, HOFF, HS_EB, HS_PLUGIN  - RF
bash run_experiment.sh uci_parallel/main_methods --dataset_name facebook_1 --n_cal 5 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/main_methods --dataset_name facebook_1 --n_cal 10 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci/main_methods --dataset_name facebook_1 --n_cal 25 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci/main_methods --dataset_name facebook_1 --n_cal 50 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci/main_methods --dataset_name facebook_1 --n_cal standard --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training

# CQR  - RF
bash run_experiment.sh uci_parallel/cqr --dataset_name facebook_1 --n_cal 5 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cqr --dataset_name facebook_1 --n_cal 10 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cqr --dataset_name facebook_1 --n_cal 25 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cqr --dataset_name facebook_1 --n_cal 50 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cqr --dataset_name facebook_1 --n_cal standard --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training

# LOCAL  - RF
bash run_experiment.sh uci_parallel/local --dataset_name facebook_1 --n_cal 5 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/local --dataset_name facebook_1 --n_cal 10 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/local --dataset_name facebook_1 --n_cal 25 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/local --dataset_name facebook_1 --n_cal 50 --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/local --dataset_name facebook_1 --n_cal standard --num_trials 300  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training

# CB (model results dir needs to include dataset name)
# NOTE: a) model results dir needs to include dataset name  b) if the experiment has been run once with fit_posterior as true, then you can set it to false to avoid the cost of refitting the posterior
bash run_experiment.sh uci_parallel/cb --dataset_name facebook_1 --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cb --dataset_name facebook_1 --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cb --dataset_name facebook_1 --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cb --dataset_name facebook_1 --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cb --dataset_name facebook_1 --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training

# CBMA
# NOTE: a) model results dir needs to include dataset name  b) if the experiment has been run once with fit_posterior as true, then you can set it to false to avoid the cost of refitting the posterior
bash run_experiment.sh uci_parallel/cbma --dataset_name facebook_1 --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cbma --dataset_name facebook_1 --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cbma --dataset_name facebook_1 --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cbma --dataset_name facebook_1 --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training
bash run_experiment.sh uci_parallel/cbma --dataset_name facebook_1 --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/facebook_1  --save_results true  --results_dir ./results/uci  --data_dir ./data/facebook_cqr/Dataset/Training


####################################################################################
# Image Datasets
####################################################################################
#######################
# UTKFaces
#######################
# DTO, DTA, HOFF, HS_EB, HS_PLUGIN  - RF
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name utkfaces --n_cal 5 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name utkfaces --n_cal 10 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name utkfaces --n_cal 25 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name utkfaces --n_cal 50 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name utkfaces --n_cal standard --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3

# LOCAL  - RF
bash run_experiment.sh image_datasets_parallel/local --dataset_name utkfaces --n_cal 5 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/local --dataset_name utkfaces --n_cal 10 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/local --dataset_name utkfaces --n_cal 25 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/local --dataset_name utkfaces --n_cal 50 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/local --dataset_name utkfaces --n_cal standard --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3

# CQR  - RF
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name utkfaces --n_cal 5 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name utkfaces --n_cal 10 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name utkfaces --n_cal 25 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name utkfaces --n_cal 50 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name utkfaces --n_cal standard --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3

# CB (model results dir needs to include dataset name)
# NOTE: a) model results dir needs to include dataset name  b) if the experiment has been run once with fit_posterior as true, then you can set it to false to avoid the cost of refitting the posterior
bash run_experiment.sh image_datasets_parallel/cb --dataset_name utkfaces --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cb --dataset_name utkfaces --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cb --dataset_name utkfaces --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cb --dataset_name utkfaces --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cb --dataset_name utkfaces --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3

# CBMA (model results dir needs to include dataset name)
# NOTE: a) model results dir needs to include dataset name  b) if the experiment has been run once with fit_posterior as true, then you can set it to false to avoid the cost of refitting the posterior
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name utkfaces --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name utkfaces --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name utkfaces --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name utkfaces --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name utkfaces --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/utkfaces  --save_results true  --results_dir ./results/image  --embeddings_path ./data/ukfaces/ood_age/hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3

#######################
# Volumetric Volume
#######################

# DTO, DTA, HOFF, HS_EB, HS_PLUGIN  - RF
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name vvolume --n_cal 5 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name vvolume --n_cal 10 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name vvolume --n_cal 25 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name vvolume --n_cal 50 --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/main_methods --dataset_name vvolume --n_cal standard --num_trials 300  --methods dto dta hoff hs_eb hs_plugin  --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl

# LOCAL  - RF
bash run_experiment.sh image_datasets_parallel/local --dataset_name vvolume --n_cal 5 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/local --dataset_name vvolume --n_cal 10 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/local --dataset_name vvolume --n_cal 25 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/local --dataset_name vvolume --n_cal 50 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/local --dataset_name vvolume --n_cal standard --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl

# CQR  - RF
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name vvolume --n_cal 5 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name vvolume --n_cal 10 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name vvolume --n_cal 25 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name vvolume --n_cal 50 --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cqr --dataset_name vvolume --n_cal standard --num_trials 300   --model_name rf  --save_results true  --exp_name final  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl

# CB (model results dir needs to include dataset name)
# NOTE: a) model results dir needs to include dataset name  b) if the experiment has been run once with fit_posterior as true, then you can set it to false to avoid the cost of refitting the posterior
bash run_experiment.sh image_datasets_parallel/cb --dataset_name vvolume --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cb --dataset_name vvolume --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cb --dataset_name vvolume --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cb --dataset_name vvolume --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cb --dataset_name vvolume --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cb_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl

# CBMA (model results dir needs to include dataset name)
# NOTE: a) model results dir needs to include dataset name  b) if the experiment has been run once with fit_posterior as true, then you can set it to false to avoid the cost of refitting the posterior
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name vvolume --n_cal 5 --num_trials 300  --use_grid true  --fit_posterior true  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name vvolume --n_cal 10 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name vvolume --n_cal 25 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name vvolume --n_cal 50 --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl
bash run_experiment.sh image_datasets_parallel/cbma --dataset_name vvolume --n_cal standard --num_trials 300  --use_grid true  --fit_posterior false  --only_fit_posterior false  --exp_name final  --model_results_dir ./results/cbma_models/vvolume  --save_results true  --results_dir ./results/image  --embeddings_path ./data/vvolume/subsets_embeddings.pkl