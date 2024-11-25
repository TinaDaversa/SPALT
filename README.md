# SPALT
This is the implementation of the method proposed in the paper entitled "Modeling Spatio-Temporal Locality in Multi-Step Forecasting of Geo-referenced Time series" authored by A.D'Aversa, G.Pio, M.Ceci, submitted to Machine Learning Journal.

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt
```
pip install -r requirements.txt
```
## Data Preparation
Download Datasets and move them to the folder "prepare_data".

* UKPV: https://huggingface.co/datasets/openclimatefix/uk_pv - (5min.parquet, metadata.csv)
* SDWPF: https://aistudio.baidu.com/competition/detail/152/0/introduction - (wtbdata_245days.csv, sdwpf_baidukddcup2022_turb_location.csv)

```
# UKPV
python prepare_data/generate_data.py --dataset_name ukpv --data prepare_data/5min.parquet --locations prepare_data/metadata.csv --n_targets 6 --n_hist_steps 12 --size_train 4 --size_test 1 --targetcol energy_target --id_key ss_id

# SDWPF
python prepare_data/generate_data.py --dataset_name swdpf --data prepare_data/wtbdata_245days.csv --locations prepare_data/sdwpf_baidukddcup2022_turb_location.csv --n_targets 6 --n_hist_steps 12 --size_train 30 --size_test 7 --targetcol patv_target --id_key turbid
```

## Model Training
```
python main.py --data $datapath --dataset_name $datasetname --n_targets $n_targets_steps --targetcol $target_col --id_key $id_key
```
e.g., `python main.py --data data/ukpv/T6H12/fold0/ --dataset_name ukpv --n_targets 6 --targetcol energy_target --id_key ss_id`.

### Credits
Credits to Marco Cerliani for the implementation of the basic linear model tree: https://github.com/cerlymarco/linear-tree


