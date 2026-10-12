<!--
 * @Description:
 * @Author: Jianping Zhou
 * @Email: jianpingzhou0927@gmail.com
 * @Date: 2024-05-03 10:52:37
-->

# MagiNet

It's a pytorch implementation of paper "MagiNet: Mask-Aware Graph Imputation Networkfor Missing Spatio-temporal Data" accepted in TKDD.

## Requirements

```shell
pip install -r requirements.txt
```

## Datasets

1. Download the raw datasets and push them into directory 'datasets/' ([download dataset](https://drive.google.com/file/d/1qsT0gnTc0MmNaisLh9xpqVcvT2CEPnu0/view?usp=sharing))

   - PEMS-BAY
   - METR-LA
   - Seattle
   - Chengdu
   - Shenzhen

2. Preprocess the dataset:

   ```shell
   python prepare_split_data.py --dataset='METR-LA'

   python prepare_miss_data.py --dataset='METR-LA' --miss_mechanism='MCAR' --miss_ratio=0.5 --seqlen=12
   ```

## How to run

```
python main.py --config_path='configs/METR-LA.yaml' --seed=0
```
