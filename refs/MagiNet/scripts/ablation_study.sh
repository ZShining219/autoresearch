###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-11-10 10:14:31
### 

# dataset="METR-LA"   # PEMS-BAY, Seattle, Chengdu, Shenzhen

# log_path="./logs/${dataset}/MCAR"

# nohup python -u main.py \
#   --config_path "configs/${dataset}.yaml" \
#   --seed 0 \
#   --learnable 0 > ${log_path}/nonlearnablepos.log 2>&1 &




dataset="PEMS-BAY"   # PEMS-BAY, Seattle, Chengdu, Shenzhen

log_path="./logs/${dataset}/MCAR"

nohup python -u main.py \
  --config_path "configs/${dataset}.yaml" \
  --seed 0 \
  --learnable 0 > ${log_path}/nonlearnablepos.log 2>&1 &





# dataset="Seattle"   # PEMS-BAY, Seattle, Chengdu, Shenzhen

# log_path="./logs/${dataset}/MCAR"

# nohup python -u main.py \
#   --config_path "configs/${dataset}.yaml" \
#   --seed 0 \
#   --learnable 0 > ${log_path}/nonlearnablepos.log 2>&1 &





# dataset="Chengdu"   # PEMS-BAY, Seattle, Chengdu, Shenzhen

# log_path="./logs/${dataset}/MCAR"

# nohup python -u main.py \
#   --config_path "configs/${dataset}.yaml" \
#   --seed 0 \
#   --learnable 0 > ${log_path}/nonlearnablepos.log 2>&1 &




# dataset="Shenzhen"   # PEMS-BAY, Seattle, Chengdu, Shenzhen

# log_path="./logs/${dataset}/MCAR"

# nohup python -u main.py \
#   --config_path "configs/${dataset}.yaml" \
#   --seed 0 \
#   --learnable 0 > ${log_path}/nonlearnablepos.log 2>&1 &

