for data_id in FD003 FD004
do

for rate in  0.01 0.02 0.05 0.001 0.005 0.0001 0.00005 0.00001 0.000005
do

python -u main.py \
  --dataset_name 'CMAPSS'\
  --DA False\
  --Classify True\
  --Data_id_CMAPSS $data_id\
  --input_length 60\
  --batch_size 32\
  --d_model 64\
  --dropout 0.1\
  --model_name 'Transformer_domain'\
  --info 'Transformer with extra classify loss'\
  --train_epochs 200\
  --learning_rate $rate\
  --is_minmax True\

done

done

