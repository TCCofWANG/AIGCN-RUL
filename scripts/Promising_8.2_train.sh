
for length in 80
do

for data_id in FD001  FD002  FD003  FD004
do

for rate in  0.01 0.02 0.05 0.001 0.005 0.0001 0.00005 0.00001 0.000005
do

python -u main.py \
  --dataset_name 'CMAPSS'\
  --DA False\
  --Classify False\
  --Data_id_CMAPSS $data_id\
  --input_length $length\
  --batch_size 100\
  --dropout 0.1\
  --model_name 'Promising_V1'\
  --info 'Promising_V1_8.2_train'\
  --train_epochs 150\
  --learning_rate $rate\
  --is_minmax True\

done

done

done

