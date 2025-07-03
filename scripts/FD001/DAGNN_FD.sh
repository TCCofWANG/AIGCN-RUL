

for data_id in FD001 FD002 FD003 FD004
do

for length in 50
do

for dmodel in 64
do

for rate in  0.001
do

for d_out in 0.1
do

for bdim in 10
do

python -u "D:/RUL/Paradise_RUL/main.py" \
  --dataset_name 'CMAPSS'\
  --DA False\
  --Classify False\
  --Data_id_CMAPSS $data_id\
  --input_length $length\
  --d_model $dmodel\
  --batch_size 128\
  --basis_dim $bdim\
  --dropout $d_out\
  --model_name 'FCSTGNN'\
  --info 'FCSTGNN expriment'\
  --train_epochs 200\
  --learning_rate $rate\
  --is_minmax True\
  --seed 1\

done

done

done

done

done

done

