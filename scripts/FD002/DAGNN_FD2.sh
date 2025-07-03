

for data_id in FD002
do

for length in 20 30 40 50 60
do

for dmodel in 32 64 128
do

for rate in  0.005 0.001 0.0005 0.0001
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
  --model_name 'BAGCN'\
  --info 'dBasisGNN expriment'\
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

