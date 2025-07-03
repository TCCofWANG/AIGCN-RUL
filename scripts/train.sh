
for length in 50
do

for id in 'DS01'
do

for model in 'Transformer'
do

for rate in 0.0001
do

python -u main.py \
  --save_path None \
  --dataset_name 'N_CMAPSS'\
  --DA False\
  --Data_id_N_CMAPSS $id \
  --Classify False\
  --input_length $length\
  --batch_size 128\
  --dropout 0.2\
  --change_len False\
  --model_name $model \
  --info 'N_CMAPSS'\
  --train_epochs 150\
  --learning_rate $rate\
  --is_minmax True\

  
done

done

done

done


for length in 50
do

for id in 'DS01'
do

for model in 'Transformer'
do

for rate in 0.0001
do

python -u main.py \
  --save_path None \
  --dataset_name 'N_CMAPSS'\
  --DA False\
  --Data_id_N_CMAPSS $id \
  --Classify False\
  --input_length $length\
  --batch_size 128\
  --dropout 0.1\
  --change_len False\
  --model_name $model \
  --info 'N_CMAPSS'\
  --train_epochs 150\
  --learning_rate $rate\
  --is_minmax True\

  
done

done

done

done


for length in 50
do

for id in 'DS01'
do

for model in 'Transformer'
do

for rate in 0.0001
do

python -u main.py \
  --save_path None \
  --dataset_name 'N_CMAPSS'\
  --DA False\
  --Data_id_N_CMAPSS $id \
  --Classify False\
  --input_length $length\
  --batch_size 100\
  --dropout 0.1\
  --change_len False\
  --model_name $model \
  --info 'N_CMAPSS'\
  --train_epochs 150\
  --learning_rate $rate\
  --is_minmax True\

  
done

done

done

done
