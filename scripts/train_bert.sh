export DATA_PATH=/home/azureuser/finance/data


python run_mlm.py \
    --model_name_or_path=distilbert-base-uncased \
    --train_file $DATA_PATH/train.txt \
    --validation_file $DATA_PATH/valid.txt \
    --do_train --line_by_line \
    --do_eval \
    --output_dir /mnt/finance/test-distil
