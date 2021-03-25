export TRAIN_FILE=/home/mpark/dev/docstructure/data/tm_wiki/hmds/train.txt
export TEST_FILE=/home/mpark/dev/docstructure/data/tm_wiki/hmds/test.txt

python run_mlm.py \
    --model_name_or_path=bert-base-uncased \
    --output_dir=outputs/lm_hmds-wiki_bert-uncased \
    --overwrite_output_dir \
    --do_train \
    --train_file=$TRAIN_FILE \
    --do_eval \
    --validation_file=$TEST_FILE \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=5.0 \
    --evaluation_strategy='steps' \
    --logging_steps=2000 \
    --eval_steps=2000 \
    
