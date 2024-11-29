import deepspeed
from deepspeed import DeepSpeedEngine
import json
import os
import pprint
import argparse
# from datasets import load_dataset, load_from_disk
from datasets import Dataset
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer  770M   
from transformers import T5ForConditionalGeneration, AutoTokenizer, TrainingArguments, Trainer



def run_training(args, model, train_data):
    print(f"Starting main loop")

    # 设置训练参数
    training_args = TrainingArguments(
        # report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        # 多GPU + DeepSpeed
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )






    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args):
    # Load and tokenize data， 本地数据
    # datasets = load_dataset('json', data_files={'train': '/home/amax/Unit_Test/projectByCo/JSON/split_data_train_new.json', 'validation': '/home/amax/Unit_Test/projectByCo/JSON/split_data_validate_new.json'})
    with open("/home/amax/Unit_Test/projectByCo/JSON/split_data_train_new.json", "r") as file:
        data = json.load(file) 
    for item in data:
        item['input'] += '<extra_id_0>'

    datasets = Dataset.from_list(data)
    tokenizer = AutoTokenizer.from_pretrained(args.load)

    def preprocess_function(examples):
        inputs = examples['input']
        outputs = examples['output']
        model_inputs = tokenizer(inputs, max_length=args.max_input_len, padding="max_length", truncation=True)
        labels = tokenizer(outputs, max_length=args.max_output_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs
    train_data = datasets.map(
            preprocess_function,
            batched=True,
            # remove_columns=datasets.column_names,
            num_proc=64,
            load_from_cache_file=False,
        )
    
    print(f'  ==> Loaded {len(train_data)} samples')
    print(f"{tokenizer.bos_token_id}   {tokenizer.pad_token_id}")
    return train_data, tokenizer


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    train_data, tokenizer2 = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.load, trust_remote_code=True, decoder_start_token_id=tokenizer2.bos_token_id, pad_token_id=tokenizer2.pad_token_id)
    model = T5ForConditionalGeneration.from_pretrained(args.load)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-input-len', default=128, type=int)
    parser.add_argument('--max-output-len', default=128, type=int)
    # Model
    parser.add_argument('--load', default='/data1/amax/CodeT5_FT/CodeT5_2b/model_770', type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int) #Batch Size Per GPU
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default="/data1/amax/CodeT5_FT/CodeT5_2b/code/deepspeed_config.json", type=str) # DeepSpeed 配置文件路径
    parser.add_argument('--fp16', default=True, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models", type=str)
    parser.add_argument('--log-freq', default=10, type=int) #每10个步骤记录一次
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)

# huggingface-cli download --resume-download Salesforce/codet5p-2b --local-dir ./model