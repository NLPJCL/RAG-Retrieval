import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from accelerate.utils import  set_seed, ProjectConfiguration
from model  import CrossEncoder
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from data import RankerDataset, RankerMulabelDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from accelerate import Accelerator

def create_adamw_optimizer(
    model,
    lr,
    weight_decay = 1e-2,
    no_decay_keywords = ('bias', 'LayerNorm', 'layernorm'),
):
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument("--dataset", help='trainset file')
    parser.add_argument('--output_dir', help='output dir')
    parser.add_argument('--save_on_epoch_end', type=int, default=0)
    parser.add_argument('--num_max_checkpoints', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=512)

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument("--lr", type=float,default=5e-5)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument("--warmup_proportion", type=float,default=0.1)
    parser.add_argument("--loss_type", type=str,default='classfication',help='chose from [regression,classficatio]' )
    parser.add_argument("--log_with", type=str,default='wandb',help='wandb,tensorboard' )


    parser.add_argument('--mixed_precision', default='fp16', help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_labels', type=int, default=1, help='mlp dim')


    args = parser.parse_args()

    return args




def main():

    args = parse_args()

    set_seed(args.seed)

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir)+'/runs',
        automatic_checkpoint_naming=True,
        total_limit=args.num_max_checkpoints,
        logging_dir=str(args.output_dir),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.log_with,
        mixed_precision=args.mixed_precision, 
        project_config=project_config
    )
    
    accelerator.init_trackers('ranker', config=vars(args))

    accelerator.print(f'Batch size: {args.batch_size}')
    accelerator.print(f'Start with seed: {args.seed}')
    accelerator.print(f'Output dir: {args.output_dir}')
    accelerator.print(f'Model_name_or_path: {args.model_name_or_path}')
    accelerator.print(f'Dataset: {args.dataset}')
    accelerator.print(f'mixed_precision: {args.mixed_precision}')
    accelerator.print(f'loss_type: {args.loss_type}')
    accelerator.print(f'gradient_accumulation_steps: {args.gradient_accumulation_steps}')
    accelerator.print(f'lr: {args.lr}')
    accelerator.print(f'max_len: {args.max_len}')



    model = CrossEncoder.from_pretrained(
        model_name_or_path = args.model_name_or_path, 
        loss_type = args.loss_type,
        num_labels = args.num_labels)
        
    model = accelerator.prepare(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.loss_type == 'classfication':
        train_datast = RankerDataset(args.dataset, tokenizer, args.max_len)
    elif args.loss_type == 'regression':
        train_datast = RankerMulabelDataset(args.dataset, tokenizer, args.max_len)
    
    num_workers=0
    train_dataloader = DataLoader(
        train_datast, 
        batch_size=args.batch_size, 
        collate_fn=train_datast.collate_fn, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
    )
    
    optimizer = create_adamw_optimizer(model, lr=args.lr)

    if args.warmup_proportion != 0:

        total_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = int(args.warmup_proportion * total_steps)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(num_warmup_steps),
            num_training_steps=total_steps,
        )

    else:
        lr_scheduler = None

    optimizer, lr_scheduler,train_dataloader = accelerator.prepare(optimizer, lr_scheduler,train_dataloader)
    
    accelerator.wait_for_everyone()

    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        train_dataloader = train_dataloader,
        validation_dataloader = None,
        accelerator = accelerator,
        epochs = args.epochs,
        lr_scheduler = lr_scheduler,
        log_interval = 10,
        save_on_epoch_end = args.save_on_epoch_end,
        tokenizer = tokenizer,
    )

    accelerator.print(f'Start training for {args.epochs} epochs')
    trainer.train()

    accelerator.wait_for_everyone()

    accelerator.print('Training finished')
    accelerator.print('Saving model')

    save_dir=args.output_dir+ '/model'

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir)

    tokenizer.save_pretrained(save_dir)



if __name__ == "__main__":
    main()