import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from accelerate.utils import set_seed, ProjectConfiguration
from model import Embedding
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from data import EmbeddingDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from accelerate import Accelerator


def create_adamw_optimizer(
        model,
        lr,
        weight_decay=1e-2,
        no_decay_keywords=('bias', 'LayerNorm', 'layernorm'),
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
    parser.add_argument("--dataset", help='trainset')
    parser.add_argument('--output_dir', help='output dir')
    parser.add_argument('--save_on_epoch_end', type=int, default=0, help='if save_on_epoch_end')
    parser.add_argument('--num_max_checkpoints', type=int, default=5)
    parser.add_argument('--query_max_len', type=int, default=128)
    parser.add_argument('--passage_max_len', type=int, default=512)

    parser.add_argument('--epochs', type=int, default=2, help='epoch nums')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument("--warmup_proportion", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--log_with", type=str, default='wandb', help='wandb,tensorboard')

    parser.add_argument('--neg_nums', type=int, default=15)
    parser.add_argument('--mixed_precision', default='fp16', help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--use_mrl', action='store_true', help='if mrl loss')
    parser.add_argument('--mrl_dims', type=str, help='list of mrl dims', default='128, 256, 512, 768, 1024, 1280, 1536, 1792')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    set_seed(args.seed)

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir) + '/runs',
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

    accelerator.init_trackers('embedding', config=vars(args))

    accelerator.print(f'Batch size: {args.batch_size}')
    accelerator.print(f'Start with seed: {args.seed}')
    accelerator.print(f'Output dir: {args.output_dir}')
    accelerator.print(f'Model_name_or_path: {args.model_name_or_path}')
    accelerator.print(f'Dataset: {args.dataset}')
    accelerator.print(f'mixed_precision: {args.mixed_precision}')

    accelerator.print(f'gradient_accumulation_steps: {args.gradient_accumulation_steps}')
    accelerator.print(f'temperature: {args.temperature}')
    accelerator.print(f'log_with: {args.log_with}')
    accelerator.print(f'neg_nums: {args.neg_nums}')

    accelerator.print(f'query_max_len: {args.query_max_len}')
    accelerator.print(f'passage_max_len: {args.passage_max_len}')
    accelerator.print(f'use_mrl: {args.use_mrl}')

    if args.use_mrl:
        mrl_dims = list(map(int, args.mrl_dims.split(",")))
    else:
        mrl_dims = []

    accelerator.print(f'mrl_dims: {str(mrl_dims)}')

    model = Embedding.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        use_mrl=args.use_mrl,
        mrl_dims=mrl_dims
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_datast = EmbeddingDataset(
        train_data_path=args.dataset,
        tokenizer=tokenizer,
        neg_nums=args.neg_nums,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
    )
    num_workers = 0
    train_dataloader = DataLoader(
        train_datast,
        batch_size=args.batch_size,
        collate_fn=train_datast.collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    accelerator.print(f'train_dataloader total is : {len(train_dataloader)}')
    accelerator.print(f'train_dataloader data_type is : {train_datast.data_type}')

    optimizer = create_adamw_optimizer(
            model, lr=float(args.lr)
    )
    assert 0 <= args.warmup_proportion < 1
    total_steps = (
        len(train_dataloader) * args.epochs
    ) // accelerator.gradient_state.num_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
        num_training_steps=total_steps,
    )


    model, optimizer, lr_scheduler,train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler,train_dataloader)

    accelerator.wait_for_everyone()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=None,
        accelerator=accelerator,
        epochs=args.epochs,
        lr_scheduler=lr_scheduler,
        log_interval=10,
        save_on_epoch_end=args.save_on_epoch_end,
        tokenizer=tokenizer,
    )

    accelerator.print(f'Start training for {args.epochs} epochs')
    trainer.train()

    accelerator.print('Training finished')
    accelerator.print('Saving model')
    save_dir = args.output_dir + '/model'

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
