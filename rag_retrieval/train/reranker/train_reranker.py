import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


import torch
from accelerate.utils import set_seed, ProjectConfiguration
from modeling import SeqClassificationRanker
from transformers import get_cosine_schedule_with_warmup
from data import RankerDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from accelerate import Accelerator


def create_adamw_optimizer(
    model,
    lr,
    weight_decay=1e-2,
    no_decay_keywords=("bias", "LayerNorm", "layernorm"),
    special_lr_groups=None,  # 额外的学习率分组
):
    parameters = list(model.named_parameters())
    assigned_params = set()  # 用于追踪已分配的参数

    optimizer_grouped_parameters = []

    # 1. 先处理 special_lr_groups
    if special_lr_groups:
        for group in special_lr_groups:
            matched_params = [
                p
                for n, p in parameters
                if any(keyword in n for keyword in group["keywords"])
                and n not in assigned_params
            ]
            if matched_params:  # 如果有匹配的参数
                optimizer_grouped_parameters.append(
                    {
                        "params": matched_params,
                        "weight_decay": group.get("weight_decay", weight_decay),
                        "lr": group["lr"],  # 特定学习率
                    }
                )
                # 更新已分配的参数集合
                assigned_params.update(
                    [
                        n
                        for n, p in parameters
                        if any(keyword in n for keyword in group["keywords"])
                        and n not in assigned_params
                    ]
                )

    # 2. 再处理默认的 no_decay 和 decay 分组
    optimizer_grouped_parameters.append(
        {
            "params": [
                p
                for n, p in parameters
                if not any(nd in n for nd in no_decay_keywords)
                and n not in assigned_params
            ],
            "weight_decay": weight_decay,
        }
    )
    optimizer_grouped_parameters.append(
        {
            "params": [
                p
                for n, p in parameters
                if any(nd in n for nd in no_decay_keywords) and n not in assigned_params
            ],
            "weight_decay": 0.0,
        }
    )
    # 确保模型的所有参数分散在优化器中的所有参数组中，不重复不遗漏
    assert sum(
        [
            len(optimizer_grouped_parameters[i]["params"])
            for i in range(len(optimizer_grouped_parameters))
        ]
    ) == len(parameters)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", default="hfl/chinese-roberta-wwm-ext", required=True
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="choose from [SeqClassificationRanker]",
    )
    parser.add_argument("--train_dataset", help="training file", required=True)
    parser.add_argument("--val_dataset", help="validation file", default=None)
    parser.add_argument("--output_dir", help="output dir", required=True)
    parser.add_argument("--save_on_epoch_end", type=int, default=0)
    parser.add_argument("--num_max_checkpoints", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="point_ce",
        help="chose from [point_ce, point_mse]",
    )
    parser.add_argument(
        "--log_with", type=str, default="wandb", help="wandb, tensorboard"
    )
    # args.mixed_precision 会覆盖 deepspeed config 文件中的 mixed_precision 配置，除非 args.mixed_precision = None
    parser.add_argument("--mixed_precision", type=str, default=None)
    # deepspeed config 文件中的 gradient_accumulation_steps 配置会覆盖 args.gradient_accumulation_steps
    # 所以删除 deepspeed config 文件中的 gradient_accumulation_steps 配置
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_labels", type=int, default=1, help="mlp dim")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    set_seed(args.seed)

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir) + "/runs",
        automatic_checkpoint_naming=True,
        total_limit=args.num_max_checkpoints,
        logging_dir=str(args.output_dir),
    )

    accelerator = Accelerator(
        project_config=project_config,
        log_with=args.log_with,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    accelerator.init_trackers("ranker", config=vars(args))
    accelerator.print(f"Train Args from User Input: {vars(args)}")

    if args.model_type == "SeqClassificationRanker":
        model = SeqClassificationRanker.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            loss_type=args.loss_type,
            num_labels=args.num_labels,
            query_format="query: {}",
            document_format="document: {}",
            seq=" ",
            special_token="<score>"
        )
        train_dataset = RankerDataset(
            args.train_dataset,
            target_model=model,
            max_len=args.max_len
        )
        if args.val_dataset:
            val_dataset = RankerDataset(
                args.val_dataset,
                target_model=model,
                max_len=args.max_len
            )
    else:
        raise ValueError("暂未支持其他模型类型")


    num_workers = 10
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_dataloader = None
    if args.val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # special_lr_groups = None
    # 确保 special_lr_groups 的关键词定义合法
    special_lr_groups = [
        {"keywords": ["score.weight"], "lr": 1e-4},
    ]

    optimizer = create_adamw_optimizer(
        model, lr=args.lr, special_lr_groups=special_lr_groups
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

    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = (
        accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, val_dataloader
        )
    )

    accelerator.wait_for_everyone()

    trainer = Trainer(
        model=model,
        tokenizer=model.tokenizer,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        accelerator=accelerator,
        epochs=args.epochs,
        lr_scheduler=lr_scheduler,
        log_interval=args.log_interval * accelerator.gradient_state.num_steps,
        save_on_epoch_end=args.save_on_epoch_end
    )

    accelerator.print(f"Start training for {args.epochs} epochs ...")
    trainer.train()
    accelerator.print("Training finished!")

    accelerator.print("Saving model ...")
    save_dir = args.output_dir + "/model"
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir, safe_serialization=True)
    model.tokenizer.save_pretrained(save_dir)
    accelerator.print("Saving Successfully!")


if __name__ == "__main__":
    main()
