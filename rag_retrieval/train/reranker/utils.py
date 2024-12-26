import random


def map_label_to_continuous(label, min_label, max_label):
    """
    Maps a discrete label in the range [min_label, max_label] to a continuous value in [0, 1].

    Args:
        label (int): The discrete label to be mapped.
        min_label (int): The minimum value of the discrete label range.
        max_label (int): The maximum value of the discrete label range.

    Returns:
        float: A continuous value in the range [0, 1].
    """
    if label < min_label or label > max_label:
        raise ValueError("Label is out of range.")

    return (label - min_label) / (max_label - min_label)


def visualize_label_distribution(label_distribution):
    # 生成火花线图
    def _generate_sparkline(data, chars):
        max_value = max(data)
        min_value = min(data)
        range_value = max_value - min_value
        if range_value == 0:
            range_value = 1  # 避免除以零
        sparkline = ""
        for value in data:
            index = int((value - min_value) / range_value * (len(chars) - 1))
            sparkline += chars[index]
        return sparkline

    # 定义区间
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # 初始化区间计数
    bin_counts = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins) - 1)}
    # 遍历 label_distribution 并将值分配到相应的区间
    for key, count in label_distribution.items():
        value = float(key)
        for i in range(len(bins) - 1):
            if bins[i] <= value <= bins[i + 1]:
                bin_counts[f"{bins[i]}-{bins[i+1]}"] += count
                break
    # 计算总计数
    total_count = sum(bin_counts.values())
    # 生成火花线图数据
    sparkline_data = [count for count in bin_counts.values()]
    # 定义火花线图的字符
    sparkline_chars = ["▁", "▃", "▅", "▆", "█"]

    # 打印火花线图
    print(
        f"Loaded {total_count} data: "
        + _generate_sparkline(sparkline_data, sparkline_chars)
    )
    # 打印每个区间的总计数和比例
    for bin_range, count in bin_counts.items():
        proportion = count / total_count
        print(f"{bin_range}: {count} ({proportion:.2%})")


def shuffle_text(text, shuffle_ratio=0.15):
    """
    Shuffle the input text based on a given shuffle ratio.

    Args:
        text (str): Input text.
        shuffle_ratio (float): The probability of shuffling the text. Default is 0.15.

    Returns:
        str: Shuffled text if conditions are met, otherwise the original text.
    """
    if shuffle_ratio > 0 and len(text) > 100 and random.random() < shuffle_ratio:
        chunk_size = len(text) // 3 + 1
        split_text = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        random.shuffle(split_text)
        return " ".join(split_text)
    else:
        return text
    

def create_adamw_optimizer_with_special_lr_groups(
    model,
    lr: float,
    weight_decay=1e-2,
    no_decay_keywords=("bias", "LayerNorm", "layernorm"),
    special_lr_groups=None,  # 额外的学习率分组
):
    """
    Create an AdamW optimizer with special learning rate groups.
        查看 model 具体结构信息，确保 special_lr_groups 的关键词能够匹配到模型的参数。
        special_lr_groups = [
            {"keywords": ["score.weight"], "lr": 1e-4},
        ]
    """
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