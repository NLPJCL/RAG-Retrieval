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