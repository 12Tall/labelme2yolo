import os
import shutil
import click
from pathlib import Path
from converter import Converter


def process_common_options(input_dir, output_dir, split_ratios, threads):
    """处理公共参数:类型转化,目录创建,参数分割等"""
    print(f"处理输入参数……")
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir or input_path /
                       "output").expanduser().resolve()

    # 处理拆分比例
    temp_ratios = [float(i) for i in (split_ratios.split(',')[:3] + [0]*3)[:3]]
    total = sum(temp_ratios)
    processed_ratios = [i/total for i in temp_ratios]

    # 清空并创建输出目录
    if output_path.exists():
        print(f"清空现有输出目录: {output_path}")
        shutil.rmtree(output_path)

    # 创建目录结构
    for dir_1 in ["train", "val", "test"]:
        for dir_2 in ["labels", "images"]:
            (output_path / dir_1 / dir_2).mkdir(parents=True, exist_ok=True)

    print(f"创建数据集文件夹……")
    return input_path, output_path, processed_ratios, int(threads)


# 定义通用选项组
common_options = [
    click.option('-i', '--input-dir', required=True,
                 type=click.Path(exists=True, file_okay=False), help='输入文件目录'),
    click.option('-o', '--output-dir',
                 type=click.Path(file_okay=False), help='输出目录'),
    click.option('-s', '--split-ratios', default='7,2,1',
                 help='train/val/test 拆分比例'),
    click.option('-t', '--threads', default=os.cpu_count(),
                 type=int, help='线程数')
]


def add_common_options(func):
    """添加通用选项的装饰器"""
    for option in reversed(common_options):
        func = option(func)
    return func


@click.group()
def cli():
    """Labelme2Yolo 转换工具"""
    pass


@cli.command()
@add_common_options
def detect(input_dir, output_dir, split_ratios, threads):
    """目标检测格式转换"""
    input_dir, output_dir, split_ratios, threads = process_common_options(
        input_dir, output_dir, split_ratios, threads)
    converter = Converter(input_dir,
                          output_dir,
                          split_ratios=split_ratios,
                          threads=threads)
    converter.read_jsons_labels()
    converter.to_detect()


@cli.command()
@add_common_options
@click.option('--dim', default=3, type=int, help='关键点维度：2或3')
def pose(input_dir, output_dir, split_ratios, threads, dim):
    """姿态检测格式转换"""
    input_dir, output_dir, split_ratios, threads = process_common_options(
        input_dir, output_dir, split_ratios, threads)
    converter = Converter(input_dir,
                          output_dir,
                          split_ratios=split_ratios,
                          threads=threads,
                          dim=dim)
    converter.read_jsons_labels()
    converter.to_pose()


@cli.command()
@add_common_options
def seg(input_dir, output_dir, split_ratios, threads):
    """语义分割格式转换"""
    input_dir, output_dir, split_ratios, threads = process_common_options(
        input_dir, output_dir, split_ratios, threads)
    converter = Converter(input_dir,
                          output_dir,
                          split_ratios=split_ratios,
                          threads=threads)
    converter.read_jsons_labels()
    converter.to_segment()


if __name__ == "__main__":
    cli()
