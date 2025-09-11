# labelme 标注结果转yolo 格式  

可以将labelme 的标注结果转化为yolo 需要的格式,目前只兼容`detect`,`pose`和`segment` 三种.  

## 运行  
克隆本仓库,初始化安装uv 环境后执行`main.py` 即可:  
```bash  
git clone git@github.com:12Tall/labelme2yolo.git  
cd labelme2yolo  
uv sync  
# source .venv/bin/activate
uv run main.py --help
```
输出结果如下
```console
$ uv run main.py --help
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  Labelme2Yolo 转换工具

Options:
  --help  Show this message and exit.

Commands:
  detect  目标检测格式转换
  pose    姿态检测格式转换
  seg     语义分割格式转换
```

具体二级命令的参数说明,以`pose` 为例:  
```console
$ uv run main.py pose --help
Usage: main.py pose [OPTIONS]

  姿态检测格式转换

Options:
  -i, --input-dir DIRECTORY   输入文件目录  [required]
  -o, --output-dir DIRECTORY  输出目录
  -s, --split-ratios TEXT     train/val/test 拆分比例
  -t, --threads INTEGER       线程数
  --dim INTEGER               关键点维度：2或3
  --help                      Show this message and exit.
```

运行例子：  
```console
$ uv run main.py pose -i ../edge_pose/  -o ./output
处理输入参数……
清空现有输出目录: /home/doumiao2/labelme2yolo/output
创建数据集文件夹……
读取names 和keypoints
开始转换
并行处理52个文件（8 线程）
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 763.58it/s]
```

## 规则  
参见[笔记](https://12tall.github.io/2025/09/09/yolo-overview/#%E6%95%B0%E6%8D%AE%E9%9B%86),其中姿态检测中通过`group_id` 表示关键点与矩形的归属关系,通过在[labelme 中设置图形flag 控制关键点的可见性](https://12tall.github.io/2025/09/09/yolo-overview/#labelme).  

> 因为时间原因，代码并未很好的梳理。如有问题，请随时联系。谢谢！