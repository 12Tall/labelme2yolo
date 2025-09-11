
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import json
from pathlib import Path
import random
import shutil
from time import sleep
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, Template
import tqdm


class Converter():

    template_dir = "templates"  # 模板所在文件夹
    jinja2_env = Environment(loader=FileSystemLoader(template_dir))

    def normalize_bbox(points: List[Tuple[float, float]], img_w: int, img_h: int):
        """
        返回值：
        ((x_center, y_center), (bbox_w, bbox_h), (xmin, xmax, ymin, ymax)) = 
            Converter.normalize_bbox(shape['points'], img_width, img_height)
        """
        (x1, y1), (x2, y2) = points
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        x_center, y_center = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bbox_w, bbox_h = xmax - xmin, ymax - ymin
        return (x_center, y_center), (bbox_w, bbox_h), (xmin, xmax, ymin, ymax)

    def extract_points_in_bbox(points: List[Dict[str, Any]],
                               bbox: Tuple[float, float, float, float],
                               label_names: List[str],
                               img_width,
                               img_height,
                               shape_group: Optional[str],
                               dim: int = 3):
        """
        提取矩形框内的关键点，并格式化为 YOLO Pose 格式字符串。

        Args:
            points (list): JSON 里的所有 point shape
            group_id (int): 当前矩形的 group_id
            bbox (tuple): (xmin, xmax, ymin, ymax)
            label_names (list): 关键点名称顺序
            img_w (int): 图像宽度
            img_h (int): 图像高度
            dim (int): 维度，2 或 3

        Returns:
            str: "x1 y1 (vis) x2 y2 (vis) ..."
        """
        (xmin, xmax, ymin, ymax) = bbox
        txt_content = ""
        bbox_keypoints_dict = {}
        for point in points:
            (p_x, p_y) = point["points"][0]
            point_label = point['label']
            point_group = point.get('group_id')
            p_vis = 1 if point['flags'].get('hidden', False) else 2
            # print(x1, x2)
            if (point_group == shape_group and xmin <= p_x <= xmax and ymin <= p_y <= ymax):
                if dim == 2:
                    bbox_keypoints_dict[point_label] = [p_x, p_y]
                else:
                    bbox_keypoints_dict[point_label] = [p_x, p_y, p_vis]

        for k in label_names["point"]:
            k_str = ""
            if k in bbox_keypoints_dict:
                p = bbox_keypoints_dict[k]
                if dim == 2:
                    k_str = f"{p[0]/img_width:.8f} {p[1]/img_height:.8f} "
                else:
                    k_str = f"{p[0]/img_width:.8f} {p[1]/img_height:.8f} {p[2]} "
            else:
                if dim == 2:
                    k_str = '0 0 '
                else:
                    k_str = '0 0 0 '
            txt_content += k_str
        txt_content += '\n'
        return txt_content

    def ToDetect(json_path, output_dir, ds_type, label_names, dim):
        # print(json_path, output_dir, ds_type, label_names, dim)
        json_file_name = Path(json_path).stem
        json_data = Converter.read_json(json_path)
        # 获取输入文件目录
        output_images_dir, output_labels_dir = Converter.output_dirs(
            output_dir, ds_type)

        img_width = json_data['imageWidth']
        img_height = json_data['imageHeight']
        shapes = json_data['shapes']
        txt_content = ""
        for shape in shapes:
            shape_type = shape.get("shape_type")
            if shape_type != "rectangle":
                continue
            label_id = label_names['rectangle'].index(shape['label'])
            # 统一成 左上/右下
            ((x_center, y_center), (bbox_w, bbox_h), _) = Converter.normalize_bbox(
                shape['points'], img_width, img_height)
            txt_content += f"{label_id} {x_center:.8f} {y_center:.8f} {bbox_w:.8f} {bbox_h:.8f}\n"
        # print(txt_content)

        with open(output_labels_dir / f"{json_file_name}.txt", 'w') as f:
            f.write(txt_content)
        for i in Path(json_path).parent.glob(f"{json_file_name}.*"):
            if not i.suffix in [".json", ".txt"]:
                shutil.copy(i, output_images_dir / i.name)

    def ToPose(json_path, output_dir, ds_type, label_names, dim):
        # print(json_path, output_dir, ds_type, label_names, dim)
        json_file_name = Path(json_path).stem
        json_data = Converter.read_json(json_path)
        # 获取输入文件目录
        output_images_dir, output_labels_dir = Converter.output_dirs(
            output_dir, ds_type)

        img_width = json_data['imageWidth']
        img_height = json_data['imageHeight']
        shapes = json_data['shapes']
        rectangles = [shape for shape in shapes if shape.get(
            'shape_type') == 'rectangle']
        points = [shape for shape in shapes if shape.get(
            'shape_type') == 'point']
        txt_content = ""
        for shape in rectangles:
            # 获取标签名的索引
            shape_label_id = label_names['rectangle'].index(shape['label'])
            shape_group = shape.get("group_id")
            ((x_center, y_center), (bbox_w, bbox_h),
             bbox) = Converter.normalize_bbox(shape['points'], img_width, img_height)
            txt_content += f"{shape_label_id} {x_center:.8f} {y_center:.8f} {bbox_w:.8f} {bbox_h:.8f} "

            # 获取矩形内的点
            txt_content += Converter.extract_points_in_bbox(
                points, bbox, label_names, img_width, img_height, shape_group, dim)

        with open(output_labels_dir / f"{json_file_name}.txt", 'w') as f:
            f.write(txt_content)
        for i in Path(json_path).parent.glob(f"{json_file_name}.*"):
            if not i.suffix in [".json", ".txt"]:
                shutil.copy(i, output_images_dir / i.name)

    def ToSegment(json_path, output_dir, ds_type, label_names, dim):
        json_file_name = Path(json_path).stem
        json_data = Converter.read_json(json_path)
        # 获取输入文件目录
        output_images_dir, output_labels_dir = Converter.output_dirs(
            output_dir, ds_type)

        img_width = json_data['imageWidth']
        img_height = json_data['imageHeight']
        shapes = json_data['shapes']
        txt_content = ""
        for shape in shapes:
            shape_type = shape.get("shape_type")
            if shape_type != "polygon":
                continue
            label_id = label_names['polygon'].index(shape['label'])
            points_str = " ".join(f"{x/img_width:.8f} {y/img_height:.8f}" for x, y in shape["points"])
            txt_content += f"{label_id} {points_str} \n"
        # print(txt_content)

        with open(output_labels_dir / f"{json_file_name}.txt", 'w') as f:
            f.write(txt_content)
        for i in Path(json_path).parent.glob(f"{json_file_name}.*"):
            if not i.suffix in [".json", ".txt"]:
                shutil.copy(i, output_images_dir / i.name)


    def __init__(self,
                 input_dir: Path,
                 output_dir: Path,
                 split_ratios: List[float] = [0.7, 0.2, 0.1],
                 dim: Literal[2, 3] = 3,
                 threads: int = 4):
        self.input_dir: Path = input_dir.expanduser()
        self.input_json_files = list(self.input_dir.glob("*.json"))
        random.shuffle(self.input_json_files)  # 打乱顺序，保证后续拆分均匀
        self.split_ratios: List[float] = split_ratios
        self.output_dir: Path = output_dir.expanduser()
        self.dim: Literal[2, 3] = dim
        self.threads = threads
        self.label_names = {
            "rectangle": [],
            "polygon": [],
            "point": [],
            # 下面两种暂时用不着
            # "circle": [],
            # "linestrip": []
        }

    def output_dirs(output_dir: Path, ds_type):
        return (
            output_dir / ds_type / "images",
            output_dir / ds_type / "labels",
        )

    def read_json(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        return json_data

    def read_jsons_labels(self):
        print("读取names 和keypoints")
        for json_path in self.input_json_files:
            json_data = Converter.read_json(json_path)
            for shape in json_data["shapes"]:
                shape_type = shape.get("shape_type")
                label = shape.get("label")

                if not self.label_names.get(shape_type, None) is None and not label in self.label_names[shape_type]:
                    self.label_names[shape_type].append(label)

        return json_data

    def to_detect(self):
        print("开始转换")
        self.process_json_parallelly(Converter.ToDetect)
        tmpl = Converter.jinja2_env.get_template("detect.j2")
        yaml_content = tmpl.render(
            root_dir=self.output_dir.absolute(),
            names=self.label_names['rectangle']
        )
        with open(self.output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

    def to_pose(self):
        print("开始转换")
        self.process_json_parallelly(Converter.ToPose)
        tmpl = Converter.jinja2_env.get_template("pose.j2")
        yaml_content = tmpl.render(
            root_dir=self.output_dir.absolute(),
            names=self.label_names['rectangle'],
            key_points=self.label_names['point'],
            dim=self.dim
        )
        with open(self.output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

    def to_segment(self):
        print("开始转换")
        self.process_json_parallelly(Converter.ToSegment)
        tmpl = Converter.jinja2_env.get_template("segment.j2")
        
        yaml_content = tmpl.render(
            root_dir=self.output_dir.absolute(),
            names=self.label_names['polygon']
        )
        with open(self.output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

    def process_json_parallelly(self, ToFunc):
        n = len(list(self.input_json_files))
        print(f"并行处理{n}个文件（{self.threads} 线程）")
        split_indices = [int(n * sum(self.split_ratios[:i+1]))
                         for i in range(len(self.split_ratios)-1)]
        # (input_json_path, ds_type)
        tasks = [(self.input_json_files[idx], 'train' if idx < split_indices[0]
                  else "val" if idx < split_indices[1] else "test") for idx in range(n)]
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {executor.submit(
                ToFunc, task[0], self.output_dir, task[1], self.label_names, self.dim
            ): task for task in tasks}

            # 显示进度条
            for f in tqdm.tqdm(as_completed(futures), total=len(futures)):
                _ = f.result()
