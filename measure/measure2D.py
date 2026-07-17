"""
Halcon Metrology 模块 - 直线与圆测量

核心原理：
1. 沿几何位置生成多个 1D 测量矩形
2. 对每个矩形执行边缘检测
3. 收集边缘点并拟合几何形状
"""

import json
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from measure.measure1D import Halcon1DMeasure
from measure.constants import EPS
from measure.viz import to_bgr, draw_text_shadow


_MEASURE2D_TYPE_REGISTRY: Dict[str, type] = {}


class _BaseMeasureObject:
    """Shared drawing helpers for LineMeasureObject and CircleMeasureObject."""

    measure_rectangles: list
    edge_points: list
    result: dict | None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config params to a JSON-compatible dict. Subclasses must override."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_BaseMeasureObject":
        """Deserialize from a dict. Subclasses must override."""
        raise NotImplementedError

    def _draw_measure_rectangles(
        self, img: np.ndarray, color: Tuple[int, int, int], thickness: int
    ):
        """Draw all measurement rectangles."""
        for rect in self.measure_rectangles:
            self._draw_single_rectangle(
                img, rect['center'], rect['angle'],
                rect['length2'], rect['length1'],
                color, thickness,
            )

    def _draw_single_rectangle(
        self, img: np.ndarray, center: Tuple[float, float], angle: float,
        length1: float, length2: float, color: Tuple[int, int, int],
        thickness: int,
    ):
        """Draw a single measurement rectangle on *img* (in-place)."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        corners = np.array([
            [center[1] - length1 / 2 * cos_a - length2 / 2 * sin_a,
             center[0] - length1 / 2 * sin_a + length2 / 2 * cos_a],
            [center[1] + length1 / 2 * cos_a - length2 / 2 * sin_a,
             center[0] + length1 / 2 * sin_a + length2 / 2 * cos_a],
            [center[1] + length1 / 2 * cos_a + length2 / 2 * sin_a,
             center[0] + length1 / 2 * sin_a - length2 / 2 * cos_a],
            [center[1] - length1 / 2 * cos_a + length2 / 2 * sin_a,
             center[0] - length1 / 2 * sin_a - length2 / 2 * cos_a],
        ], dtype=np.int32)
        cv2.polylines(img, [corners], True, color, thickness)
        cv2.circle(img, (int(center[1]), int(center[0])), 2, color, -1)


class LineMeasureObject(_BaseMeasureObject):
    """
    直线测量对象
    
    在指定直线位置生成多个垂直于直线的测量矩形，
    检测边缘点后拟合直线。
    
    使用示例:
        line_obj = LineMeasureObject(
            start=(100, 50),      # 起点 (row, col)
            end=(100, 350),       # 终点 (row, col)
            measure_length1=10,   # 沿直线方向半长度
            measure_length2=20,   # 垂直直线方向半宽度
            num_measures=10       # 测量点数
        )
        result = line_obj.measure(image)
        vis_img = line_obj.visualize(image)
    """
    
    def __init__(self, 
                 start: Tuple[float, float], 
                 end: Tuple[float, float],
                 measure_length1: float, 
                 measure_length2: float,
                 num_measures: int = 10,
                 sigma: float = 1.0, 
                 threshold: float = 30.0,
                 transition: str = 'all'):
        """
        初始化直线测量对象
        
        参数:
            start: 直线起点 (row, col)
            end: 直线终点 (row, col)
            measure_length1: 测量矩形沿直线方向的半长度
            measure_length2: 测量矩形垂直直线方向的半宽度
            num_measures: 测量点数（默认10）
            sigma: 高斯平滑参数（默认1.0）
            threshold: 边缘检测阈值（默认30.0）
            transition: 边缘类型 'positive'|'negative'|'all'（默认'all'）
        """
        self.start = np.array(start, dtype=np.float64)
        self.end = np.array(end, dtype=np.float64)
        self.measure_length1 = measure_length1
        self.measure_length2 = measure_length2
        self.num_measures = num_measures
        self.sigma = sigma
        self.threshold = threshold
        self.transition = transition
        
        # 计算直线参数
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        if self.length > EPS:
            self.direction_normalized = self.direction / self.length
        else:
            self.direction_normalized = np.array([1.0, 0.0])
        
        # 测量方向（垂直于直线）
        self.measure_angle = np.arctan2(
            self.direction_normalized[0],  # row 对应 y
            self.direction_normalized[1]   # col 对应 x
        ) + np.pi / 2  # 垂直方向
        
        # 存储结果
        self.result: Optional[Dict[str, Any]] = None
        self.edge_points: List[Tuple[float, float]] = []
        self.measure_rectangles: List[Dict] = []
        self._debug_info: Dict = {}
        
    def _generate_measure_rectangles(self) -> List[Dict]:
        """生成测量矩形参数"""
        rectangles = []
        
        for i in range(self.num_measures):
            # 沿直线均匀分布的测量点
            if self.num_measures > 1:
                t = i / (self.num_measures - 1)
            else:
                t = 0.5
            
            center = self.start + t * self.direction
            
            rectangles.append({
                'center': tuple(center),
                'angle': self.measure_angle,
                'length1': self.measure_length1,
                'length2': self.measure_length2,
                'index': i
            })
        
        return rectangles
    
    def measure(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行直线测量
        
        参数:
            image: 输入灰度图像
            
        返回:
            result: 包含拟合结果的字典
                - 'params': 直线参数 (a, b, c) where ax + by + c = 0
                - 'start': 拟合直线起点 (row, col)
                - 'end': 拟合直线终点 (row, col)
                - 'angle': 直线角度（弧度）
                - 'num_points': 使用的边缘点数
        """
        self.edge_points = []
        self.measure_rectangles = self._generate_measure_rectangles()
        
        for rect in self.measure_rectangles:
            # 创建 1D 测量对象
            # 注意：Halcon1DMeasure 的 angle 参数
            # 0 表示水平向右，π/2 表示垂直向下
            measure = Halcon1DMeasure(
                row=rect['center'][0],
                col=rect['center'][1],
                angle=rect['angle'],
                length1=rect['length1'],  # 沿测量方向的半长度
                length2=rect['length2'],  # 垂直测量方向的半宽度
                interpolation='linear'
            )
            
            # 执行边缘检测
            try:
                row_edges, col_edges, amplitudes, _ = measure.measure_pos(
                    image, self.sigma, self.threshold, self.transition, 'all'
                )
                
                # 收集边缘点（只取第一个边缘，避免重复）
                if len(row_edges) > 0:
                    # 选择最接近直线中心的边缘点
                    # 计算 ROI 中心到直线起点的距离
                    edge_distances = []
                    for r, c in zip(row_edges, col_edges):
                        # 计算到直线的投影距离
                        point = np.array([r, c])
                        proj = self._project_point_to_line(point)
                        dist = np.linalg.norm(point - proj)
                        edge_distances.append(dist)
                    
                    # 选择距离直线最近的边缘点
                    min_idx = np.argmin(edge_distances)
                    self.edge_points.append((col_edges[min_idx], row_edges[min_idx]))
                    self._debug_info[rect['index']] = {
                        'all_edges': list(zip(col_edges, row_edges, amplitudes)),
                        'selected': (col_edges[min_idx], row_edges[min_idx], amplitudes[min_idx])
                    }
            except Exception as e:
                print(f"Warning: Measure rectangle {rect['index']} failed: {e}")
                continue
        
        # 拟合直线
        if len(self.edge_points) >= 2:
            self.result = self._fit_line(self.edge_points)
        else:
            self.result = None
            print(f"Warning: Only {len(self.edge_points)} edge points found, need at least 2")
        
        return self.result
    
    def _project_point_to_line(self, point: np.ndarray) -> np.ndarray:
        """将点投影到初始直线上"""
        v = point - self.start
        t = np.dot(v, self.direction_normalized)
        t = np.clip(t, 0, self.length)
        return self.start + t * self.direction_normalized
    
    def _fit_line(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        最小二乘拟合直线
        
        使用 SVD 方法求解直线方程 ax + by + c = 0
        """
        points_arr = np.array(points)
        x = points_arr[:, 0]  # col
        y = points_arr[:, 1]  # row
        
        # 使用 SVD 方法拟合直线
        A = np.column_stack([x, y, np.ones(len(x))])
        U, S, Vt = np.linalg.svd(A)
        params = Vt[-1, :]
        
        a, b, c = params
        
        # 归一化参数
        norm = np.sqrt(a**2 + b**2)
        if norm > EPS:
            a, b, c = a/norm, b/norm, c/norm
        
        # 计算方向向量
        # 直线方向垂直于法向量 (a, b)
        direction = np.array([-b, a])
        
        # 计算直线上的一点（所有点的质心投影到直线上）
        cx = np.mean(x)
        cy = np.mean(y)
        
        # 投影到直线
        d = (a * cx + b * cy + c) / (a**2 + b**2 + EPS)
        x0 = cx - a * d
        y0 = cy - b * d
        
        # 计算拟合直线在边缘点范围内的端点
        t_values = (x - x0) * direction[0] + (y - y0) * direction[1]
        t_min, t_max = np.min(t_values), np.max(t_values)
        
        # 延伸一点
        extend = (t_max - t_min) * 0.1
        t_min -= extend
        t_max += extend
        
        fit_start = (y0 + t_min * direction[1], x0 + t_min * direction[0])
        fit_end = (y0 + t_max * direction[1], x0 + t_max * direction[0])
        
        # 计算拟合误差
        distances = np.abs(a * x + b * y + c) / (np.sqrt(a**2 + b**2) + EPS)
        mean_error = np.mean(distances)
        max_error = np.max(distances)
        
        return {
            'params': (a, b, c),
            'point': (y0, x0),
            'direction': direction.tolist(),
            'start': fit_start,
            'end': fit_end,
            'angle': np.arctan2(direction[1], direction[0]),
            'num_points': len(points),
            'mean_error': mean_error,
            'max_error': max_error
        }
    
    def visualize(self, image: np.ndarray,
                  show_rectangles: bool = True,
                  show_edge_points: bool = True,
                  show_fitted_line: bool = True,
                  show_labels: bool = True,
                  rect_color: Tuple[int, int, int] = (0, 255, 255),
                  edge_color_positive: Tuple[int, int, int] = (0, 255, 0),
                  edge_color_negative: Tuple[int, int, int] = (0, 0, 255),
                  line_color: Tuple[int, int, int] = (255, 0, 255),
                  line_thickness: int = 2,
                  point_radius: int = 5,wait_time=-1) -> np.ndarray:
        """
        可视化测量结果
        
        参数:
            image: 输入图像（灰度或彩色）
            show_rectangles: 是否显示测量矩形
            show_edge_points: 是否显示边缘点
            show_fitted_line: 是否显示拟合直线
            show_labels: 是否显示标签
            rect_color: 矩形颜色 (B, G, R)
            edge_color_positive: 正边缘颜色
            edge_color_negative: 负边缘颜色
            line_color: 拟合直线颜色
            line_thickness: 线条粗细
            point_radius: 点半径
            
        返回:
            可视化后的彩色图像
        """
        # 转换为彩色图像
        vis_img = to_bgr(image)
        
        # 1. 绘制测量矩形
        if show_rectangles:
            self._draw_measure_rectangles(vis_img, rect_color, line_thickness)
        
        # 2. 绘制边缘点
        if show_edge_points:
            self._draw_edge_points(vis_img, edge_color_positive, 
                                   edge_color_negative, point_radius, show_labels)
        
        # 3. 绘制拟合直线
        if show_fitted_line and self.result is not None:
            self._draw_fitted_line(vis_img, line_color, line_thickness)
        
        # 添加标题信息
        self._draw_info(vis_img)
        if wait_time != -1:
            cv2.imshow("Measure Result", vis_img)
            cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        
        return vis_img

    def _draw_edge_points(self, img: np.ndarray,
                          color_positive: Tuple[int, int, int],
                          color_negative: Tuple[int, int, int],
                          radius: int,
                          show_labels: bool):
        """绘制边缘点"""
        for i, (x, y) in enumerate(self.edge_points):
            # 检查是否有幅度信息
            if i in self._debug_info:
                amp = self._debug_info[i]['selected'][2]
                if amp > 0:
                    color = color_positive
                else:
                    color = color_negative
            else:
                color = color_positive
            
            # 绘制边缘点
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            cv2.circle(img, (int(x), int(y)), radius + 2, (0, 0, 0), 1)
            
            # 显示序号
            if show_labels:
                cv2.putText(img, str(i), (int(x) + radius + 3, int(y) - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_fitted_line(self, img: np.ndarray,
                          color: Tuple[int, int, int],
                          thickness: int):
        """绘制拟合直线"""
        if self.result is None:
            return
        
        start = self.result['start']
        end = self.result['end']
        
        # 绘制直线（双层效果）
        cv2.line(img, 
                (int(start[1]), int(start[0])),
                (int(end[1]), int(end[0])),
                (0, 0, 0), thickness + 2)
        cv2.line(img, 
                (int(start[1]), int(start[0])),
                (int(end[1]), int(end[0])),
                color, thickness)
        
        # 在直线终点绘制比例箭头
        direction = self.result['direction']
        dr, dc = direction[0], direction[1]
        line_len = np.sqrt(dr**2 + dc**2)
        # 箭头长度 = 线段长度的 ~15%，限制在 [12, 50] px
        arrow_len = max(12.0, min(line_len * 0.15, 50.0))

        # 起点用实心圆标注
        cv2.circle(img, (int(start[1]), int(start[0])), 4, (255, 255, 0), -1)

        # 终点比例箭头
        if line_len > 1e-6:
            udr, udc = dr / line_len, dc / line_len
            tip = (int(end[1] + udc * arrow_len),
                   int(end[0] + udr * arrow_len))
            cv2.arrowedLine(img, (int(end[1]), int(end[0])), tip,
                            color, thickness, tipLength=0.3)
    

    def _draw_radius_lines(self, img: np.ndarray,
                           max_color: Tuple[int, int, int],
                           min_color: Tuple[int, int, int],
                           thickness: int):
        """绘制最长半径和最短半径线"""
        if self.result is None:
            return
        
        center = self.result['center']
        max_point = self.result.get('max_radius_point')
        min_point = self.result.get('min_radius_point')
        
        if max_point and min_point:
            # 绘制最长半径（默认红色）
            cv2.line(img,
                    (int(center[1]), int(center[0])),
                    (int(max_point[0]), int(max_point[1])),
                    max_color, thickness + 1, cv2.LINE_AA)
            
            # 绘制最短半径（默认蓝色）
            cv2.line(img,
                    (int(center[1]), int(center[0])),
                    (int(min_point[0]), int(min_point[1])),
                    min_color, thickness + 1, cv2.LINE_AA)
            
            # 在端点绘制小圆点
            cv2.circle(img, (int(max_point[0]), int(max_point[1])), 4, max_color, -1)
            cv2.circle(img, (int(min_point[0]), int(min_point[1])), 4, min_color, -1)
    
    def _draw_info(self, img: np.ndarray):
        """绘制信息文本"""
        h, w = img.shape[:2]
        
        # 标题
        title = f'Line Measure: {self.num_measures} rectangles'
        draw_text_shadow(img, title, (10, 25), color=(255, 255, 255), font_scale=0.6, thickness=1)
        
        # 结果信息
        if self.result:
            info1 = f'Edge points: {self.result["num_points"]}'
            info2 = f'Mean error: {self.result["mean_error"]:.3f} px'
            info3 = f'Angle: {np.degrees(self.result["angle"]):.2f} deg'
            
            draw_text_shadow(img, info1, (10, 50), color=(255, 255, 255), font_scale=0.5, thickness=1)
            
            draw_text_shadow(img, info2, (10, 70), color=(255, 255, 255), font_scale=0.5, thickness=1)
            
            draw_text_shadow(img, info3, (10, 90), color=(255, 255, 255), font_scale=0.5, thickness=1)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LineMeasureObject config to a JSON-compatible dict."""
        return {
            "object_type": "LineMeasureObject",
            "start": self.start.tolist(),
            "end": self.end.tolist(),
            "measure_length1": self.measure_length1,
            "measure_length2": self.measure_length2,
            "num_measures": self.num_measures,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "transition": self.transition,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineMeasureObject":
        """Reconstruct a LineMeasureObject from a dict.

        The constructor recomputes all derived fields (direction, length,
        direction_normalized, measure_angle) from start/end.

        Raises:
            ValueError: If required keys are missing.
        """
        required = ("start", "end", "measure_length1", "measure_length2")
        for key in required:
            if key not in data:
                raise ValueError(
                    f"LineMeasureObject.from_dict: missing required key '{key}'"
                )
        return cls(
            start=tuple(data["start"]),
            end=tuple(data["end"]),
            measure_length1=float(data["measure_length1"]),
            measure_length2=float(data["measure_length2"]),
            num_measures=int(data.get("num_measures", 10)),
            sigma=float(data.get("sigma", 1.0)),
            threshold=float(data.get("threshold", 30.0)),
            transition=data.get("transition", "all"),
        )


class CircleMeasureObject(_BaseMeasureObject):
    """
    圆测量对象
    
    在指定圆周位置生成多个指向圆心的测量矩形，
    检测边缘点后拟合圆。
    
    使用示例:
        circle_obj = CircleMeasureObject(
            center=(200, 200),     # 圆心 (row, col)
            radius=50,             # 预期半径
            radius_min=40,         # 最小半径
            radius_max=60,         # 最大半径
            measure_length1=10,    # 径向半长度
            measure_length2=5,     # 切向半宽度
            num_measures=16        # 测量点数
        )
        result = circle_obj.measure(image)
        vis_img = circle_obj.visualize(image)
    """
    
    def __init__(self,
                 center: Tuple[float, float],
                 radius: float,
                 measure_length1: float,
                 measure_length2: float,
                 radius_min: Optional[float] = None,
                 radius_max: Optional[float] = None,
                 num_measures: int = 12,
                 sigma: float = 1.0,
                 threshold: float = 30.0,
                 transition: str = 'all',
                 start_phi: float = 0.0,
                 end_phi: float = 2 * np.pi):
        """
        初始化圆测量对象
        
        参数:
            center: 圆心坐标 (row, col)
            radius: 预期半径
            radius_min: 最小半径（默认 None → radius - measure_length1）
            radius_max: 最大半径（默认 None → radius + measure_length1）
            measure_length1: 测量矩形径向半长度
            measure_length2: 测量矩形切向半宽度
            num_measures: 圆周测量点数（默认12）
            sigma: 高斯平滑参数（默认1.0）
            threshold: 边缘检测阈值（默认30.0）
            transition: 边缘类型 'positive'|'negative'|'all'
            start_phi: 起始角度（弧度，默认0）
            end_phi: 结束角度（弧度，默认2π）
        """
        self.center = np.array(center, dtype=np.float64)
        self.radius = radius
        self.measure_length1 = measure_length1
        self.measure_length2 = measure_length2
        self.radius_min = radius_min if radius_min is not None else radius - measure_length1
        self.radius_max = radius_max if radius_max is not None else radius + measure_length1
        self.num_measures = num_measures
        self.sigma = sigma
        self.threshold = threshold
        self.transition = transition
        self.start_phi = start_phi
        self.end_phi = end_phi
        
        # 存储结果
        self.result: Optional[Dict[str, Any]] = None
        self.edge_points: List[Tuple[float, float]] = []
        self.measure_rectangles: List[Dict] = []
        self._debug_info: Dict = {}
    
    def _generate_measure_rectangles(self) -> List[Dict]:
        """生成圆周测量矩形参数"""
        rectangles = []
        
        # 计算角度范围
        if self.end_phi > self.start_phi:
            phi_range = self.end_phi - self.start_phi
        else:
            phi_range = self.end_phi + 2 * np.pi - self.start_phi
        
        angle_step = phi_range / self.num_measures
        
        for i in range(self.num_measures):
            # 计算当前角度（从标准角度系统：0=右，π/2=下）
            phi = self.start_phi + i * angle_step
            
            # 圆周上的点位置
            # 注意：row 对应 y（向下），col 对应 x（向右）
            # 在图像坐标系中：sin 对应 row，cos 对应 col
            row = self.center[0] + self.radius * np.sin(phi)
            col = self.center[1] + self.radius * np.cos(phi)
            
            # 测量矩形角度（指向圆心）
            # 从圆周点指向圆心的方向
            measure_angle = phi + np.pi  # 反方向
            
            rectangles.append({
                'center': (row, col),
                'angle': measure_angle,
                'length1': self.measure_length1,
                'length2': self.measure_length2,
                'radial_angle': phi,
                'index': i
            })
        
        return rectangles
    
    def measure(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行圆测量
        
        参数:
            image: 输入灰度图像
            
        返回:
            result: 包含拟合结果的字典
                - 'center': 圆心 (row, col)
                - 'radius': 半径
                - 'num_points': 使用的边缘点数
                - 'mean_error': 平均拟合误差
        """
        self.edge_points = []
        self.measure_rectangles = self._generate_measure_rectangles()
        
        for rect in self.measure_rectangles:
            # 创建 1D 测量对象
            # 矩形沿径向方向（指向圆心）
            measure = Halcon1DMeasure(
                row=rect['center'][0],
                col=rect['center'][1],
                angle=rect['angle'],
                length1=self.measure_length1,  # 径向长度
                length2=self.measure_length2,  # 切向长度
                interpolation='linear'
            )
            
            # 执行边缘检测
            try:
                row_edges, col_edges, amplitudes, _ = measure.measure_pos(
                    image, self.sigma, self.threshold, self.transition, 'all'
                )
                
                # 筛选在半径范围内的边缘点
                valid_edges = []
                for r, c, amp in zip(row_edges, col_edges, amplitudes):
                    # 计算到圆心的距离
                    dist = np.sqrt((r - self.center[0])**2 + (c - self.center[1])**2)
                    if self.radius_min <= dist <= self.radius_max:
                        valid_edges.append((c, r, amp, dist))
                
                # 选择最接近预期半径的边缘点
                if valid_edges:
                    # 按距离预期半径的偏差排序
                    valid_edges.sort(key=lambda x: abs(x[3] - self.radius))
                    best_edge = valid_edges[0]
                    self.edge_points.append((best_edge[0], best_edge[1]))
                    
                    self._debug_info[rect['index']] = {
                        'all_edges': [(e[0], e[1], e[2]) for e in valid_edges],
                        'selected': (best_edge[0], best_edge[1], best_edge[2]),
                        'radial_angle': rect['radial_angle']
                    }
            except Exception as e:
                print(f"Warning: Measure rectangle {rect['index']} failed: {e}")
                continue
        
        # 拟合圆
        if len(self.edge_points) >= 3:
            self.result = self._fit_circle(self.edge_points)
        else:
            self.result = None
            print(f"Warning: Only {len(self.edge_points)} edge points found, need at least 3")
        
        return self.result
    
    def _fit_circle(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        最小二乘拟合圆
        
        使用代数方法拟合圆方程 (x-xc)² + (y-yc)² = R²
        """
        points_arr = np.array(points)
        x = points_arr[:, 0]  # col
        y = points_arr[:, 1]  # row
        n = len(points)
        
        # 方法：构建线性方程组
        # 圆方程: x² + y² - 2xc*x - 2yc*y + (xc² + yc² - R²) = 0
        # 令 A = -2xc, B = -2yc, C = xc² + yc² - R²
        # 则: x² + y² + A*x + B*y + C = 0
        
        M = np.column_stack([x, y, np.ones(n)])
        b = -(x**2 + y**2)
        
        # 最小二乘求解
        params, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
        A, B, C = params
        
        # 计算圆心和半径
        xc = -A / 2
        yc = -B / 2
        R_squared = xc**2 + yc**2 - C
        
        if R_squared > 0:
            R = np.sqrt(R_squared)
        else:
            R = 0
        
        # 计算拟合误差（每个点到圆的距离偏差）
        distances = np.sqrt((x - xc)**2 + (y - yc)**2)
        errors = np.abs(distances - R)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # 计算椭圆度（最大直径 - 最小直径）
        if n >= 3:
            max_idx = np.argmax(distances)
            min_idx = np.argmin(distances)
            max_radius = distances[max_idx]
            min_radius = distances[min_idx]
            max_radius_point = (x[max_idx], y[max_idx])  # (col, row)
            min_radius_point = (x[min_idx], y[min_idx])  # (col, row)
            ellipticity = 2 * (max_radius - min_radius)
        else:
            ellipticity = 0.0
            max_radius = min_radius = 0.0
            max_radius_point = min_radius_point = None
        
        return {
            'center': (yc, xc),  # (row, col)
            'radius': R,
            'num_points': n,
            'mean_error': mean_error,
            'max_error': max_error,
            'ellipticity': ellipticity,
            'max_radius': max_radius,
            'min_radius': min_radius,
            'max_radius_point': max_radius_point,  # (col, row)
            'min_radius_point': min_radius_point,  # (col, row)
        }
    
    def visualize(self, image: np.ndarray,
                  show_rectangles: bool = True,
                  show_edge_points: bool = True,
                  show_fitted_circle: bool = True,
                  show_labels: bool = True,
                  show_center_lines: bool = False,
                  show_search_radii: bool = True,
                  show_radius_lines: bool = True,
                  rect_color: Tuple[int, int, int] = (0, 255, 255),
                  edge_color: Tuple[int, int, int] = (0, 255, 0),
                  circle_color: Tuple[int, int, int] = (255, 0, 255),
                  center_color: Tuple[int, int, int] = (0, 0, 255),
                  radius_min_color: Tuple[int, int, int] = (255, 255, 0),
                  radius_max_color: Tuple[int, int, int] = (0, 165, 255),
                  max_radius_color: Tuple[int, int, int] = (0, 0, 255),
                  min_radius_color: Tuple[int, int, int] = (255, 0, 0),
                  line_thickness: int = 2,
                  point_radius: int = 5) -> np.ndarray:
        """
        可视化测量结果

        参数:
            image: 输入图像（灰度或彩色）
            show_rectangles: 是否显示测量矩形
            show_edge_points: 是否显示边缘点
            show_fitted_circle: 是否显示拟合圆
            show_labels: 是否显示标签
            show_center_lines: 是否显示圆心到边缘点的连线
            show_search_radii: 是否显示搜索半径范围（radius_min / radius_max）
            show_radius_lines: 是否显示最长/最短半径线
            rect_color: 矩形颜色 (B, G, R)
            edge_color: 边缘点颜色
            circle_color: 拟合圆颜色
            center_color: 圆心颜色
            radius_min_color: 最小搜索半径圆颜色 (B, G, R)
            radius_max_color: 最大搜索半径圆颜色 (B, G, R)
            max_radius_color: 最长半径线颜色 (B, G, R)
            min_radius_color: 最短半径线颜色 (B, G, R)
            line_thickness: 线条粗细
            point_radius: 点半径

        返回:
            可视化后的彩色图像
        """
        # 转换为彩色图像
        vis_img = to_bgr(image)

        # 1. 绘制测量矩形
        if show_rectangles:
            self._draw_measure_rectangles(vis_img, rect_color, line_thickness)

        # 2. 绘制边缘点
        if show_edge_points:
            self._draw_edge_points(vis_img, edge_color, point_radius,
                                   show_labels, show_center_lines, center_color)

        # 3. 绘制搜索半径范围
        if show_search_radii:
            self._draw_search_radii(vis_img, radius_min_color, radius_max_color, line_thickness)

        # 4. 绘制拟合圆
        if show_fitted_circle and self.result is not None:
            self._draw_fitted_circle(vis_img, circle_color, center_color, line_thickness)

        # 5. 绘制最长/最短半径线
        if show_radius_lines and self.result is not None:
            self._draw_radius_lines(vis_img, max_radius_color, min_radius_color, line_thickness)

        # 添加信息
        self._draw_info(vis_img)

        return vis_img

    def _draw_measure_rectangles(
        self, img: np.ndarray, color: Tuple[int, int, int], thickness: int
    ):
        """绘制所有测量矩形（Circle 使用不同的 rect dict 键名约定）"""
        for rect in self.measure_rectangles:
            self._draw_single_rectangle(
                img, rect['center'], rect['angle'],
                rect['length1'], rect['length2'],
                color, thickness,
            )

    def _draw_edge_points(self, img: np.ndarray,
                          color: Tuple[int, int, int],
                          radius: int,
                          show_labels: bool,
                          show_center_lines: bool,
                          center_color: Tuple[int, int, int]):
        """绘制边缘点"""
        for i, (x, y) in enumerate(self.edge_points):
            # 绘制边缘点
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            cv2.circle(img, (int(x), int(y)), radius + 2, (0, 0, 0), 1)
            
            # 绘制到圆心的连线
            if show_center_lines and self.result is not None:
                center = self.result['center']
                cv2.line(img, (int(center[1]), int(center[0])),
                        (int(x), int(y)), center_color, 1, cv2.LINE_AA)
            
            # 显示序号
            if show_labels:
                cv2.putText(img, str(i), (int(x) + radius + 3, int(y) - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_dashed_circle(self, img: np.ndarray,
                            center: Tuple[float, float],
                            radius: float,
                            color: Tuple[int, int, int],
                            thickness: int = 1,
                            dash_len: int = 10,
                            gap_len: int = 6):
        """绘制虚线圆（用短线段近似圆弧）"""
        cx, cy = int(center[1]), int(center[0])
        r = int(radius)
        if r <= 0:
            return

        total_angle = (dash_len + gap_len) / r
        n_segments = int(2 * np.pi / total_angle)
        if n_segments == 0:
            return

        for i in range(n_segments):
            a_start = i * total_angle
            a_end = a_start + dash_len / r

            n_pts = max(2, int(dash_len / 3))
            pts = []
            for j in range(n_pts + 1):
                a = a_start + j * (a_end - a_start) / n_pts
                x = int(cx + r * np.cos(a))
                y = int(cy + r * np.sin(a))
                pts.append([x, y])

            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, dtype=np.int32)], False,
                              color, thickness, cv2.LINE_AA)

    def _draw_search_radii(self, img: np.ndarray,
                           min_color: Tuple[int, int, int],
                           max_color: Tuple[int, int, int],
                           thickness: int):
        """绘制搜索半径范围（radius_min 和 radius_max 虚线圆）"""
        center = self.center
        r_min = getattr(self, 'radius_min', None)
        r_max = getattr(self, 'radius_max', None)

        if r_min is not None and r_min > 0:
            self._draw_dashed_circle(img, center, r_min, min_color,
                                     max(1, thickness - 1), dash_len=8, gap_len=5)
        if r_max is not None and r_max > 0 and r_max > (r_min or 0):
            self._draw_dashed_circle(img, center, r_max, max_color,
                                     max(1, thickness - 1), dash_len=8, gap_len=5)

    def _draw_fitted_circle(self, img: np.ndarray,
                            circle_color: Tuple[int, int, int],
                            center_color: Tuple[int, int, int],
                            thickness: int):
        """绘制拟合圆"""
        if self.result is None:
            return

        center = self.result['center']
        radius = self.result['radius']
        
        # 绘制圆（双层效果）
        cv2.circle(img, (int(center[1]), int(center[0])), 
                  int(radius), (0, 0, 0), thickness + 2)
        cv2.circle(img, (int(center[1]), int(center[0])), 
                  int(radius), circle_color, thickness)
        
        # 绘制圆心
        cv2.circle(img, (int(center[1]), int(center[0])), 5, center_color, -1)
        cv2.circle(img, (int(center[1]), int(center[0])), 7, (0, 0, 0), 2)
        
        # 绘制十字标记
        cross_size = 10
        cv2.line(img, 
                (int(center[1]) - cross_size, int(center[0])),
                (int(center[1]) + cross_size, int(center[0])),
                center_color, 1)
        cv2.line(img,
                (int(center[1]), int(center[0]) - cross_size),
                (int(center[1]), int(center[0]) + cross_size),
                center_color, 1)
    

    def _draw_radius_lines(self, img: np.ndarray,
                           max_color: Tuple[int, int, int],
                           min_color: Tuple[int, int, int],
                           thickness: int):
        """绘制最长半径和最短半径线"""
        if self.result is None:
            return
        
        center = self.result['center']
        max_point = self.result.get('max_radius_point')
        min_point = self.result.get('min_radius_point')
        
        if max_point and min_point:
            # 绘制最长半径（默认红色）
            cv2.line(img,
                    (int(center[1]), int(center[0])),
                    (int(max_point[0]), int(max_point[1])),
                    max_color, thickness + 1, cv2.LINE_AA)
            
            # 绘制最短半径（默认蓝色）
            cv2.line(img,
                    (int(center[1]), int(center[0])),
                    (int(min_point[0]), int(min_point[1])),
                    min_color, thickness + 1, cv2.LINE_AA)
            
            # 在端点绘制小圆点
            cv2.circle(img, (int(max_point[0]), int(max_point[1])), 4, max_color, -1)
            cv2.circle(img, (int(min_point[0]), int(min_point[1])), 4, min_color, -1)
    
    def _draw_info(self, img: np.ndarray):
        """绘制信息文本"""
        h, w = img.shape[:2]
        
        # 标题
        title = f'Circle Measure: {self.num_measures} rectangles'
        draw_text_shadow(img, title, (10, 25), color=(255, 255, 255), font_scale=0.6, thickness=1)
        
        # 结果信息
        if self.result:
            center = self.result['center']
            info1 = f'Center: ({center[1]:.1f}, {center[0]:.1f})'
            info2 = f'Radius: {self.result["radius"]:.2f} px'
            info3 = f'Edge points: {self.result["num_points"]}'
            info4 = f'Mean error: {self.result["mean_error"]:.3f} px'
            
            draw_text_shadow(img, info1, (10, 50), color=(255, 255, 255), font_scale=0.5, thickness=1)
            
            draw_text_shadow(img, info2, (10, 70), color=(255, 255, 255), font_scale=0.5, thickness=1)
            
            draw_text_shadow(img, info3, (10, 90), color=(255, 255, 255), font_scale=0.5, thickness=1)
            
            draw_text_shadow(img, info4, (10, 110), color=(255, 255, 255), font_scale=0.5, thickness=1)

            r_min = getattr(self, 'radius_min', None)
            r_max = getattr(self, 'radius_max', None)
            if r_min is not None and r_max is not None:
                info5 = f'Search radii: [{r_min:.1f}, {r_max:.1f}] px'
                draw_text_shadow(img, info5, (10, 130), color=(255, 255, 255), font_scale=0.5, thickness=1)
            
            # 椭圆度信息
            if 'ellipticity' in self.result:
                info6 = f'Ellipticity: {self.result["ellipticity"]:.2f} px'
                draw_text_shadow(img, info6, (10, 150), color=(255, 255, 255), font_scale=0.5, thickness=1)
                
                if 'max_radius' in self.result and 'min_radius' in self.result:
                    info7 = f'Radius range: [{self.result["min_radius"]:.2f}, {self.result["max_radius"]:.2f}]'
                    draw_text_shadow(img, info7, (10, 170), color=(255, 255, 255), font_scale=0.5, thickness=1)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize CircleMeasureObject config to a JSON-compatible dict."""
        return {
            "object_type": "CircleMeasureObject",
            "center": self.center.tolist(),
            "radius": self.radius,
            "radius_min": self.radius_min,
            "radius_max": self.radius_max,
            "measure_length1": self.measure_length1,
            "measure_length2": self.measure_length2,
            "num_measures": self.num_measures,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "transition": self.transition,
            "start_phi": self.start_phi,
            "end_phi": self.end_phi,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircleMeasureObject":
        """Reconstruct a CircleMeasureObject from a dict.

        Raises:
            ValueError: If required keys are missing.
        """
        required = ("center", "radius", "measure_length1", "measure_length2")
        for key in required:
            if key not in data:
                raise ValueError(
                    f"CircleMeasureObject.from_dict: missing required key '{key}'"
                )
        return cls(
            center=tuple(data["center"]),
            radius=float(data["radius"]),
            measure_length1=float(data["measure_length1"]),
            measure_length2=float(data["measure_length2"]),
            radius_min=data.get("radius_min"),   # None triggers auto-compute
            radius_max=data.get("radius_max"),
            num_measures=int(data.get("num_measures", 12)),
            sigma=float(data.get("sigma", 1.0)),
            threshold=float(data.get("threshold", 30.0)),
            transition=data.get("transition", "all"),
            start_phi=float(data.get("start_phi", 0.0)),
            end_phi=float(data.get("end_phi", 2 * np.pi)),
        )


class MetrologyModel:
    """
    Metrology 模型管理器
    
    用于管理多个测量对象并统一执行测量。
    
    使用示例:
        model = MetrologyModel()
        
        # 添加直线测量对象
        line_idx = model.add_line_measure(start=(100, 50), end=(100, 350), ...)
        
        # 添加圆测量对象
        circle_idx = model.add_circle_measure(center=(200, 200), radius=50, ...)
        
        # 执行测量
        model.measure(image)
        
        # 获取结果
        line_result = model.get_result(line_idx)
        circle_result = model.get_result(circle_idx)
        
        # 可视化
        vis_img = model.visualize(image)
    """
    
    def __init__(self):
        """初始化 Metrology 模型"""
        self.objects: List[Dict] = []
        self.results: List[Optional[Dict]] = []
        self._counter = 0
    
    def add_line_measure(self, 
                         start: Tuple[float, float],
                         end: Tuple[float, float],
                         measure_length1: float,
                         measure_length2: float,
                         num_measures: int = 10,
                         sigma: float = 1.0,
                         threshold: float = 30.0,
                         transition: str = 'all') -> int:
        """
        添加直线测量对象
        
        返回:
            index: 测量对象索引
        """
        obj = LineMeasureObject(
            start=start,
            end=end,
            measure_length1=measure_length1,
            measure_length2=measure_length2,
            num_measures=num_measures,
            sigma=sigma,
            threshold=threshold,
            transition=transition
        )
        
        idx = self._counter
        self._counter += 1
        
        self.objects.append({
            'index': idx,
            'type': 'line',
            'object': obj
        })
        self.results.append(None)
        
        return idx
    
    def add_circle_measure(self,
                           center: Tuple[float, float],
                           radius: float,
                           measure_length1: float,
                           measure_length2: float,
                           radius_min: Optional[float] = None,
                           radius_max: Optional[float] = None,
                           num_measures: int = 12,
                           sigma: float = 1.0,
                           threshold: float = 30.0,
                           transition: str = 'all',
                           start_phi: float = 0.0,
                           end_phi: float = 2 * np.pi) -> int:
        """
        添加圆测量对象

        返回:
            index: 测量对象索引
        """
        obj = CircleMeasureObject(
            center=center,
            radius=radius,
            measure_length1=measure_length1,
            measure_length2=measure_length2,
            radius_min=radius_min,
            radius_max=radius_max,
            num_measures=num_measures,
            sigma=sigma,
            threshold=threshold,
            transition=transition,
            start_phi=start_phi,
            end_phi=end_phi
        )
        
        idx = self._counter
        self._counter += 1
        
        self.objects.append({
            'index': idx,
            'type': 'circle',
            'object': obj
        })
        self.results.append(None)
        
        return idx
    
    def measure(self, image: np.ndarray) -> None:
        """
        执行所有测量对象的测量
        """
        for i, item in enumerate(self.objects):
            obj = item['object']
            result = obj.measure(image)
            self.results[i] = result
    
    def get_result(self, index: int) -> Optional[Dict]:
        """
        获取指定测量对象的结果
        """
        for i, item in enumerate(self.objects):
            if item['index'] == index:
                return self.results[i]
        return None
    
    def get_object(self, index: int) -> Optional[Any]:
        """
        获取指定测量对象
        """
        for item in self.objects:
            if item['index'] == index:
                return item['object']
        return None
    
    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        可视化所有测量结果
        
        参数:
            image: 输入图像
            **kwargs: 传递给各对象 visualize 方法的参数
            
        返回:
            可视化后的图像
        """
        vis_img = to_bgr(image)
        
        for item in self.objects:
            obj = item['object']
            vis_img = obj.visualize(vis_img, **kwargs)

        return vis_img

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire MetrologyModel to a JSON-compatible dict.

        Each contained LineMeasureObject or CircleMeasureObject is
        nested inline for a fully self-contained result.
        """
        return {
            "version": 1,
            "counter": self._counter,
            "objects": [
                {
                    "index": item["index"],
                    "type": item["type"],
                    "object": item["object"].to_dict(),
                }
                for item in self.objects
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetrologyModel":
        """Reconstruct a MetrologyModel from a dict.

        Raises:
            ValueError: If version is unsupported or object type is unknown.
        """
        version = data.get("version", 0)
        if version != 1:
            raise ValueError(
                f"Unsupported MetrologyModel version: {version}. Expected 1."
            )

        model = cls()
        model._counter = data.get("counter", 0)

        for item in data.get("objects", []):
            obj_data = item.get("object", {})
            inner_type = obj_data.get("object_type", "")

            cls_type = _MEASURE2D_TYPE_REGISTRY.get(inner_type)
            if cls_type is None:
                raise ValueError(
                    f"Unknown measure object type: '{inner_type}'. "
                    f"Known types: {list(_MEASURE2D_TYPE_REGISTRY.keys())}"
                )

            obj = cls_type.from_dict(obj_data)
            model.objects.append({
                "index": item.get("index", model._counter),
                "type": item.get("type", ""),
                "object": obj,
            })
            model.results.append(None)
            idx = item.get("index", 0)
            if idx >= model._counter:
                model._counter = idx + 1

        return model

    def save(self, filepath: str) -> None:
        """Serialize the model to a JSON file.

        Args:
            filepath: Path to the output .json file.
        """
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "MetrologyModel":
        """Deserialize a MetrologyModel from a JSON file.

        Args:
            filepath: Path to a .json file saved by MetrologyModel.save().

        Returns:
            A fully reconstructed MetrologyModel ready for measure().

        Raises:
            FileNotFoundError: If filepath doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
            ValueError: If version is unsupported or object type is unknown.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Populate the type registry (must come after class definitions)
_MEASURE2D_TYPE_REGISTRY.update({
    "LineMeasureObject": LineMeasureObject,
    "CircleMeasureObject": CircleMeasureObject,
})
