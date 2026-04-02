import cv2
import numpy as np
from typing import Tuple, List

class Halcon1DMeasure:
    """
    OpenCV实现的1D测量工具，模拟Halcon的measure_pos和measure_pairs功能
    """
    
    def __init__(self, row: float, col: float, angle: float, 
                 length1: float, length2: float, 
                 img_width: int, img_height: int,
                 interpolation: str = 'bilinear'):
        """
        初始化测量对象（对应gen_measure_rectangle2）
        
        参数:
            row, col: ROI中心坐标
            angle: 主轴角度（弧度）
            length1: ROI长度（沿测量方向）
            length2: ROI宽度（垂直测量方向）
            img_width, img_height: 图像尺寸
            interpolation: 'nearest', 'bilinear', 'bicubic'
        """
        self.row = row
        self.col = col
        self.angle = angle
        self.length1 = length1
        self.length2 = length2
        self.img_width = img_width
        self.img_height = img_height
        self.interpolation = interpolation
        
        # 预计算投影线参数
        self._precompute_projection_lines()
    
    def _precompute_projection_lines(self):
        """预计算所有投影线的起点和终点"""
        # 主轴单位向量
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        
        # 垂直于主轴的单位向量
        cos_perp = np.cos(self.angle + np.pi/2)
        sin_perp = np.sin(self.angle + np.pi/2)
        
        # 沿主轴的采样点（整数像素距离）
        num_samples = int(self.length1)
        self.samples = np.linspace(-self.length1/2, self.length1/2, num_samples)
        
        # 每个采样点对应的投影线起点和终点
        self.line_starts = []
        self.line_ends = []
        
        half_width = self.length2 / 2
        for s in self.samples:
            # 主轴上的点
            center_x = self.col + s * cos_a
            center_y = self.row + s * sin_a
            
            # 投影线两端
            start_x = center_x - half_width * cos_perp
            start_y = center_y - half_width * sin_perp
            end_x = center_x + half_width * cos_perp
            end_y = center_y + half_width * sin_perp
            
            self.line_starts.append((start_x, start_y))
            self.line_ends.append((end_x, end_y))
    
    def _sample_line(self, img: np.ndarray, start: Tuple[float, float], 
                     end: Tuple[float, float], num_points: int = 100) -> np.ndarray:
        """
        沿一条线采样灰度值（支持亚像素插值）
        
        对应Halcon的插值计算
        """
        x0, y0 = start
        x1, y1 = end
        
        # 沿线的采样点
        t = np.linspace(0, 1, num_points)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        
        # 整数部分和小数部分
        x_floor = np.floor(x).astype(int)
        y_floor = np.floor(y).astype(int)
        dx = x - x_floor
        dy = y - y_floor
        
        # 边界检查
        valid_mask = (x_floor >= 0) & (x_floor < self.img_width - 1) & \
                     (y_floor >= 0) & (y_floor < self.img_height - 1)
        
        if not np.any(valid_mask):
            return np.zeros(num_points)
        
        # 双线性插值
        if self.interpolation == 'bilinear':
            f00 = img[y_floor[valid_mask], x_floor[valid_mask]]
            f10 = img[y_floor[valid_mask], x_floor[valid_mask] + 1]
            f01 = img[y_floor[valid_mask] + 1, x_floor[valid_mask]]
            f11 = img[y_floor[valid_mask] + 1, x_floor[valid_mask] + 1]
            
            interpolated = (1 - dx[valid_mask]) * (1 - dy[valid_mask]) * f00 + \
                           dx[valid_mask] * (1 - dy[valid_mask]) * f10 + \
                           (1 - dx[valid_mask]) * dy[valid_mask] * f01 + \
                           dx[valid_mask] * dy[valid_mask] * f11
            
            result = np.zeros(num_points)
            result[valid_mask] = interpolated
            return result
        
        elif self.interpolation == 'nearest':
            # 最近邻插值
            x_round = np.round(x).astype(int)
            y_round = np.round(y).astype(int)
            return img[y_round, x_round]
        
        else:
            raise ValueError(f"Unsupported interpolation: {self.interpolation}")
    
    def measure_pos(self, img: np.ndarray, sigma: float = 1.0, 
                    threshold: float = 30.0, transition: str = 'all',
                    select: str = 'all') -> Tuple[List[float], List[float], 
                                                  List[float], List[float]]:
        """
        执行边缘检测（对应Halcon的measure_pos）
        
        参数:
            img: 输入灰度图像
            sigma: 高斯平滑标准差
            threshold: 边缘幅度阈值
            transition: 'positive' | 'negative' | 'all'
            select: 'first' | 'last' | 'all'
        
        返回:
            row_edges, col_edges: 边缘坐标
            amplitudes: 边缘幅度
            distances: 连续边缘间距
        """
        # Step 1: 提取灰度轮廓
        profile = []
        for start, end in zip(self.line_starts, self.line_ends):
            line_values = self._sample_line(img, start, end, num_points=200)
            profile.append(np.mean(line_values))
        profile = np.array(profile)
        
        # Step 2: 高斯平滑
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        profile_smooth = cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float32),
                                          (kernel_size, 1), sigma).flatten()
        
        # Step 3: 计算一阶导数
        gradient = np.gradient(profile_smooth)
        
        # 导数幅度缩放（匹配Halcon行为）
        gradient = gradient * np.sqrt(2 * np.pi) * sigma
        
        # Step 4: 查找局部极值点
        from scipy.signal import find_peaks
        # 上升沿（暗到亮）对应导数的正峰值
        peaks_positive, _ = find_peaks(gradient, height=threshold)
        # 下降沿（亮到暗）对应导数的负峰值（取负号后找峰值）
        peaks_negative, _ = find_peaks(-gradient, height=threshold)
        peaks_negative = peaks_negative
        
        # Step 5: 根据transition筛选
        if transition == 'positive':
            peak_indices = peaks_positive
        elif transition == 'negative':
            peak_indices = peaks_negative
        else:  # 'all'
            peak_indices = np.sort(np.concatenate([peaks_positive, peaks_negative]))
        
        # Step 6: 亚像素精炼（抛物线拟合）
        refined_positions = []
        refined_amplitudes = []
        
        for idx in peak_indices:
            # 在极值点附近进行二次拟合
            window_size = 3
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(gradient), idx + window_size // 2 + 1)
            
            if end_idx - start_idx < 3:
                refined_pos = idx
            else:
                x_fit = np.arange(start_idx, end_idx)
                y_fit = gradient[start_idx:end_idx]
                
                # 二次多项式拟合: ax^2 + bx + c
                coeffs = np.polyfit(x_fit, y_fit, 2)
                a, b, c = coeffs
                
                # 极值点位置: x = -b/(2a)
                if abs(a) > 1e-10:
                    refined_pos = -b / (2 * a)
                else:
                    refined_pos = idx
            
            refined_positions.append(refined_pos)
            refined_amplitudes.append(gradient[idx])
        
        # Step 7: 转换为图像坐标
        row_edges, col_edges = [], []
        for pos in refined_positions:
            # 沿主轴的偏移
            s = self.samples[int(np.clip(pos, 0, len(self.samples) - 1))]
            
            edge_row = self.row + s * np.sin(self.angle)
            edge_col = self.col + s * np.cos(self.angle)
            
            row_edges.append(edge_row)
            col_edges.append(edge_col)
        
        # Step 8: 计算连续边缘间距
        distances = []
        if len(row_edges) > 1:
            for i in range(len(row_edges) - 1):
                dx = col_edges[i+1] - col_edges[i]
                dy = row_edges[i+1] - row_edges[i]
                distances.append(np.sqrt(dx**2 + dy**2))
        
        # Step 9: 根据select筛选
        if select == 'first' and row_edges:
            return [row_edges[0]], [col_edges[0]], [refined_amplitudes[0]], []
        elif select == 'last' and row_edges:
            return [row_edges[-1]], [col_edges[-1]], [refined_amplitudes[-1]], []
        
        return row_edges, col_edges, refined_amplitudes, distances
    
    def measure_pairs(self, img: np.ndarray, sigma: float = 1.0,
                      threshold: float = 30.0, transition: str = 'negative',
                      select: str = 'all'):
        """
        执行边缘对检测（对应Halcon的measure_pairs）
        """
        # 获取所有边缘
        row_edges, col_edges, amplitudes, _ = \
            self.measure_pos(img, sigma, threshold, 'all', 'all')
        
        if len(row_edges) < 2:
            return [], [], [], [], [], [], [], []
        
        # 根据transition进行配对
        pairs_row1, pairs_col1, pairs_amp1 = [], [], []
        pairs_row2, pairs_col2, pairs_amp2 = [], [], []
        centers_row, centers_col = [], []
        intra_distances, inter_distances = [], []
        
        i = 0
        while i < len(row_edges) - 1:
            # 检查是否形成有效的边对
            amp1 = amplitudes[i]
            amp2 = amplitudes[i + 1]
            
            if transition == 'positive':
                # 暗->亮->暗: 第一个正，第二个负
                if amp1 > 0 and amp2 < 0:
                    self._add_pair(i, i+1, row_edges, col_edges, amplitudes,
                                   pairs_row1, pairs_col1, pairs_amp1,
                                   pairs_row2, pairs_col2, pairs_amp2,
                                   centers_row, centers_col,
                                   intra_distances, inter_distances)
                    i += 2
                else:
                    i += 1
            elif transition == 'negative':
                # 亮->暗->亮: 第一个负，第二个正
                if amp1 < 0 and amp2 > 0:
                    self._add_pair(i, i+1, row_edges, col_edges, amplitudes,
                                   pairs_row1, pairs_col1, pairs_amp1,
                                   pairs_row2, pairs_col2, pairs_amp2,
                                   centers_row, centers_col,
                                   intra_distances, inter_distances)
                    i += 2
                else:
                    i += 1
            else:  # 'all'
                # 自动判断：第一个边缘的类型决定配对方式
                if amp1 > 0 and amp2 < 0:
                    self._add_pair(i, i+1, row_edges, col_edges, amplitudes,
                                   pairs_row1, pairs_col1, pairs_amp1,
                                   pairs_row2, pairs_col2, pairs_amp2,
                                   centers_row, centers_col,
                                   intra_distances, inter_distances)
                    i += 2
                elif amp1 < 0 and amp2 > 0:
                    self._add_pair(i, i+1, row_edges, col_edges, amplitudes,
                                   pairs_row1, pairs_col1, pairs_amp1,
                                   pairs_row2, pairs_col2, pairs_amp2,
                                   centers_row, centers_col,
                                   intra_distances, inter_distances)
                    i += 2
                else:
                    i += 1
        
        # 计算边对之间的间距
        for i in range(len(pairs_row1) - 1):
            center1_x = centers_col[i]
            center1_y = centers_row[i]
            center2_x = centers_col[i + 1]
            center2_y = centers_row[i + 1]
            inter_distances.append(np.sqrt((center2_x - center1_x)**2 + 
                                           (center2_y - center1_y)** 2))
        
        # 根据select筛选
        if select == 'first' and pairs_row1:
            return ([pairs_row1[0]], [pairs_col1[0]], [pairs_amp1[0]],
                    [pairs_row2[0]], [pairs_col2[0]], [pairs_amp2[0]],
                    [centers_row[0]], [centers_col[0]],
                    [intra_distances[0]], inter_distances)
        elif select == 'last' and pairs_row1:
            last_idx = len(pairs_row1) - 1
            return ([pairs_row1[last_idx]], [pairs_col1[last_idx]], [pairs_amp1[last_idx]],
                    [pairs_row2[last_idx]], [pairs_col2[last_idx]], [pairs_amp2[last_idx]],
                    [centers_row[last_idx]], [centers_col[last_idx]],
                    [intra_distances[last_idx]], inter_distances)
        
        return (pairs_row1, pairs_col1, pairs_amp1,
                pairs_row2, pairs_col2, pairs_amp2,
                centers_row, centers_col,
                intra_distances, inter_distances)
    
    def _add_pair(self, idx1: int, idx2: int,
                  row_edges: List[float], col_edges: List[float], amplitudes: List[float],
                  pairs_row1: List, pairs_col1: List, pairs_amp1: List,
                  pairs_row2: List, pairs_col2: List, pairs_amp2: List,
                  centers_row: List, centers_col: List,
                  intra_distances: List, inter_distances: List):
        """添加一个边缘对"""
        pairs_row1.append(row_edges[idx1])
        pairs_col1.append(col_edges[idx1])
        pairs_amp1.append(amplitudes[idx1])
        
        pairs_row2.append(row_edges[idx2])
        pairs_col2.append(col_edges[idx2])
        pairs_amp2.append(amplitudes[idx2])
        
        # 计算中心点
        center_row = (row_edges[idx1] + row_edges[idx2]) / 2
        center_col = (col_edges[idx1] + col_edges[idx2]) / 2
        centers_row.append(center_row)
        centers_col.append(center_col)
        
        # 计算边对内距离
        dx = col_edges[idx2] - col_edges[idx1]
        dy = row_edges[idx2] - row_edges[idx1]
        intra_distances.append(np.sqrt(dx**2 + dy**2))

    def draw_measure_region(self, img: np.ndarray, 
                            rect_color: Tuple[int, int, int] = (0, 255, 0),
                            rect_thickness: int = 2,
                            axis_color: Tuple[int, int, int] = (255, 0, 0),
                            axis_thickness: int = 2,
                            arrow_length: float = 20) -> np.ndarray:
        """
        在图像上绘制旋转矩形框和带箭头的主轴线
        
        参数:
            img: 输入图像（BGR格式）
            rect_color: 矩形边框颜色 (B, G, R)
            rect_thickness: 矩形边框粗细
            axis_color: 主轴线颜色 (B, G, R)
            axis_thickness: 主轴线粗细
            arrow_length: 箭头长度
            
        返回:
            绘制后的图像
        """
        # 创建图像副本以避免修改原图
        result = img.copy()

        # 计算旋转矩形的四个顶点
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        
        # 矩形尺寸
        width = self.length1  # 沿主轴方向
        height = self.length2  # 垂直主轴方向
        
        # 矩形中心
        cx, cy = self.col, self.row
        
        # 计算四个顶点
        dx = width / 2
        dy = height / 2
        
        # 计算四个顶点的坐标
        p1 = (int(cx + dx * cos_a - dy * sin_a), int(cy + dx * sin_a + dy * cos_a))
        p2 = (int(cx - dx * cos_a - dy * sin_a), int(cy - dx * sin_a + dy * cos_a))
        p3 = (int(cx - dx * cos_a + dy * sin_a), int(cy - dx * sin_a - dy * cos_a))
        p4 = (int(cx + dx * cos_a + dy * sin_a), int(cy + dx * sin_a - dy * cos_a))
        
        # 绘制旋转矩形
        cv2.line(result, p1, p2, rect_color, rect_thickness)
        cv2.line(result, p2, p3, rect_color, rect_thickness)
        cv2.line(result, p3, p4, rect_color, rect_thickness)
        cv2.line(result, p4, p1, rect_color, rect_thickness)
        
        # 计算主轴线的起点和终点
        axis_length = max(width, height) / 2 + arrow_length
        start_point = (int(cx - axis_length * cos_a), int(cy - axis_length * sin_a))
        end_point = (int(cx + axis_length * cos_a), int(cy + axis_length * sin_a))
        
        # 绘制主轴线
        cv2.line(result, start_point, end_point, axis_color, axis_thickness)
        
        # 绘制箭头
        arrow_angle = np.pi / 6  # 箭头角度
        arrow_len = arrow_length
        
        # 计算箭头的两个侧边
        arrow_p1 = (
            int(end_point[0] - arrow_len * np.cos(self.angle - arrow_angle)),
            int(end_point[1] - arrow_len * np.sin(self.angle - arrow_angle))
        )
        arrow_p2 = (
            int(end_point[0] - arrow_len * np.cos(self.angle + arrow_angle)),
            int(end_point[1] - arrow_len * np.sin(self.angle + arrow_angle))
        )
        
        # 绘制箭头
        cv2.line(result, end_point, arrow_p1, axis_color, axis_thickness)
        cv2.line(result, end_point, arrow_p2, axis_color, axis_thickness)
        
        return result


# 使用示例
# if __name__ == "__main__":
#     # 读取图像
#     img = cv2.imread('fuse.png', cv2.IMREAD_GRAYSCALE)
#     h, w = img.shape
    
#     # 创建测量对象
#     measure = Halcon1DMeasure(
#         row=300, col=300, angle=np.pi/2,
#         length1=80, length2=10,
#         img_width=w, img_height=h,
#         interpolation='bilinear'
#     )