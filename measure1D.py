import cv2
import numpy as np
from typing import Tuple, List, Optional
from scipy.ndimage import gaussian_filter1d

class Halcon1DMeasure:
    """
    OpenCV实现的1D测量工具（简化版）
    
    核心思路：
    1. 将斜矩形ROI通过仿射变换旋转为正矩形
    2. 在正矩形上水平方向统计每列的平均灰度值
    3. 进行边缘检测和亚像素定位
    4. 将结果逆变换回原图坐标
    """
    
    def __init__(self, row: float, col: float, angle: float, 
                 length1: float, length2: float,
                 interpolation: str = 'linear'):
        """
        初始化测量对象
        
        参数:
            row, col: ROI中心坐标
            angle: 主轴角度（弧度，0表示水平向右，π/2表示垂直向下）
            length1: ROI长度（沿测量方向的尺寸）
            length2: ROI宽度（垂直测量方向的尺寸）
            interpolation: 'linear' | 'cubic' | 'nearest'
        """
        self.row = row
        self.col = col
        self.angle = angle
        self.length1 = length1
        self.length2 = length2
        self.interpolation = interpolation
        
        # 计算仿射变换矩阵（将斜矩形转为正矩形）
        self.rotation_matrix = None
        self.inverse_matrix = None
        self._compute_transform_matrices()
    
    def _compute_transform_matrices(self):
        """
        计算仿射变换矩阵和逆矩阵
        
        关键：将ROI中心旋转到原点，旋转-angle角度，再平移回目标图像中心
        """
        # 旋转角度
        angle_deg = np.degrees(self.angle)
        
        # 计算正矩形在目标图像中的中心位置
        target_center_x = self.length1 / 2
        target_center_y = self.length2 / 2
        
        # 计算旋转矩阵：将原图中的点旋转到正矩形坐标系
        self.rotation_matrix = cv2.getRotationMatrix2D(
            center=(self.col, self.row),
            angle=angle_deg,
            scale=1.0
        )
        
        # 调整旋转矩阵，使ROI移动到目标图像中心
        self.rotation_matrix[0, 2] += target_center_x - self.col
        self.rotation_matrix[1, 2] += target_center_y - self.row
        
        # 计算逆矩阵（用于将结果转回原图坐标）
        # 逆变换：先反向平移，再反向旋转
        self.inverse_matrix = cv2.invertAffineTransform(self.rotation_matrix)
    
    def extract_roi(self, img: np.ndarray) -> np.ndarray:
        """
        提取ROI并转换为正矩形
        
        返回：
            aligned_img: 形状为 (length2, length1) 的正矩形图像
        """
        # 确定插值标志
        interp_flag = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC
        }.get(self.interpolation, cv2.INTER_LINEAR)
        
        # 执行仿射变换
        target_size = (int(self.length1), int(self.length2))
        aligned_img = cv2.warpAffine(
            img,
            self.rotation_matrix,
            target_size,
            flags=interp_flag,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned_img
    
    def _transform_points_to_original(self, points: np.ndarray) -> np.ndarray:
        """
        将正矩形坐标系中的点变换回原图坐标系
        
        参数:
            points: 形状为 (N, 2) 的数组，每行为 [x, y]（正矩形坐标）
        
        返回:
            transformed: 形状为 (N, 2) 的数组，每行为 [col, row]（原图坐标）
        """
        # 转换为齐次坐标
        num_points = points.shape[0]
        homogeneous = np.hstack([points, np.ones((num_points, 1))])
        
        # 应用逆变换矩阵
        transformed = (self.inverse_matrix @ homogeneous.T).T
        
        return transformed[:, :2]
    
    def measure_pos(self, img: np.ndarray, sigma: float = 1.0, 
                    threshold: float = 30.0, transition: str = 'all',
                    select: str = 'all', normalize_threshold: bool = False,
                    debug: bool = False) -> Tuple[List[float], List[float], 
                                                  List[float], List[float]]:
        """
        执行边缘检测
        
        参数:
            img: 输入图像（灰度图）
            sigma: 高斯平滑参数，控制平滑程度
            threshold: 边缘检测阈值
                - 如果 normalize_threshold=False: 范围 [0, 255]，表示灰度对比度
                - 如果 normalize_threshold=True: 范围 [0, 1]，表示对比度比例（0.1表示10%对比度）
            transition: 边缘类型
                - 'positive': 正边缘（暗->亮，幅度>0）
                - 'negative': 负边缘（亮->暗，幅度<0）
                - 'all': 所有边缘
            select: 返回选择
                - 'all': 返回所有边缘
                - 'first': 只返回第一个
                - 'last': 只返回最后一个
            normalize_threshold: 是否将阈值归一化到 [0, 1] 范围
                - False: 阈值范围 [0, 255]（默认，与Halcon一致）
                - True: 阈值范围 [0, 1]，更直观
            debug: 是否显示调试信息
        
        返回:
            row_edges: 边缘的行坐标
            col_edges: 边缘的列坐标
            amplitudes: 边缘幅度（正=暗->亮，负=亮->暗）
            distances: 连续边缘之间的距离
        """
        # Step 1: 提取并对齐ROI
        aligned = self.extract_roi(img)
        h, w = aligned.shape  # h = length2, w = length1
        
        # Step 2: 提取水平方向的灰度轮廓
        # 对每一列求平均，得到一维轮廓
        profile = np.mean(aligned, axis=0)  # shape: (length1,)
        
        # Debug 1: 显示原始灰度轮廓
        if debug:
            self._debug_profile(profile, "1. Original Profile (灰度轮廓)", 
                              color=(0, 0, 255), window_name='Debug_1_Original_Profile')
        
        # Step 3: 高斯平滑
        if sigma > 0:
            profile_smooth = gaussian_filter1d(profile.reshape(-1, 1).astype(np.float32), sigma=sigma).flatten()
        else:
            profile_smooth = profile.astype(np.float32)
        
        # Debug 2: 显示平滑后的灰度轮廓
        if debug:
            self._debug_profile(profile_smooth, "2. Smoothed Profile (平滑后灰度轮廓)", 
                              color=(0, 255, 0), window_name='Debug_2_Smoothed_Profile')
        
        # Step 4: 计算一阶导数
        gradient = np.gradient(profile_smooth)
        
        # Debug 3: 显示原始一阶导数
        if debug:
            self._debug_profile(gradient, "3. Raw Gradient (原始一阶导数)", 
                              color=(255, 0, 255), zero_line=True, 
                              window_name='Debug_3_Raw_Gradient')
        
        # # 导数幅度缩放（匹配Halcon行为）
        # gradient = gradient * np.sqrt(2 * np.pi) * sigma
        
        # 处理归一化阈值
        if normalize_threshold:
            # 阈值范围 [0, 1] 转换为 [0, 255]
            # 对于8位图像，最大对比度是255
            effective_threshold = threshold * 255.0
            if debug:
                print(f"[归一化阈值] 输入阈值={threshold:.4f} (0-1范围)")
                print(f"[归一化阈值] 有效阈值={effective_threshold:.2f} (原始范围)")
        else:
            # 阈值范围 [0, 255]（默认，与Halcon一致）
            effective_threshold = threshold
        
        # Debug 4: 显示缩放后的一阶导数
        if debug:
            self._debug_profile(gradient, "4. Scaled Gradient (缩放后一阶导数)", 
                              color=(255, 255, 0), zero_line=True, 
                              window_name='Debug_4_Scaled_Gradient')
            print(f"[阈值信息] 有效阈值={effective_threshold:.2f}")
        
        # Step 5: 查找局部极值点
        peaks_positive, _ = self._find_peaks(gradient, effective_threshold)
        peaks_negative, _ = self._find_peaks(-gradient, effective_threshold)
        
        # Step 6: 根据transition筛选
        if transition == 'positive':
            peak_indices = peaks_positive
        elif transition == 'negative':
            peak_indices = peaks_negative
        else:  # 'all'
            peak_indices = np.sort(np.concatenate([peaks_positive, peaks_negative]))
        
        # Step 7: 亚像素精炼
        refined_positions = []
        refined_amplitudes = []
        
        for idx in peak_indices:
            refined_pos, refined_amp = self._refine_subpixel(profile_smooth, gradient, idx)
            refined_positions.append(refined_pos)
            refined_amplitudes.append(refined_amp)
        
        if not refined_positions:
            return [], [], [], []
        
        # Debug 5: 在ROI小图上标记峰值点
        if debug:
            self._debug_roi_with_peaks(aligned, refined_positions, refined_amplitudes,
                                      window_name='Debug_5_ROI_with_Peaks')
            # 在梯度图上标记峰值点
            self._debug_profile_with_peaks(gradient, refined_positions, refined_amplitudes,
                                          "5. Gradient with Peaks (梯度+峰值点)", 
                                          window_name='Debug_5b_Gradient_with_Peaks')
        
        # Step 8: 转换为原图坐标
        # 在正矩形中，边缘位于 (x, y)，其中 x 是水平位置，y 是垂直中心
        points_aligned = np.array([
            [pos, self.length2 / 2] for pos in refined_positions
        ])
        
        points_original = self._transform_points_to_original(points_aligned)
        
        row_edges = points_original[:, 1].tolist()
        col_edges = points_original[:, 0].tolist()
        
        # Step 9: 计算连续边缘间距
        distances = []
        if len(row_edges) > 1:
            for i in range(len(row_edges) - 1):
                dx = col_edges[i+1] - col_edges[i]
                dy = row_edges[i+1] - row_edges[i]
                distances.append(np.sqrt(dx**2 + dy**2))
        
        # Debug 6: 在原图上标记检测到的边缘点
        if debug:
            self._debug_original_with_edges(img, row_edges, col_edges, refined_amplitudes,
                                           transition, window_name='Debug_6_Original_with_Edges')
        
        # Step 10: 根据select筛选
        if select == 'first':
            return [row_edges[0]], [col_edges[0]], [refined_amplitudes[0]], []
        elif select == 'last':
            return [row_edges[-1]], [col_edges[-1]], [refined_amplitudes[-1]], []
        
        return row_edges, col_edges, refined_amplitudes, distances
    
    def measure_pairs(self, img: np.ndarray, sigma: float = 1.0,
                      threshold: float = 30.0, transition: str = 'negative',
                      select: str = 'all',debug: bool = False):
        """
        执行边缘对检测
        
        返回:
            (row1, col1, amp1): 第一条边缘的坐标和幅度
            (row2, col2, amp2): 第二条边缘的坐标和幅度
            (center_row, center_col): 边对中心的坐标
            (intra_distance): 边对内距离
            (inter_distance): 边对之间距离
        """
        # 获取所有边缘
        row_edges, col_edges, amplitudes, _ = \
            self.measure_pos(img, sigma, threshold, transition, select,False,debug)
        
        if len(row_edges) < 2:
            return [], [], [], [], [], [], [], [], [],[]
        
        # 配对逻辑
        pairs_row1, pairs_col1, pairs_amp1 = [], [], []
        pairs_row2, pairs_col2, pairs_amp2 = [], [], []
        centers_row, centers_col = [], []
        intra_distances = []
        
        i = 0
        while i < len(row_edges) - 1:
            amp1 = amplitudes[i]
            amp2 = amplitudes[i + 1]
            
            valid_pair = False
            if transition == 'positive':
                # 暗->亮->暗
                valid_pair = (amp1 > 0 and amp2 < 0)
            elif transition == 'negative':
                # 亮->暗->亮
                valid_pair = (amp1 < 0 and amp2 > 0)
            else:  # 'all'
                valid_pair = (amp1 > 0 and amp2 < 0) or (amp1 < 0 and amp2 > 0)
            
            if valid_pair:
                # 添加边对
                pairs_row1.append(row_edges[i])
                pairs_col1.append(col_edges[i])
                pairs_amp1.append(amplitudes[i])
                
                pairs_row2.append(row_edges[i + 1])
                pairs_col2.append(col_edges[i + 1])
                pairs_amp2.append(amplitudes[i + 1])
                
                # 计算中心点
                center_row = (row_edges[i] + row_edges[i + 1]) / 2
                center_col = (col_edges[i] + col_edges[i + 1]) / 2
                centers_row.append(center_row)
                centers_col.append(center_col)
                
                # 计算边对内距离
                dx = col_edges[i + 1] - col_edges[i]
                dy = row_edges[i + 1] - row_edges[i]
                intra_distances.append(np.sqrt(dx**2 + dy**2))
                
                i += 2
            else:
                i += 1
        
        # 计算边对之间的间距
        inter_distances = []
        if len(centers_row) > 1:
            for i in range(len(centers_row) - 1):
                dx = centers_col[i + 1] - centers_col[i]
                dy = centers_row[i + 1] - centers_row[i]
                inter_distances.append(np.sqrt(dx**2 + dy**2))
        
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
    
    def _find_peaks(self, signal: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        查找信号中的峰值
        
        参数:
            signal: 一维信号
            threshold: 最小峰值高度
        
        返回:
            peaks: 峰值索引数组
            properties: 峰值属性（未使用，为兼容scipy接口）
        """
        # 简单的峰值检测：局部最大值且超过阈值
        peaks = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                if signal[i] > threshold:
                    peaks.append(i)
        
        return np.array(peaks, dtype=np.int32), None
    
    def _refine_subpixel(self, profile: np.ndarray, gradient: np.ndarray, 
                         peak_idx: int) -> Tuple[float, float]:
        """
        亚像素精炼
        
        使用二次多项式拟合梯度极值点附近的数据
        """
        # 窗口大小
        window_size = 3
        start_idx = max(0, peak_idx - window_size // 2)
        end_idx = min(len(gradient), peak_idx + window_size // 2 + 1)
        
        if end_idx - start_idx < 3:
            return float(peak_idx), float(gradient[peak_idx])
        
        # 提取窗口内的数据
        x = np.arange(start_idx, end_idx)
        y = gradient[start_idx:end_idx]
        
        # 二次多项式拟合: ax^2 + bx + c
        try:
            coeffs = np.polyfit(x, y, 2)
            a, b, c = coeffs
            
            # 极值点位置: x = -b / (2a)
            if abs(a) > 1e-10:
                refined_pos = -b / (2 * a)
            else:
                refined_pos = peak_idx
            
            # 限制在有效范围内
            refined_pos = max(start_idx, min(end_idx - 1, refined_pos))
            
            # 计算该位置的幅度
            refined_amp = a * refined_pos**2 + b * refined_pos + c
            
        except Exception:
            refined_pos = peak_idx
            refined_amp = gradient[peak_idx]
        
        return refined_pos, refined_amp
    
    def get_profile(self, img: np.ndarray) -> np.ndarray:
        """
        获取ROI的灰度轮廓（用于调试）
        
        返回:
            profile: 一维灰度轮廓数组
        """
        aligned = self.extract_roi(img)
        profile = np.mean(aligned, axis=0)
        return profile
    
    def _debug_profile(self, profile: np.ndarray, title: str, 
                      color: Tuple[int, int, int] = (0, 255, 0),
                      zero_line: bool = False, window_name: str = 'Debug_Profile',wait_time = 1000):
        """
        调试：显示一维信号（轮廓或梯度）的折线图
        
        参数:
            profile: 一维信号数组
            title: 图像标题
            color: 折线颜色 (B, G, R)
            zero_line: 是否绘制零线（用于梯度显示）
            window_name: 窗口名称
        """
        # 创建轮廓图像
        img_height = 400
        img_width = len(profile)
        vis = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # 归一化轮廓到图像高度
        profile_min = profile.min()
        profile_max = profile.max()
        
        if profile_max - profile_min > 1e-10:
            profile_norm = (profile - profile_min) / (profile_max - profile_min)
        else:
            profile_norm = np.zeros_like(profile)
        
        # 缩放到图像高度（留出上下边距）
        margin = 40
        profile_scaled = (profile_norm * (img_height - 2 * margin)).astype(np.int32)
        
        # 绘制折线
        points = []
        for i, val in enumerate(profile_scaled):
            y = img_height - margin - val
            points.append([i, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(vis, [points], False, color, 2)
        
        # 绘制零线（用于梯度）
        if zero_line:
            zero_y = img_height - margin - int((0 - profile_min) / (profile_max - profile_min + 1e-10) * (img_height - 2 * margin))
            cv2.line(vis, (0, zero_y), (img_width, zero_y), (128, 128, 128), 1, cv2.LINE_AA)
            
        # 绘制基准线
        cv2.line(vis, (0, img_height - margin), (img_width, img_height - margin), (0, 0, 0), 1)
        
        # 添加标题
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示数值范围
        range_text = f'[{profile_min:.1f}, {profile_max:.1f}]'
        cv2.putText(vis, range_text, (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(vis, range_text, (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imshow(window_name, vis)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    
    def _debug_profile_with_peaks(self, profile: np.ndarray, positions: List[float], 
                                  amplitudes: List[float], title: str,
                                  window_name: str = 'Debug_Profile_with_Peaks',wait_time=1000):
        """
        调试：在梯度图上标记峰值点
        
        参数:
            profile: 梯度信号数组
            positions: 峰值位置（可以是亚像素）
            amplitudes: 峰值幅度
            title: 图像标题
            window_name: 窗口名称
        """
        # 创建轮廓图像
        img_height = 400
        img_width = len(profile)
        vis = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # 归一化轮廓
        profile_min = profile.min()
        profile_max = profile.max()
        
        if profile_max - profile_min > 1e-10:
            profile_norm = (profile - profile_min) / (profile_max - profile_min)
        else:
            profile_norm = np.zeros_like(profile)
        
        margin = 40
        profile_scaled = (profile_norm * (img_height - 2 * margin)).astype(np.int32)
        
        # 绘制折线
        points = []
        for i, val in enumerate(profile_scaled):
            y = img_height - margin - val
            points.append([i, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(vis, [points], False, (100, 100, 100), 2)
        
        # 绘制零线
        zero_y = img_height - margin - int((0 - profile_min) / (profile_max - profile_min + 1e-10) * (img_height - 2 * margin))
        cv2.line(vis, (0, zero_y), (img_width, zero_y), (128, 128, 128), 1, cv2.LINE_AA)
        
        # 标记峰值点
        for pos, amp in zip(positions, amplitudes):
            # 计算峰值在图像中的位置
            peak_y = img_height - margin - int((amp - profile_min) / (profile_max - profile_min + 1e-10) * (img_height - 2 * margin))
            
            # 根据幅度正负选择颜色
            if amp > 0:
                color = (0, 255, 0)  # 绿色 - 正峰值
            else:
                color = (0, 0, 255)  # 红色 - 负峰值
            
            # 绘制圆点
            cv2.circle(vis, (int(pos), peak_y), 6, color, -1)
            cv2.circle(vis, (int(pos), peak_y), 8, (0, 0, 0), 2)
            
            # 绘制垂直线到零线
            cv2.line(vis, (int(pos), peak_y), (int(pos), zero_y), color, 1, cv2.LINE_AA)
            
            # 显示幅度值
            amp_text = f'{amp:.1f}'
            cv2.putText(vis, amp_text, (int(pos) + 8, peak_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
            cv2.putText(vis, amp_text, (int(pos) + 8, peak_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 绘制基准线
        cv2.line(vis, (0, img_height - margin), (img_width, img_height - margin), (0, 0, 0), 1)
        
        # 添加标题
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 统计信息
        stats_text = f'Peaks: {len(positions)} (+{sum(1 for a in amplitudes if a > 0)}/-{sum(1 for a in amplitudes if a < 0)})'
        cv2.putText(vis, stats_text, (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(vis, stats_text, (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imshow(window_name, vis)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    
    def _debug_roi_with_peaks(self, roi: np.ndarray, positions: List[float], 
                             amplitudes: List[float], window_name: str = 'Debug_ROI_with_Peaks',wait_time=1000):
        """
        调试：在ROI小图上标记峰值点（沿垂直中心线）
        
        参数:
            roi: ROI图像（正矩形）
            positions: 峰值位置（沿水平方向）
            amplitudes: 峰值幅度
            window_name: 窗口名称
        """
        # 转换为彩色图像
        if len(roi.shape) == 2:
            vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        else:
            vis = roi.copy()
        
        h, w = roi.shape[:2]
        center_y = h // 2
        
        # 绘制中心线
        cv2.line(vis, (0, center_y), (w, center_y), (128, 128, 128), 1, cv2.LINE_AA)
        
        # 标记峰值点
        for pos, amp in zip(positions, amplitudes):
            # 根据幅度正负选择颜色
            if amp > 0:
                color = (0, 255, 0)  # 绿色 - 正峰值（亮->暗）
            else:
                color = (0, 0, 255)  # 红色 - 负峰值（暗->亮）
            
            # 绘制垂直线
            cv2.line(vis, (int(pos), 0), (int(pos), h), color, 1, cv2.LINE_AA)
            
            # 在中心位置绘制圆点
            cv2.circle(vis, (int(pos), center_y), 5, color, -1)
            cv2.circle(vis, (int(pos), center_y), 7, (0, 0, 0), 2)
            
            # 显示幅度值
            amp_text = f'{amp:.1f}'
            text_y = center_y - 15 if amp > 0 else center_y + 25
            cv2.putText(vis, amp_text, (int(pos) - 15, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(vis, amp_text, (int(pos) - 15, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加标题
        cv2.putText(vis, f'ROI with {len(positions)} Peaks', (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis, f'ROI with {len(positions)} Peaks', (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 绘制边框
        cv2.rectangle(vis, (0, 0), (w-1, h-1), (0, 0, 0), 2)
        
        cv2.imshow(window_name, vis)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    
    def _debug_original_with_edges(self, img: np.ndarray, row_edges: List[float], 
                                   col_edges: List[float], amplitudes: List[float],
                                   transition: str, window_name: str = 'Debug_Original_with_Edges',wait_time=1000):
        """
        调试：在原图上标记检测到的边缘点
        
        参数:
            img: 原始图像
            row_edges: 边缘的行坐标
            col_edges: 边缘的列坐标
            amplitudes: 边缘幅度
            transition: 过渡类型
            window_name: 窗口名称
        """
        # 转换为彩色图像
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
        
        # 先绘制ROI矩形框
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        dx_half = self.length1 / 2
        dy_half = self.length2 / 2
        
        corners = np.array([
            [self.col - dx_half * cos_a + dy_half * sin_a,
             self.row - dx_half * sin_a - dy_half * cos_a],
            [self.col + dx_half * cos_a + dy_half * sin_a,
             self.row + dx_half * sin_a - dy_half * cos_a],
            [self.col + dx_half * cos_a - dy_half * sin_a,
             self.row + dx_half * sin_a + dy_half * cos_a],
            [self.col - dx_half * cos_a - dy_half * sin_a,
             self.row - dx_half * sin_a + dy_half * cos_a]
        ], dtype=np.int32)
        
        # 绘制矩形框（使用更明显的颜色和线条）
        # 外层黑色边框，增强对比度
        cv2.polylines(vis, [corners], True, (0, 0, 0), 5)
        # 内层亮青色边框
        cv2.polylines(vis, [corners], True, (0, 255, 255), 3)
        
        # 绘制中心点
        cv2.circle(vis, (int(self.col), int(self.row)), 5, (255, 0, 255), -1)
        
        # 绘制边缘点
        for r, c, amp in zip(row_edges, col_edges, amplitudes):
            # 根据幅度正负选择颜色
            if amp > 0:
                color = (0, 255, 255)  # 黄色 - 正峰值（暗->亮）
            else:
                color = (255, 0, 255)  # 洋红 - 负峰值（亮->暗）
            
            # 绘制大圆点
            cv2.circle(vis, (int(c), int(r)), 8, color, -1)
            cv2.circle(vis, (int(c), int(r)), 10, (0, 0, 0), 2)
            
            # 显示坐标和幅度
            label = f'({c:.1f},{r:.1f}) {amp:.1f}'
            cv2.putText(vis, label, (int(c) + 15, int(r)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(vis, label, (int(c) + 15, int(r)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加标题
        title = f'Edges: {len(row_edges)} ({transition})'
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 添加ROI信息
        roi_info = f'ROI: ({self.col:.0f},{self.row:.0f}) {np.degrees(self.angle):.1f}°'
        cv2.putText(vis, roi_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis, roi_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, vis)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    
    def draw_roi_on_image(self, img: np.ndarray, 
                         rect_color: Tuple[int, int, int] = (0, 255, 0),
                         arrow_color: Tuple[int, int, int] = (255, 0, 0),
                         line_thickness: int = 2,
                         arrow_length_ratio: float = 0.8,wait_time=-1) -> np.ndarray:
        """
        在图像上绘制ROI矩形框和主方向箭头
        
        参数:
            img: 输入图像（灰度或彩色）
            rect_color: 矩形框颜色 (B, G, R)，默认绿色
            arrow_color: 箭头颜色 (B, G, R)，默认红色
            line_thickness: 线条粗细，默认2
            arrow_length_ratio: 箭头长度占ROI长度的比例，默认0.8
        
        返回:
            vis_img: 绘制后的彩色图像
        """
        # 转换为彩色图像（如果是灰度图）
        if len(img.shape) == 2:
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = img.copy()
        
        # 计算ROI的四个角点（按顺时针顺序：左上、右上、右下、左下）
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        
        # 四个角点的偏移量
        dx_half = self.length1 / 2
        dy_half = self.length2 / 2
        
        # 左上、右上、右下、左下（顺时针顺序）
        corners = np.array([
            [
                self.col - dx_half * cos_a + dy_half * sin_a,
                self.row - dx_half * sin_a - dy_half * cos_a
            ],
            [
                self.col + dx_half * cos_a + dy_half * sin_a,
                self.row + dx_half * sin_a - dy_half * cos_a
            ],
            [
                self.col + dx_half * cos_a - dy_half * sin_a,
                self.row + dx_half * sin_a + dy_half * cos_a
            ],
            [
                self.col - dx_half * cos_a - dy_half * sin_a,
                self.row - dx_half * sin_a + dy_half * cos_a
            ]
        ], dtype=np.int32)
        
        # 绘制矩形框
        cv2.polylines(vis_img, [corners], True, rect_color, line_thickness)
        
        # 计算主方向箭头的起点和终点
        # 起点：矩形中心
        start_point = (int(self.col), int(self.row))
        
        # 终点：沿主轴方向延伸
        arrow_length = self.length1 * arrow_length_ratio / 2
        end_x = self.col + arrow_length * np.cos(self.angle)
        end_y = self.row + arrow_length * np.sin(self.angle)
        end_point = (int(end_x), int(end_y))
        
        # 绘制箭头
        cv2.arrowedLine(vis_img, start_point, end_point, arrow_color, 
                        line_thickness, tipLength=0.3)
        
        # 绘制中心点标记
        cv2.circle(vis_img, start_point, 4, (0, 0, 255), -1)
        
        # 可选：添加角度标签
        angle_deg = np.degrees(self.angle)
        angle_text = f'{angle_deg:.1f}°'
        text_pos = (int(self.col) + 10, int(self.row) - 10)
        cv2.putText(vis_img, angle_text, text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis_img, angle_text, text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        if wait_time != -1 and wait_time > 0:
            cv2.imshow("draw_roi_on_image",vis_img)
            cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        return vis_img

    def display_results(self, img: np.ndarray, 
                   row_edges: List[float], col_edges: List[float], amplitudes: List[float],
                   row1: List[float] = None, col1: List[float] = None, amp1: List[float] = None,
                   row2: List[float] = None, col2: List[float] = None, amp2: List[float] = None,
                   centers_row: List[float] = None, centers_col: List[float] = None,
                   intra_dist: List[float] = None, inter_dist: List[float] = None,
                   window_name: str = 'Measurement Results', wait_time: int = 0):
        """
        在原图上显示测量结果
        
        参数:
            img: 原始图像
            row_edges, col_edges, amplitudes: 单个边缘检测结果（来自measure_pos）
            row1, col1, amp1: 第一条边缘的坐标和幅度（来自measure_pairs）
            row2, col2, amp2: 第二条边缘的坐标和幅度（来自measure_pairs）
            centers_row, centers_col: 边对中心的坐标（来自measure_pairs）
            intra_dist: 边对内距离（来自measure_pairs）
            inter_dist: 边对之间距离（来自measure_pairs）
            window_name: 窗口名称
            wait_time: 等待时间（毫秒），0表示无限等待，-1表示不显示窗口
        """
        # 转换为彩色图像
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
        
        # 绘制ROI矩形框
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        dx_half = self.length1 / 2
        dy_half = self.length2 / 2
        
        # 计算矩形框的四个角点
        corners = np.array([
            [self.col - dx_half * cos_a + dy_half * sin_a,
            self.row - dx_half * sin_a - dy_half * cos_a],
            [self.col + dx_half * cos_a + dy_half * sin_a,
            self.row + dx_half * sin_a - dy_half * cos_a],
            [self.col + dx_half * cos_a - dy_half * sin_a,
            self.row + dx_half * sin_a + dy_half * cos_a],
            [self.col - dx_half * cos_a - dy_half * sin_a,
            self.row - dx_half * sin_a + dy_half * cos_a]
        ], dtype=np.int32)
        
        # 绘制矩形框（绿色）
        cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
        
        # 如果有边缘对结果，优先显示边缘对
        if row1 is not None and len(row1) > 0:
            # 绘制边缘对
            for i in range(len(row1)):
                # 第一条边（黄色）
                cv2.circle(vis, (int(col1[i]), int(row1[i])), 6, (0, 255, 255), -1)
                cv2.circle(vis, (int(col1[i]), int(row1[i])), 8, (0, 0, 0), 2)
                
                # 显示梯度弧度值
                amp_text = f'{amp1[i]:.2f}rad'
                cv2.putText(vis, amp_text, (int(col1[i]) + 12, int(row1[i])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(vis, amp_text, (int(col1[i]) + 12, int(row1[i])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # 第二条边（洋红）
                cv2.circle(vis, (int(col2[i]), int(row2[i])), 6, (255, 0, 255), -1)
                cv2.circle(vis, (int(col2[i]), int(row2[i])), 8, (0, 0, 0), 2)
                
                # 显示梯度弧度值
                amp_text = f'{amp2[i]:.2f}rad'
                cv2.putText(vis, amp_text, (int(col2[i]) + 12, int(row2[i])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(vis, amp_text, (int(col2[i]) + 12, int(row2[i])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # 绘制连接线（蓝色）
                cv2.line(vis, (int(col1[i]), int(row1[i])), (int(col2[i]), int(row2[i])), 
                        (255, 0, 0), 2, cv2.LINE_AA)
                
                # 显示对内距离
                center_x = (col1[i] + col2[i]) / 2
                center_y = (row1[i] + row2[i]) / 2
                dist_text = f'{intra_dist[i]:.2f}px'
                cv2.putText(vis, dist_text, (int(center_x) - 20, int(center_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(vis, dist_text, (int(center_x) - 20, int(center_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # 绘制边对之间的连接线（显示对间距离）
            if len(centers_row) > 1:
                for i in range(len(centers_row) - 1):
                    # 绘制连接线
                    cv2.line(vis, (int(centers_col[i]), int(centers_row[i])),
                            (int(centers_col[i+1]), int(centers_row[i+1])),
                            (0, 100, 255), 1, cv2.LINE_AA)
                    
                    # 显示对间距离
                    mid_x = (centers_col[i] + centers_col[i+1]) / 2
                    mid_y = (centers_row[i] + centers_row[i+1]) / 2
                    dist_text = f'{inter_dist[i]:.2f}px'
                    cv2.putText(vis, dist_text, (int(mid_x) + 10, int(mid_y) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    cv2.putText(vis, dist_text, (int(mid_x) + 10, int(mid_y) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
        
        # 如果只有单个边缘结果，显示单个边缘
        elif row_edges is not None and len(row_edges) > 0:
            for i in range(len(row_edges)):
                # 根据幅度正负选择颜色
                if amplitudes[i] > 0:
                    color = (0, 255, 255)  # 黄色 - 正峰值（暗->亮）
                else:
                    color = (255, 0, 255)  # 洋红 - 负峰值（亮->暗）
                
                # 绘制圆点
                cv2.circle(vis, (int(col_edges[i]), int(row_edges[i])), 6, color, -1)
                cv2.circle(vis, (int(col_edges[i]), int(row_edges[i])), 8, (0, 0, 0), 2)
                
                # 显示梯度弧度值
                amp_text = f'{amplitudes[i]:.2f}rad'
                cv2.putText(vis, amp_text, (int(col_edges[i]) + 12, int(row_edges[i])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(vis, amp_text, (int(col_edges[i]) + 12, int(row_edges[i])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加标题
        title = 'Measurement Results'
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 添加ROI信息
        roi_info = f'ROI: ({self.col:.0f},{self.row:.0f}) {np.degrees(self.angle):.1f}°'
        cv2.putText(vis, roi_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis, roi_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示结果
        if wait_time >= 0:
            cv2.imshow(window_name, vis)
            cv2.waitKey(wait_time)
            if wait_time > 0:
                cv2.destroyAllWindows()
        
        return vis

# 测试和可视化函数
def create_test_image() -> np.ndarray:
    """创建测试图像：包含几个黑色条带"""
    img = np.ones((600, 800), dtype=np.uint8) * 200
    
    # 添加几个黑色条带（垂直）
    positions = [200, 300, 400, 500]
    width = 20
    for pos in positions:
        img[:, pos:pos+width] = 50
    
    # 添加一些噪声
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def visualize_measure_result(img: np.ndarray, measure: Halcon1DMeasure,
                             result_data: dict, window_name: str = 'Result'):
    """
    可视化测量结果
    """
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 绘制ROI框
    # 计算ROI的四个角点（按顺时针顺序：左上、右上、右下、左下）
    cos_a = np.cos(measure.angle)
    sin_a = np.sin(measure.angle)
    
    # 四个角点的偏移量
    dx_half = measure.length1 / 2
    dy_half = measure.length2 / 2
    
    # 左上、右上、右下、左下（顺时针顺序）
    corners = np.array([
        [
            measure.col - dx_half * cos_a + dy_half * sin_a,
            measure.row - dx_half * sin_a - dy_half * cos_a
        ],
        [
            measure.col + dx_half * cos_a + dy_half * sin_a,
            measure.row + dx_half * sin_a - dy_half * cos_a
        ],
        [
            measure.col + dx_half * cos_a - dy_half * sin_a,
            measure.row + dx_half * sin_a + dy_half * cos_a
        ],
        [
            measure.col - dx_half * cos_a - dy_half * sin_a,
            measure.row - dx_half * sin_a + dy_half * cos_a
        ]
    ], dtype=np.int32)
    cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
    
    # 根据结果类型绘制
    if 'pairs' in result_data and result_data['pairs']:
        # 绘制边缘对
        row1, col1, amp1, row2, col2, amp2 = result_data['pairs']
        centers_row, centers_col = result_data['centers']
        intra_dist = result_data['intra_dist']
        
        for i in range(len(row1)):
            # 第一条边（青色）
            cv2.circle(vis, (int(col1[i]), int(row1[i])), 3, (255, 255, 0), -1)
            # 第二条边（洋红）
            cv2.circle(vis, (int(col2[i]), int(row2[i])), 3, (255, 0, 255), -1)
            # 中心点
            cv2.circle(vis, (int(centers_col[i]), int(centers_row[i])), 2, (0, 0, 255), -1)
            
            # 显示宽度
            text = f'{intra_dist[i]:.2f}px'
            cv2.putText(vis, text, (int(centers_col[i]) - 20, int(centers_row[i]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    elif 'edges' in result_data and result_data['edges']:
        # 绘制单个边缘
        row_edges, col_edges, amplitudes = result_data['edges']
        
        for i in range(len(row_edges)):
            cv2.circle(vis, (int(col_edges[i]), int(row_edges[i])), 3, (0, 255, 255), -1)
            cv2.putText(vis, f'{amplitudes[i]:.1f}', 
                       (int(col_edges[i]) + 5, int(row_edges[i])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    cv2.imshow(window_name, vis)


def visualize_profile(profile: np.ndarray, window_name: str = 'Profile'):
    """
    可视化灰度轮廓
    """
    # 创建轮廓图像
    h = 300
    w = len(profile)
    profile_img = np.ones((h, w), dtype=np.uint8) * 200
    
    # 归一化轮廓到图像高度
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-10)
    profile_scaled = (profile_norm * (h - 40)).astype(np.int32)
    
    # 绘制轮廓
    points = []
    for i, val in enumerate(profile_scaled):
        points.append([i, h - 20 - val])
    
    points = np.array(points, dtype=np.int32)
    cv2.polylines(profile_img, [points], False, 0, 2)
    
    # 绘制基准线
    cv2.line(profile_img, (0, h - 20), (w, h - 20), 128, 1)
    
    cv2.imshow(window_name, profile_img)


# 使用示例
if __name__ == "__main__":
    # 创建测试图像
    img = create_test_image()
    h, w = img.shape
    
    # 创建测量对象（测量垂直方向的黑色条带）
    # angle = π/2 表示垂直方向
    measure = Halcon1DMeasure(
        row=300, 
        col=400, 
        angle=np.pi/2,  # 垂直方向
        length1=200,   # 测量长度
        length2=50,    # 测量宽度
        interpolation='linear'
    )
    
    # 测试 measure_pos
    print("=" * 50)
    print("测试 measure_pos:")
    print("=" * 50)
    row_edges, col_edges, amplitudes, distances = \
        measure.measure_pos(img, sigma=1.5, threshold=15.0, transition='all', select='all')
    
    print(f"检测到 {len(row_edges)} 个边缘:")
    for i in range(len(row_edges)):
        print(f"  边缘 {i+1}: 坐标({col_edges[i]:.2f}, {row_edges[i]:.2f}), "
              f"幅度={amplitudes[i]:.2f}")
    if distances:
        print(f"连续边缘间距: {distances}")
    
    # 可视化边缘检测结果
    visualize_measure_result(img, measure, {
        'edges': (row_edges, col_edges, amplitudes)
    }, 'Edges Result')
    
    # 测试 measure_pairs
    print("\n" + "=" * 50)
    print("测试 measure_pairs:")
    print("=" * 50)
    (row1, col1, amp1, row2, col2, amp2, 
     centers_row, centers_col, intra_dist, inter_dist) = \
        measure.measure_pairs(img, sigma=1.5, threshold=15.0, 
                             transition='negative', select='all')
    
    print(f"检测到 {len(row1)} 个边缘对:")
    for i in range(len(row1)):
        print(f"  边对 {i+1}:")
        print(f"    第一条边: ({col1[i]:.2f}, {row1[i]:.2f}), 幅度={amp1[i]:.2f}")
        print(f"    第二条边: ({col2[i]:.2f}, {row2[i]:.2f}), 幅度={amp2[i]:.2f}")
        print(f"    中心点: ({centers_col[i]:.2f}, {centers_row[i]:.2f})")
        print(f"    宽度: {intra_dist[i]:.2f} 像素")
    
    if inter_dist:
        print(f"边对间距: {inter_dist}")
    
    # 可视化边缘对检测结果
    visualize_measure_result(img, measure, {
        'pairs': (row1, col1, amp1, row2, col2, amp2),
        'centers': (centers_row, centers_col),
        'intra_dist': intra_dist
    }, 'Pairs Result')
    
    # 显示灰度轮廓
    profile = measure.get_profile(img)
    visualize_profile(profile, 'Gray Profile')
    
    # 显示原始图像
    cv2.imshow('Original Image', img)
    
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
