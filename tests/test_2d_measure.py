"""
Metrology 模块测试用例

测试内容：
1. LineMeasureObject - 直线测量
2. CircleMeasureObject - 圆测量
3. MetrologyModel - 模型管理器
"""

import cv2
import numpy as np
import sys
import os
from typing import Tuple, List, Optional, Dict

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure.measure2D import LineMeasureObject, CircleMeasureObject, MetrologyModel


def create_test_image_with_line(width: int = 500, height: int = 400,
                                 line_angle: float = 0,
                                 line_position: Tuple[float, float] = (200, 250),
                                 line_thickness: int = 3) -> np.ndarray:
    """
    创建带有直线的测试图像
    
    参数:
        width: 图像宽度
        height: 图像高度
        line_angle: 直线角度（弧度）
        line_position: 直线通过的点 (row, col)
        line_thickness: 直线粗细
        
    返回:
        灰度测试图像
    """
    # 创建白色背景
    img = np.ones((height, width), dtype=np.uint8) * 200
    
    # 添加一些噪声
    noise = np.random.randint(-20, 20, (height, width), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 计算直线端点
    length = max(width, height) * 2
    cos_a = np.cos(line_angle)
    sin_a = np.sin(line_angle)
    
    row, col = line_position
    pt1 = (int(col - length * cos_a), int(row - length * sin_a))
    pt2 = (int(col + length * cos_a), int(row + length * sin_a))
    
    # 绘制黑色直线
    cv2.line(img, pt1, pt2, 50, line_thickness)
    
    # 添加一些高斯模糊
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    return img


def create_test_image_with_circle(width: int = 500, height: int = 500,
                                   center: Tuple[float, float] = (250, 250),
                                   radius: float = 100,
                                   thickness: int = 3) -> np.ndarray:
    """
    创建带有圆的测试图像
    
    参数:
        width: 图像宽度
        height: 图像高度
        center: 圆心 (row, col)
        radius: 半径
        thickness: 圆环粗细
        
    返回:
        灰度测试图像
    """
    # 创建白色背景
    img = np.ones((height, width), dtype=np.uint8) * 200
    
    # 添加一些噪声
    noise = np.random.randint(-20, 20, (height, width), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 绘制黑色圆环
    cv2.circle(img, (int(center[1]), int(center[0])), int(radius), 50, thickness)
    
    # 添加一些高斯模糊
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    return img


def create_test_image_with_rectangle(width: int = 500, height: int = 400) -> np.ndarray:
    """
    创建带有矩形的测试图像（用于测试多条直线测量）
    """
    img = np.ones((height, width), dtype=np.uint8) * 200
    
    # 添加噪声
    noise = np.random.randint(-15, 15, (height, width), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 绘制矩形
    cv2.rectangle(img, (100, 100), (400, 300), 50, 3)
    
    # 高斯模糊
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    return img


def create_test_image_with_multiple_circles(width: int = 600, height: int = 500) -> np.ndarray:
    """
    创建带有多个圆的测试图像
    """
    img = np.ones((height, width), dtype=np.uint8) * 200
    
    # 添加噪声
    noise = np.random.randint(-15, 15, (height, width), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 绘制多个圆
    circles = [
        ((150, 150), 80),   # 左上
        ((150, 450), 60),   # 右上
        ((350, 300), 100),  # 中间
    ]
    
    for center, radius in circles:
        cv2.circle(img, (int(center[1]), int(center[0])), radius, 50, 3)
    
    # 高斯模糊
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    return img

def sample_image_real():
        """创建一个测试用的简单图像"""
        #读取一张真实的图片
        img = cv2.imread('data/sample/bottleneck_2.jpg', cv2.IMREAD_GRAYSCALE)
        #将图片缩小2倍
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        return img
    
def sample_image_real_2():
    """创建一个测试用的简单图像"""
    #读取一张真实的图片
    img = cv2.imread('data/sample/bottleopen_2.jpg', cv2.IMREAD_GRAYSCALE)
    #将图片缩小2倍
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    return img

class TestLineMeasureObject:
    """直线测量对象测试"""
    
    
    @staticmethod
    def test_horizontal_line():
        """测试水平直线测量"""
        print("\n" + "="*60)
        print("测试：水平直线测量")
        print("="*60)
        
        # 创建测试图像（水平直线）
        img = create_test_image_with_line(
            width=500, height=400,
            line_angle=0,  # 水平
            line_position=(200, 250),  # row=200
            line_thickness=100
        )
        
        # 创建直线测量对象
        line_obj = LineMeasureObject(
            start=(250, 50),      # 起点 (row, col)
            end=(250, 450),       # 终点 (row, col)
            measure_length1=10,   # 沿直线方向半长度
            measure_length2=25,   # 垂直直线方向半宽度
            num_measures=12,
            sigma=1.0,
            threshold=20.0
        )
        
        # 执行测量
        result = line_obj.measure(img)
        line_obj.visualize(img,True,False,False,wait_time=1000)
        # 验证结果
        assert result is not None, "测量结果不应为空"
        assert result['num_points'] >= 10, f"边缘点数应>=10，实际为{result['num_points']}"
        
        # 验证直线角度（应该接近水平，即接近0或π）
        angle_deg = np.degrees(result['angle'])
        print(f"拟合直线角度: {angle_deg:.2f}°")
        print(f"边缘点数: {result['num_points']}")
        print(f"拟合误差: {result['mean_error']:.4f} px")
        
        # 验证直线位置（应该在 row=200 附近）
        line_row = result['point'][0]
        print(f"直线位置: row ≈ {line_row:.2f}")
        
        # 可视化
        vis_img = line_obj.visualize(img)
        
        # 保存结果
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'test_horizontal_line.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_horizontal_line.png")
        
        # 显示结果
        cv2.imshow('Horizontal Line Measure', vis_img)
        cv2.waitKey(100000)
        
        return True
    
    @staticmethod
    def test_vertical_line():
        """测试垂直直线测量"""
        print("\n" + "="*60)
        print("测试：垂直直线测量")
        print("="*60)
        
        # 创建测试图像（垂直直线）
        img = create_test_image_with_line(
            width=500, height=400,
            line_angle=np.pi/2,  # 垂直
            line_position=(200, 250)  # col=250
        )
        
        # 创建直线测量对象
        line_obj = LineMeasureObject(
            start=(50, 250),      # 起点 (row, col)
            end=(350, 250),       # 终点 (row, col)
            measure_length1=25,   # 沿直线方向半长度
            measure_length2=10,   # 垂直直线方向半宽度
            num_measures=10,
            sigma=1.0,
            threshold=20.0
        )
        
        # 执行测量
        result = line_obj.measure(img)
        
        assert result is not None, "测量结果不应为空"
        
        angle_deg = np.degrees(result['angle'])
        print(f"拟合直线角度: {angle_deg:.2f}°")
        print(f"边缘点数: {result['num_points']}")
        print(f"拟合误差: {result['mean_error']:.4f} px")
        
        # 验证直线位置（应该在 col=250 附近）
        line_col = result['point'][1]
        print(f"直线位置: col ≈ {line_col:.2f}")
        
        # 可视化
        vis_img = line_obj.visualize(img)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        cv2.imwrite(os.path.join(output_dir, 'test_vertical_line.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_vertical_line.png")
        
        cv2.imshow('Vertical Line Measure', vis_img)
        cv2.waitKey(1000)
        
        return True
    
    @staticmethod
    def test_diagonal_line():
        """测试斜线测量"""
        print("\n" + "="*60)
        print("测试：斜线测量")
        print("="*60)
        
        # 创建测试图像（45度斜线）
        angle = np.pi / 6  # 45度
        img = create_test_image_with_line(
            width=500, height=400,
            line_angle=angle,
            line_position=(200, 250),
            line_thickness=100
        )
        
        # 计算斜线的起点和终点
        length = 300
        center_row, center_col = 200, 350
        start = (center_row - length/2 * np.sin(angle), 
                 center_col - length/2 * np.cos(angle))
        end = (center_row + length/2 * np.sin(angle), 
               center_col + length/2 * np.cos(angle))
        
        # 创建直线测量对象
        line_obj = LineMeasureObject(
            start=start,
            end=end,
            measure_length1=40,
            measure_length2=100,
            num_measures=10,
            sigma=5.0,
            threshold=40
        )
        
        # 执行测量
        result = line_obj.measure(img)
        assert result is not None, "测量结果不应为空"
        
        # 可视化
        vis_img = line_obj.visualize(img)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        cv2.imwrite(os.path.join(output_dir, 'test_diagonal_line.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_diagonal_line.png")
        
        cv2.imshow('Diagonal Line Measure', vis_img)
        cv2.waitKey(1000)
        
        angle_deg = np.degrees(result['angle'])
        print(f"拟合直线角度: {angle_deg:.2f}° (预期约45°)")
        print(f"边缘点数: {result['num_points']}")
        print(f"拟合误差: {result['mean_error']:.4f} px")
        
       
        
        return True
    
    def test_horizontal_line_real(self):
        """测试水平直线测量"""
        print("\n" + "="*60)
        print("测试：水平直线测量")
        print("="*60)
        
        # 创建测试图像（水平直线）
        img = sample_image_real()
        
        # 创建直线测量对象
        line_obj = LineMeasureObject(
            start=(750 / 2,966 / 2),      # 起点 (row, col)
            end=(760/ 2,1626 / 2),       # 终点 (row, col)
            measure_length1=10,   # 沿直线方向半长度
            measure_length2=25,   # 垂直直线方向半宽度
            num_measures=12,
            sigma=1.0,
            threshold=20.0
        )
        
        # 执行测量
        result = line_obj.measure(img)
        line_obj.visualize(img,True,False,False,wait_time=10000)
        # 验证结果
        assert result is not None, "测量结果不应为空"
        assert result['num_points'] >= 10, f"边缘点数应>=10，实际为{result['num_points']}"
        
        # 验证直线角度（应该接近水平，即接近0或π）
        angle_deg = np.degrees(result['angle'])
        print(f"拟合直线角度: {angle_deg:.2f}°")
        print(f"边缘点数: {result['num_points']}")
        print(f"拟合误差: {result['mean_error']:.4f} px")
        
        # 验证直线位置（应该在 row=200 附近）
        line_row = result['point'][0]
        print(f"直线位置: row ≈ {line_row:.2f}")
        
        # 可视化
        vis_img = line_obj.visualize(img)
        
        # 保存结果
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'test_horizontal_line.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_horizontal_line.png")
        
        # 显示结果
        cv2.imshow('Horizontal Line Measure', vis_img)
        cv2.waitKey(100000)
        
        return True

class TestCircleMeasureObject:
    """圆测量对象测试"""
    
    @staticmethod
    def test_circle_measure():
        """测试圆测量"""
        print("\n" + "="*60)
        print("测试：圆测量")
        print("="*60)
        
        # 创建测试图像
        true_center = (250, 250)
        true_radius = 100
        img = create_test_image_with_circle(
            width=500, height=500,
            center=true_center,
            radius=true_radius
        )
        
        # 创建圆测量对象
        circle_obj = CircleMeasureObject(
            center=(250, 250),     # 预期圆心
            radius=100,            # 预期半径
            radius_min=80,         # 最小半径
            radius_max=120,        # 最大半径
            measure_length1=30,    # 径向半长度
            measure_length2=10,    # 切向半宽度
            num_measures=16,
            sigma=1.0,
            threshold=20.0
        )
        
        # 执行测量
        result = circle_obj.measure(img)
        
        assert result is not None, "测量结果不应为空"
        assert result['num_points'] >= 12, f"边缘点数应>=12，实际为{result['num_points']}"
        
        # 验证结果
        fitted_center = result['center']
        fitted_radius = result['radius']
        
        center_error = np.sqrt((fitted_center[0] - true_center[0])**2 + 
                               (fitted_center[1] - true_center[1])**2)
        radius_error = abs(fitted_radius - true_radius)
        
        print(f"真实圆心: ({true_center[1]}, {true_center[0]})")
        print(f"拟合圆心: ({fitted_center[1]:.2f}, {fitted_center[0]:.2f})")
        print(f"圆心误差: {center_error:.3f} px")
        print(f"真实半径: {true_radius}")
        print(f"拟合半径: {fitted_radius:.2f}")
        print(f"半径误差: {radius_error:.3f} px")
        print(f"边缘点数: {result['num_points']}")
        print(f"拟合误差: {result['mean_error']:.4f} px")
        
        # 验证精度
        assert center_error < 2.0, f"圆心误差过大: {center_error}"
        assert radius_error < 2.0, f"半径误差过大: {radius_error}"
        
        # 可视化
        vis_img = circle_obj.visualize(img, show_center_lines=True)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'test_circle.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_circle.png")
        
        cv2.imshow('Circle Measure', vis_img)
        cv2.waitKey(1000)
        
        return True
    
    @staticmethod
    def test_partial_circle():
        """测试部分圆（圆弧）测量"""
        print("\n" + "="*60)
        print("测试：圆弧测量")
        print("="*60)
        
        # 创建测试图像
        true_center = (250, 250)
        true_radius = 100
        img = create_test_image_with_circle(
            width=500, height=500,
            center=true_center,
            radius=true_radius
        )
        
        # 创建圆弧测量对象（只测量右半圆）
        circle_obj = CircleMeasureObject(
            center=(250, 250),
            radius=100,
            radius_min=80,
            radius_max=200,
            measure_length1=30,
            measure_length2=10,
            num_measures=10,
            sigma=1.0,
            threshold=20.0,
            start_phi=-np.pi/2,  # 从-90度开始
            end_phi=np.pi/2      # 到90度结束（右半圆）
        )
        
        # 执行测量
        result = circle_obj.measure(img)
        
        assert result is not None, "测量结果不应为空"
        
        print(f"拟合圆心: ({result['center'][1]:.2f}, {result['center'][0]:.2f})")
        print(f"拟合半径: {result['radius']:.2f}")
        print(f"边缘点数: {result['num_points']}")
        
        # 可视化
        vis_img = circle_obj.visualize(img, show_center_lines=True)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        cv2.imwrite(os.path.join(output_dir, 'test_partial_circle.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_partial_circle.png")
        
        cv2.imshow('Partial Circle Measure', vis_img)
        cv2.waitKey(10000)
        
        return True
    
    @staticmethod
    def test_small_circle():
        """测试小圆测量"""
        print("\n" + "="*60)
        print("测试：小圆测量")
        print("="*60)
        
        # 创建测试图像（小圆）
        true_center = (200, 200)
        true_radius = 30
        img = create_test_image_with_circle(
            width=400, height=400,
            center=true_center,
            radius=true_radius,
            thickness=2
        )
        
        # 创建圆测量对象
        circle_obj = CircleMeasureObject(
            center=(200, 200),
            radius=30,
            radius_min=20,
            radius_max=40,
            measure_length1=15,
            measure_length2=5,
            num_measures=12,
            sigma=1.0,
            threshold=15.0
        )
        
        # 执行测量
        result = circle_obj.measure(img)
        
        assert result is not None, "测量结果不应为空"
        
        print(f"拟合圆心: ({result['center'][1]:.2f}, {result['center'][0]:.2f})")
        print(f"拟合半径: {result['radius']:.2f}")
        print(f"边缘点数: {result['num_points']}")
        
        # 可视化
        vis_img = circle_obj.visualize(img)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        cv2.imwrite(os.path.join(output_dir, 'test_small_circle.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_small_circle.png")
        
        cv2.imshow('Small Circle Measure', vis_img)
        cv2.waitKey(1000)
        
        return True

    @staticmethod
    def test_circle_real():
        """测试真实的原图测量"""
        print("\n" + "="*60)
        print("测试：原图测量")
        print("="*60)
        
        # 创建测试图像（小圆）
        img = sample_image_real_2()
        # 创建圆测量对象
        circle_obj = CircleMeasureObject(
            center=(1016 / 2,1370 / 2),
            radius=200,
            radius_min=100,
            radius_max=220,
            measure_length1=200,
            measure_length2=20,
            num_measures=12,
            sigma=15.0,
            threshold=40.0
        )
        
        # 执行测量
        result = circle_obj.measure(img)

        # 可视化
        vis_img = circle_obj.visualize(img)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        cv2.imwrite(os.path.join(output_dir, 'test_real_circle.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_real_circle.png")
        
        cv2.imshow('real Circle Measure', vis_img)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        


class TestMetrologyModel:
    """Metrology 模型管理器测试"""
    
    @staticmethod
    def test_multiple_objects():
        """测试多对象测量"""
        print("\n" + "="*60)
        print("测试：多对象测量（矩形边缘）")
        print("="*60)
        
        # 创建测试图像（矩形）
        img = create_test_image_with_rectangle(width=500, height=400)
        
        # 创建 Metrology 模型
        model = MetrologyModel()
        
        # 添加四条直线测量
        # 上边
        idx_top = model.add_line_measure(
            start=(100, 100),
            end=(100, 400),
            measure_length1=10,
            measure_length2=20,
            num_measures=8
        )
        
        # 下边
        idx_bottom = model.add_line_measure(
            start=(300, 100),
            end=(300, 400),
            measure_length1=10,
            measure_length2=20,
            num_measures=8
        )
        
        # 左边
        idx_left = model.add_line_measure(
            start=(100, 100),
            end=(300, 100),
            measure_length1=20,
            measure_length2=10,
            num_measures=6
        )
        
        # 右边
        idx_right = model.add_line_measure(
            start=(100, 400),
            end=(300, 400),
            measure_length1=20,
            measure_length2=10,
            num_measures=6
        )
        
        # 执行测量
        model.measure(img)
        
        # 获取结果
        for name, idx in [('上边', idx_top), ('下边', idx_bottom), 
                          ('左边', idx_left), ('右边', idx_right)]:
            result = model.get_result(idx)
            if result:
                print(f"{name}: 角度={np.degrees(result['angle']):.2f}°, "
                      f"点数={result['num_points']}, 误差={result['mean_error']:.4f}")
        
        # 可视化
        vis_img = model.visualize(img)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'test_multiple_lines.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_multiple_lines.png")
        
        cv2.imshow('Multiple Lines Measure', vis_img)
        cv2.waitKey(1000)
        
        return True
    
    @staticmethod
    def test_mixed_objects():
        """测试混合对象测量（直线+圆）"""
        print("\n" + "="*60)
        print("测试：混合对象测量")
        print("="*60)
        
        # 创建带有矩形和圆的测试图像
        img = np.ones((500, 600), dtype=np.uint8) * 200
        noise = np.random.randint(-15, 15, (500, 600), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 绘制矩形
        cv2.rectangle(img, (50, 100), (200, 250), 50, 3)
        
        # 绘制圆
        cv2.circle(img, (450, 200), 80, 50, 3)
        
        # 高斯模糊
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # 创建 Metrology 模型
        model = MetrologyModel()
        
        # 添加直线测量（矩形的左边）
        idx_line = model.add_line_measure(
            start=(100, 50),
            end=(250, 50),
            measure_length1=20,
            measure_length2=10,
            num_measures=8
        )
        
        # 添加圆测量
        idx_circle = model.add_circle_measure(
            center=(200, 450),
            radius=80,
            radius_min=60,
            radius_max=100,
            measure_length1=25,
            measure_length2=8,
            num_measures=16
        )
        
        # 执行测量
        model.measure(img)
        
        # 获取结果
        line_result = model.get_result(idx_line)
        circle_result = model.get_result(idx_circle)
        
        if line_result:
            print(f"直线: 角度={np.degrees(line_result['angle']):.2f}°, "
                  f"点数={line_result['num_points']}")
        
        if circle_result:
            print(f"圆: 圆心=({circle_result['center'][1]:.1f}, {circle_result['center'][0]:.1f}), "
                  f"半径={circle_result['radius']:.2f}, 点数={circle_result['num_points']}")
        
        # 可视化
        vis_img = model.visualize(img)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        cv2.imwrite(os.path.join(output_dir, 'test_mixed_objects.png'), vis_img)
        print(f"可视化结果已保存到: {output_dir}/test_mixed_objects.png")
        
        cv2.imshow('Mixed Objects Measure', vis_img)
        cv2.waitKey(1000)
        
        return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" " * 15 + "Metrology 模块测试套件")
    print("="*70)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    tests = [
        ("直线测量 - 水平线", TestLineMeasureObject.test_horizontal_line),
        ("直线测量 - 垂直线", TestLineMeasureObject.test_vertical_line),
        ("直线测量 - 斜线", TestLineMeasureObject.test_diagonal_line),
        ("圆测量 - 完整圆", TestCircleMeasureObject.test_circle_measure),
        ("圆测量 - 圆弧", TestCircleMeasureObject.test_partial_circle),
        ("圆测量 - 小圆", TestCircleMeasureObject.test_small_circle),
        ("多对象测量 - 矩形", TestMetrologyModel.test_multiple_objects),
        ("混合对象测量", TestMetrologyModel.test_mixed_objects),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✓ {name}: 通过")
            else:
                failed += 1
                print(f"✗ {name}: 失败")
        except Exception as e:
            failed += 1
            print(f"✗ {name}: 异常 - {e}")
    
    print("\n" + "="*70)
    print(f"测试结果: 通过 {passed}/{len(tests)}, 失败 {failed}/{len(tests)}")
    print("="*70)
    
    cv2.destroyAllWindows()
    
    return failed == 0


if __name__ == '__main__':
    from typing import Tuple
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
