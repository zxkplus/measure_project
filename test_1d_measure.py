import pytest
import cv2
import numpy as np
from measure import Halcon1DMeasure,create_test_image

class TestHalcon1DMeasure:
    """测试Halcon1DMeasure类的各个功能"""
    
    @pytest.fixture
    def sample_image(self):
        """创建一个测试用的简单图像"""
        # 创建一个400x400的灰度图像
        img = np.zeros((400, 400), dtype=np.uint8)
        
        # 在图像中央添加一个垂直的亮条（模拟边缘）
        cv2.rectangle(img, (180, 100), (220, 300), 255, -1)
        
        # 添加一些噪声
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    @pytest.fixture
    def measure_obj(self, sample_image):
        """创建一个测量对象"""
        h, w = sample_image.shape
        return Halcon1DMeasure(
            row=200, col=200, angle=0,
            length1=100, length2=40,
            interpolation='bilinear'
        )
    
    def test_initialization(self, measure_obj):
        """测试初始化参数是否正确设置"""
        assert measure_obj.row == 200
        assert measure_obj.col == 200
        assert measure_obj.angle == 0
        assert measure_obj.length1 == 100
        assert measure_obj.length2 == 40
        assert measure_obj.interpolation == 'bilinear'
    

    def test_basic_debug(self,sample_image):
        """测试基本的调试功能"""
        print("=" * 60)
        print("测试基本调试功能")
        print("=" * 60)
        
        # 创建测试图像
        img = create_test_image()
        
        # 创建测量对象
        measure = Halcon1DMeasure(
            row=300,
            col=400,
            angle=0,
            length1=400,
            length2=50,
            interpolation='linear'
        )
        
        print("\n执行测量（debug=True）...")
        print("将显示多个调试窗口，请按任意键继续...")
        
        try:
            measure.draw_roi_on_image(img,wait_time=1000)
            # 执行测量（开启调试）
            row_edges, col_edges, amplitudes, distances = \
                measure.measure_pos(img, sigma=1.5, threshold=15.0, 
                                transition='all', select='all', debug=True)
            
            print(f"\n✓ 测试成功！")
            print(f"检测到 {len(row_edges)} 个边缘")
            print(f"边缘坐标:")
            for i in range(len(row_edges)):
                print(f"  [{i+1}] ({col_edges[i]:.2f}, {row_edges[i]:.2f}), "
                    f"幅度={amplitudes[i]:.2f}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败！")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_measure_pairs(self, measure_obj, sample_image):
        """测试边缘对测量功能"""
        # 执行测量
        (row1, col1, amp1, row2, col2, amp2, 
         center_row, center_col, intra_dist, inter_dist) = measure_obj.measure_pairs(
            sample_image, sigma=1.0, threshold=30.0, transition='negative', select='all'
        )
        
        # 检查是否检测到了边缘对
        assert len(row1) > 0
        assert len(col1) > 0
        assert len(amp1) > 0
        assert len(row2) > 0
        assert len(col2) > 0
        assert len(amp2) > 0
        assert len(center_row) > 0
        assert len(center_col) > 0
        assert len(intra_dist) > 0
        
        # 检查边缘对之间的距离是否合理
        for dist in intra_dist:
            assert 0 < dist < 100  # 边缘对之间的距离应该在合理范围内
    
    def test_draw_measure_region_gray(self, measure_obj, sample_image):
        """测试绘制测量区域功能（灰度图输入）"""
        # 绘制测量区域
        result = measure_obj.draw_measure_region(sample_image)
        
        # 检查输出是否为彩色图像
        assert len(result.shape) == 3
        assert result.shape[2] == 3
        
        # 检查图像尺寸是否保持不变
        assert result.shape[:2] == sample_image.shape
    
    def test_draw_measure_region_color(self, measure_obj, sample_image):
        """测试绘制测量区域功能（彩色图输入）"""
        # 将灰度图转换为彩色图
        color_img = cv2.cvtColor(sample_image, cv2.COLOR_GRAY2BGR)
        
        # 绘制测量区域
        result = measure_obj.draw_measure_region(color_img)
        
        # 检查输出是否为彩色图像
        assert len(result.shape) == 3
        assert result.shape[2] == 3
        
        # 检查图像尺寸是否保持不变
        assert result.shape == color_img.shape
        
        # 检查原图是否未被修改
        assert np.array_equal(color_img, cv2.cvtColor(sample_image, cv2.COLOR_GRAY2BGR))
    
    def test_draw_measure_region_custom_colors(self, measure_obj, sample_image):
        """测试使用自定义颜色绘制测量区域"""
        # 使用自定义颜色绘制
        result = measure_obj.draw_measure_region(
            sample_image, 
            rect_color=(255, 0, 0),  # 蓝色矩形
            axis_color=(0, 255, 0)   # 绿色轴线
        )
        
        # 检查输出是否为彩色图像
        assert len(result.shape) == 3
        assert result.shape[2] == 3
        
        # 检查图像尺寸是否保持不变
        assert result.shape[:2] == sample_image.shape
