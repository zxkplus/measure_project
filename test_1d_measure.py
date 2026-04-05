import pytest
import cv2
import numpy as np
from measure1D import Halcon1DMeasure,create_test_image

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
    def sample_image_real(self):
        """创建一个测试用的简单图像"""
        #读取一张真实的图片
        img = cv2.imread('data/sample/bottleneck_2.jpg', cv2.IMREAD_GRAYSCALE)
        #将图片缩小2倍
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        return img
    
    @pytest.fixture
    def sample_image_real_2(self):
        """创建一个测试用的简单图像"""
        #读取一张真实的图片
        img = cv2.imread('data/sample/bottleopen_2.jpg', cv2.IMREAD_GRAYSCALE)
        #将图片缩小2倍
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
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
    

    def test_basic_debug(self,sample_image_real):
        """测试基本的调试功能"""
        print("=" * 60)
        print("测试基本调试功能")
        print("=" * 60)
        
        # 创建测试图像
        img = sample_image_real
        
        # 创建测量对象
        measure = Halcon1DMeasure(
            row=1366 / 2,
            col=1294 / 2,
            angle=0,
            length1=600,
            length2=50,
            interpolation='linear'
        )
        
        print("\n执行测量（debug=True）...")
        print("将显示多个调试窗口，请按任意键继续...")
        
        try:
            measure.draw_roi_on_image(img,wait_time=1000)
            # 执行测量（开启调试）
            row_edges, col_edges, amplitudes, distances = \
                measure.measure_pos(img, sigma=10, threshold=24.0, 
                                transition='all', select='all', debug=True)
            print(f"\n✓ 测试成功！")
            print(f"检测到 {len(row_edges)} 个边缘")
            print(f"边缘坐标:")
            for i in range(len(row_edges)):
                print(f"  [{i+1}] ({col_edges[i]:.2f}, {row_edges[i]:.2f}), "
                    f"幅度={amplitudes[i]:.2f}")
            # 显示边缘检测结果
            print("\n显示边缘检测结果...")
            measure.display_results(img, row_edges, col_edges, amplitudes, 
                                window_name='Edge Detection Results', wait_time=1000)
            print("\n✓ 测试成功完成！")
            
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败！")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_measure_pairs(self, sample_image_real):
       # 创建测试图像
        img = sample_image_real
        
        # 创建测量对象
        measure = Halcon1DMeasure(
            row=1366 / 2,
            col=1294 / 2,
            angle=0,
            length1=600,
            length2=50,
            interpolation='linear'
        )
        
        print("\n执行测量（debug=True）...")
        print("将显示多个调试窗口，请按任意键继续...")
        
        try:
            measure.draw_roi_on_image(img,wait_time=1000)
            # 执行边缘对检测
            print("\n执行边缘对检测...")
            (row1, col1, amp1, row2, col2, amp2, 
            center_row, center_col, intra_dist, inter_dist) = \
                measure.measure_pairs(img, sigma=15, threshold=79.0, 
                                transition='all', select='all',debug=True)
            
            print(f"\n✓ 边缘对检测成功！")
            print(f"检测到 {len(row1)} 个边缘对")
            print(f"边缘对详情:")
            for i in range(len(row1)):
                print(f"  [{i+1}] 第一条边: ({col1[i]:.2f}, {row1[i]:.2f}), 幅度={amp1[i]:.2f}")
                print(f"       第二条边: ({col2[i]:.2f}, {row2[i]:.2f}), 幅度={amp2[i]:.2f}")
                print(f"       中心点: ({center_col[i]:.2f}, {center_row[i]:.2f})")
                print(f"       宽度: {intra_dist[i]:.2f} 像素")
            
            if inter_dist:
                print(f"\n边缘对间距: {[f'{d:.2f}' for d in inter_dist]}")
            
            # 显示边缘对检测结果
            print("\n显示边缘对检测结果...")
            measure.display_results(img, row_edges=None, col_edges=None, amplitudes=None,
                                row1=row1, col1=col1, amp1=amp1,
                                row2=row2, col2=col2, amp2=amp2,
                                centers_row=center_row, centers_col=center_col,
                                intra_dist=intra_dist, inter_dist=inter_dist,
                                window_name='Pair Detection Results', wait_time=500000)
            
            print("\n✓ 测试成功完成！")
            
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败！")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            return False
    def test_measure_pairs_angle_90(self, sample_image_real_2):
       # 创建测试图像
        img = sample_image_real_2
        
        # 创建测量对象
        measure = Halcon1DMeasure(
            row=1066 / 2,
            col=1294 / 2,
            angle= np.pi / 4  ,#测试垂直的情况
            length1=600,
            length2=50,
            interpolation='linear'
        )
        
        print("\n执行测量（debug=True）...")
        print("将显示多个调试窗口，请按任意键继续...")
        
        try:
            measure.draw_roi_on_image(img,wait_time=1000)
            # 执行边缘对检测
            print("\n执行边缘对检测...")
            (row1, col1, amp1, row2, col2, amp2, 
            center_row, center_col, intra_dist, inter_dist) = \
                measure.measure_pairs(img, sigma=10, threshold=14.8, 
                                transition='all', select='all',debug=True)
            
            print(f"\n✓ 边缘对检测成功！")
            print(f"检测到 {len(row1)} 个边缘对")
            print(f"边缘对详情:")
            for i in range(len(row1)):
                print(f"  [{i+1}] 第一条边: ({col1[i]:.2f}, {row1[i]:.2f}), 幅度={amp1[i]:.2f}")
                print(f"       第二条边: ({col2[i]:.2f}, {row2[i]:.2f}), 幅度={amp2[i]:.2f}")
                print(f"       中心点: ({center_col[i]:.2f}, {center_row[i]:.2f})")
                print(f"       宽度: {intra_dist[i]:.2f} 像素")
            
            if inter_dist:
                print(f"\n边缘对间距: {[f'{d:.2f}' for d in inter_dist]}")
            
            # 显示边缘对检测结果
            print("\n显示边缘对检测结果...")
            measure.display_results(img, row_edges=None, col_edges=None, amplitudes=None,
                                row1=row1, col1=col1, amp1=amp1,
                                row2=row2, col2=col2, amp2=amp2,
                                centers_row=center_row, centers_col=center_col,
                                intra_dist=intra_dist, inter_dist=inter_dist,
                                window_name='Pair Detection Results', wait_time=500000)
            
            print("\n✓ 测试成功完成！")
            
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败！")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_measure_pairs_angle_45(self, sample_image_real_2):
       # 创建测试图像
        img = sample_image_real_2
        
        # 创建测量对象
        measure = Halcon1DMeasure(
            row=1066 / 2,
            col=1294 / 2,
            angle= np.pi / 4  ,#测试垂直的情况
            length1=600,
            length2=50,
            interpolation='linear'
        )
        
        print("\n执行测量（debug=True）...")
        print("将显示多个调试窗口，请按任意键继续...")
        
        try:
            measure.draw_roi_on_image(img,wait_time=1000)
            # 执行边缘对检测
            print("\n执行边缘对检测...")
            (row1, col1, amp1, row2, col2, amp2, 
            center_row, center_col, intra_dist, inter_dist) = \
                measure.measure_pairs(img, sigma=10, threshold=12.0, 
                                transition='all', select='all',debug=True)
            
            print(f"\n✓ 边缘对检测成功！")
            print(f"检测到 {len(row1)} 个边缘对")
            print(f"边缘对详情:")
            for i in range(len(row1)):
                print(f"  [{i+1}] 第一条边: ({col1[i]:.2f}, {row1[i]:.2f}), 幅度={amp1[i]:.2f}")
                print(f"       第二条边: ({col2[i]:.2f}, {row2[i]:.2f}), 幅度={amp2[i]:.2f}")
                print(f"       中心点: ({center_col[i]:.2f}, {center_row[i]:.2f})")
                print(f"       宽度: {intra_dist[i]:.2f} 像素")
            
            if inter_dist:
                print(f"\n边缘对间距: {[f'{d:.2f}' for d in inter_dist]}")
            
            # 显示边缘对检测结果
            print("\n显示边缘对检测结果...")
            measure.display_results(img, row_edges=None, col_edges=None, amplitudes=None,
                                row1=row1, col1=col1, amp1=amp1,
                                row2=row2, col2=col2, amp2=amp2,
                                centers_row=center_row, centers_col=center_col,
                                intra_dist=intra_dist, inter_dist=inter_dist,
                                window_name='Pair Detection Results', wait_time=500000)
            
            print("\n✓ 测试成功完成！")
            
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败！")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            return False
        
