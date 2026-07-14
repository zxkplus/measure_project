import cv2
import numpy as np
import glob
import os
 
# --- 1. 参数设置 ---
CIRCLE_GRID = (7, 7)  # 圆形标定板行列数
circle_spacing = 15.0  # 圆心间距（毫米）
# 网格类型标志：SYMMETRIC_GRID 用于对称网格，ASYMMETRIC_GRID 用于非对称网格
flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def sort_corners_canonical(corners, grid_size):
    """将 findCirclesGrid 检测到的点重排为统一的行优先（左上->右下）顺序。
    
    Args:
        corners: (N,1,2) 数组，findCirclesGrid 返回的圆心坐标。
        grid_size: (rows, cols) 网格行列数。
    Returns:
        (N,1,2) 重排后的圆心坐标。
    """
    rows, cols = grid_size
    pts = corners.reshape(-1, 2)
    # 按 y 排序，然后每 rows 个一组为一行，行内按 x 排序
    y_order = np.argsort(pts[:, 1])
    sorted_pts = pts[y_order]
    # 按行切分
    row_arrays = np.array_split(sorted_pts, rows)
    canonical = []
    for row in row_arrays:
        row_sorted = row[np.argsort(row[:, 0])]
        canonical.append(row_sorted)
    result = np.vstack(canonical).astype(np.float32)
    return result.reshape(-1, 1, 2)

# --- 2. 准备世界坐标系点 (x,y,z) -> z=0 ---
rows, cols = CIRCLE_GRID
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * circle_spacing
 
objpoints = [] # 世界坐标点集合
imgpoints_l = [] # 左图像素点集合
imgpoints_r = [] # 右图像素点集合
 
# --- 3. 读取图片并提取角点 ---
base_dir = "/home/industai/桌面/temp/圆形标定板/part1"
left_images = []
right_images = []
for i in range(1, 10):  # 1 to 9
    folder = os.path.join(base_dir, str(i))
    left_images.append(os.path.join(folder, "2.jpg"))
    right_images.append(os.path.join(folder, "1.jpg"))
 
for idx, (fname_l, fname_r) in enumerate(zip(left_images, right_images), 1):
    img_l = cv2.imread(fname_l)
    img_r = cv2.imread(fname_r)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
 
    # 寻找圆心
    ret_l, corners_l = cv2.findCirclesGrid(gray_l, CIRCLE_GRID, flags=flags)
    ret_r, corners_r = cv2.findCirclesGrid(gray_r, CIRCLE_GRID, flags=flags)
    # 统一排序：按行优先（左上->右下）排列
    if ret_l:
        corners_l = sort_corners_canonical(corners_l, CIRCLE_GRID)
    if ret_r:
        corners_r = sort_corners_canonical(corners_r, CIRCLE_GRID)
 
    if ret_l and ret_r:
        objpoints.append(objp)
        
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        
        # 可视化检测效果（保存到文件）
        cv2.drawChessboardCorners(img_l, CIRCLE_GRID, corners_l, ret_l)
        output_dir = 'output/circle_detection'
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f'left_{idx}.jpg'), img_l)
        cv2.drawChessboardCorners(img_r, CIRCLE_GRID, corners_r, ret_r)
        cv2.imwrite(os.path.join(output_dir, f'right_{idx}.jpg'), img_r)
        print(f'图像对 {idx}: 检测到圆心，已保存可视化结果。')
    else:
        print(f'图像对 {idx}: 未检测到圆心，请检查图像质量。')
        
# --- 4. 执行标定 ---
# 获取图像尺寸
h, w = gray_l.shape[:2]

# 先单目标定获取内参初值
print('\n左相机单目标定...')
ret_l, K1, D1, rv1, tv1 = cv2.calibrateCamera(
    objpoints, imgpoints_l, (w, h), None, None,
    criteria=criteria,
)
print(f'  左 RMS: {ret_l:.4f}')

print('右相机单目标定...')
ret_r, K2, D2, rv2, tv2 = cv2.calibrateCamera(
    objpoints, imgpoints_r, (w, h), None, None,
    criteria=criteria,
)
print(f'  右 RMS: {ret_r:.4f}')

# 用单目内参加双目标定
print('\n双目标定中...')
retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K1, D1, K2, D2, (w, h),
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)
 
print("=== 标定结果 ===")
print(f"左相机内参:\n{K1}")
print(f"左相机畸变:\n{D1}")
print(f"右相机内参:\n{K2}")
print(f"右相机畸变:\n{D2}")
print(f"旋转矩阵 R:\n{R}")
print(f"平移向量 T:\n{T}") # T的范数即为基线距离



# --- 5. 立体校正 ---
# 计算校正变换矩阵
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    K1, D1, K2, D2, (w, h), R, T, 
    alpha=0, # 0表示裁剪掉黑边，1表示保留所有像素
    flags=cv2.CALIB_ZERO_DISPARITY
)
 
# 生成查找表 (Map)
map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)
 
# --- 6. 应用校正并显示 ---
img_l = cv2.imread('/home/industai/桌面/temp/圆形标定板/part1/1/2.jpg')
img_r = cv2.imread('/home/industai/桌面/temp/圆形标定板/part1/1/1.jpg')
 
rectified_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
rectified_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)
 
# 将左右图拼在一起查看效果
result = np.hstack([rectified_l, rectified_r])

# 画几条横线验证是否对齐
for i in range(0, result.shape[0], 50):
    cv2.line(result, (0, i), (result.shape[1], i), (0, 255, 0), 1)
cv2.imwrite('output/rectified_result.jpg', result)
