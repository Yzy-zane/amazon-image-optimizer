"""
图片质量分析器
对Amazon商品图片进行多维度质量评估：
- 技术指标（分辨率、文件大小、数量、格式）
- 主图合规性（白底、产品占比、文字检测、阴影）
- 辅图丰富度（多角度、场景图、信息图、细节图）
- 设计质量（光线、构图、色彩、清晰度）
- 营销效果（卖点突出、视觉吸引、场景代入、信息完整）
"""

import json
import os
import logging
import math
from collections import defaultdict

import numpy as np
from PIL import Image, ImageStat
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")


# ============================================================
# 工具函数
# ============================================================

def load_image_pil(path):
    """用PIL加载图片"""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"无法加载图片 {path}: {e}")
        return None


def load_image_cv(path):
    """用OpenCV加载图片"""
    try:
        img = cv2.imread(path)
        if img is None:
            # 尝试处理中文路径
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"无法加载图片(cv) {path}: {e}")
        return None


def clamp_score(score, min_val, max_val):
    """限制分数范围"""
    return max(min_val, min(max_val, score))


# ============================================================
# 3.1 技术指标 (客观)
# ============================================================

def score_resolution(img_pil):
    """
    分辨率评分 (0-10)
    >=2000x2000 → 10分
    >=1500x1500 → 8分
    >=1000x1000 → 6分
    >=500x500 → 3分
    <500x500 → 1分
    """
    if img_pil is None:
        return 0, (0, 0)
    w, h = img_pil.size
    min_dim = min(w, h)
    max_dim = max(w, h)

    if min_dim >= 2000:
        score = 10
    elif min_dim >= 1500:
        score = 8 + 2 * (min_dim - 1500) / 500
    elif min_dim >= 1000:
        score = 6 + 2 * (min_dim - 1000) / 500
    elif min_dim >= 500:
        score = 3 + 3 * (min_dim - 500) / 500
    else:
        score = max(1, 3 * min_dim / 500)

    return clamp_score(round(score, 1), 0, 10), (w, h)


def score_file_size(file_path):
    """
    文件大小评分 (0-5)
    100KB-2MB → 5分 (理想范围)
    2MB-5MB → 4分
    50KB-100KB → 3分
    >5MB → 2分
    <50KB → 1分
    """
    try:
        size_bytes = os.path.getsize(file_path)
    except OSError:
        return 0, 0

    size_kb = size_bytes / 1024

    if 100 <= size_kb <= 2048:
        score = 5
    elif 2048 < size_kb <= 5120:
        score = 4
    elif 50 <= size_kb < 100:
        score = 3
    elif size_kb > 5120:
        score = 2
    else:
        score = max(0.5, size_kb / 50)

    return clamp_score(round(score, 1), 0, 5), size_kb


def score_image_count(image_files):
    """
    图片数量评分 (0-10)
    >=7张 → 10分
    6张 → 9分
    5张 → 7分
    4张 → 5分
    3张 → 3分
    2张 → 2分
    1张 → 1分
    """
    count = len(image_files)
    if count >= 7:
        score = 10
    elif count == 6:
        score = 9
    elif count == 5:
        score = 7
    elif count == 4:
        score = 5
    elif count == 3:
        score = 3
    elif count == 2:
        score = 2
    elif count == 1:
        score = 1
    else:
        score = 0

    return score, count


def score_format_quality(img_pil, file_path):
    """
    格式质量评分 (0-5)
    检查图片格式和压缩质量
    """
    if img_pil is None:
        return 0

    score = 3  # 基础分

    fmt = img_pil.format or ""
    ext = os.path.splitext(file_path)[1].lower()

    # JPEG/PNG是Amazon首选格式
    if fmt.upper() in ("JPEG", "PNG") or ext in (".jpg", ".jpeg", ".png"):
        score += 1

    # 检查是否为sRGB色彩空间（更好的颜色表现）
    if img_pil.mode == "RGB":
        score += 0.5

    # 检查图片宽高比（正方形或接近正方形更好）
    w, h = img_pil.size
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio <= 1.2:
        score += 0.5  # 接近正方形

    return clamp_score(round(score, 1), 0, 5)


# ============================================================
# 3.2 主图合规性 (客观)
# ============================================================

def score_white_background(img_cv):
    """
    背景纯白检测 (0-10)
    分析四角及边缘像素是否接近白色 RGB≈(255,255,255)
    """
    if img_cv is None:
        return 0, 0.0

    h, w = img_cv.shape[:2]
    margin = max(5, min(w, h) // 20)  # 边缘检测区域

    # 提取四个角和边缘区域的像素
    regions = [
        img_cv[0:margin, 0:margin],                # 左上
        img_cv[0:margin, w - margin:w],             # 右上
        img_cv[h - margin:h, 0:margin],             # 左下
        img_cv[h - margin:h, w - margin:w],         # 右下
        img_cv[0:margin, :],                        # 上边
        img_cv[:, 0:margin],                        # 左边
        img_cv[:, w - margin:w],                    # 右边
    ]

    white_ratios = []
    for region in regions:
        if region.size == 0:
            continue
        # 检查像素是否接近白色 (B,G,R都>240)
        white_mask = np.all(region > 240, axis=2)
        ratio = np.mean(white_mask)
        white_ratios.append(ratio)

    if not white_ratios:
        return 0, 0.0

    avg_white_ratio = np.mean(white_ratios)

    if avg_white_ratio >= 0.95:
        score = 10
    elif avg_white_ratio >= 0.85:
        score = 8 + 2 * (avg_white_ratio - 0.85) / 0.10
    elif avg_white_ratio >= 0.70:
        score = 5 + 3 * (avg_white_ratio - 0.70) / 0.15
    elif avg_white_ratio >= 0.50:
        score = 2 + 3 * (avg_white_ratio - 0.50) / 0.20
    else:
        score = max(0, 2 * avg_white_ratio / 0.50)

    return clamp_score(round(score, 1), 0, 10), round(avg_white_ratio, 3)


def score_product_ratio(img_cv):
    """
    产品占比评分 (0-10)
    非白色区域面积/总面积，目标≥85%时满分
    """
    if img_cv is None:
        return 0, 0.0

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 白色像素判定：灰度值>240
    non_white_mask = gray < 240
    product_ratio = np.mean(non_white_mask)

    # 如果是白底图，产品占比在合理范围内得分高
    # 太小说明产品不够突出，太大说明可能没有白底
    if 0.30 <= product_ratio <= 0.85:
        # 理想范围：30%-85%
        if product_ratio >= 0.60:
            score = 8 + 2 * (product_ratio - 0.60) / 0.25
        else:
            score = 5 + 3 * (product_ratio - 0.30) / 0.30
    elif product_ratio > 0.85:
        # 产品占比过大，可能整张图都是产品（也可以）
        score = 7
    elif product_ratio >= 0.15:
        score = 3 + 2 * (product_ratio - 0.15) / 0.15
    else:
        score = max(1, 3 * product_ratio / 0.15)

    return clamp_score(round(score, 1), 0, 10), round(product_ratio, 3)


def score_text_logo_detection(img_cv):
    """
    文字/Logo检测 (0-5)
    在主图中检测文字区域，主图应该尽量少文字
    5分 = 无文字/Logo, 0分 = 大量文字覆盖
    """
    if img_cv is None:
        return 0

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 分析小而密集的连通区域（可能是文字）
    text_like_regions = 0
    total_text_area = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        # 文字特征：小面积、特定宽高比
        if 10 < area < (w * h * 0.01) and 0.2 < cw / max(ch, 1) < 5:
            text_like_regions += 1
            total_text_area += area

    text_coverage = total_text_area / (w * h) if (w * h) > 0 else 0

    # 少文字得高分
    if text_coverage < 0.01:
        score = 5
    elif text_coverage < 0.03:
        score = 4
    elif text_coverage < 0.05:
        score = 3
    elif text_coverage < 0.10:
        score = 2
    else:
        score = 1

    return clamp_score(score, 0, 5)


def score_shadow_detection(img_cv):
    """
    阴影/道具检测 (0-5)
    分析底部区域的灰度梯度变化
    5分 = 无明显阴影, 0分 = 严重阴影
    """
    if img_cv is None:
        return 0

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 分析底部20%区域
    bottom_region = gray[int(h * 0.8):h, :]

    # 计算灰度梯度
    if bottom_region.size == 0:
        return 5

    gradient_y = cv2.Sobel(bottom_region, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.mean(np.abs(gradient_y))

    # 底部区域的平均亮度
    mean_brightness = np.mean(bottom_region)

    # 白底图底部应该很亮（接近255），且梯度小
    if mean_brightness > 245 and gradient_magnitude < 5:
        score = 5  # 非常干净
    elif mean_brightness > 230 and gradient_magnitude < 10:
        score = 4
    elif mean_brightness > 200:
        score = 3
    elif mean_brightness > 150:
        score = 2
    else:
        score = 1

    return clamp_score(score, 0, 5)


# ============================================================
# 3.3 辅图丰富度
# ============================================================

def compute_histogram(img_cv):
    """计算图片的颜色直方图"""
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def score_multi_angle(image_files, images_dir):
    """
    多角度展示评分 (0-10)
    通过图片间直方图差异度来判断是否展示了不同角度
    """
    if len(image_files) < 2:
        return 0, 0

    histograms = []
    for f in image_files:
        img = load_image_cv(os.path.join(images_dir, f))
        if img is not None:
            histograms.append(compute_histogram(img))

    if len(histograms) < 2:
        return 0, 0

    # 计算所有图片对的直方图差异
    differences = []
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            diff = cv2.compareHist(
                histograms[i].reshape(-1, 1).astype(np.float32),
                histograms[j].reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA
            )
            differences.append(diff)

    avg_diff = np.mean(differences) if differences else 0

    # 差异度越大说明角度越丰富
    if avg_diff >= 0.6:
        score = 10
    elif avg_diff >= 0.4:
        score = 7 + 3 * (avg_diff - 0.4) / 0.2
    elif avg_diff >= 0.2:
        score = 4 + 3 * (avg_diff - 0.2) / 0.2
    else:
        score = max(1, 4 * avg_diff / 0.2)

    return clamp_score(round(score, 1), 0, 10), round(avg_diff, 3)


def is_scene_image(img_cv):
    """判断是否为场景图（非纯色背景）"""
    if img_cv is None:
        return False

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 分析边缘区域的颜色复杂度
    margin = max(10, min(w, h) // 10)
    edges_regions = [
        gray[0:margin, :],
        gray[h - margin:h, :],
        gray[:, 0:margin],
        gray[:, w - margin:w],
    ]

    # 如果边缘区域的标准差大，说明背景复杂（场景图）
    stds = [np.std(r) for r in edges_regions if r.size > 0]
    avg_std = np.mean(stds) if stds else 0

    # 同时检查非白色/非纯色的比例
    non_uniform = np.std(gray)

    return avg_std > 20 and non_uniform > 40


def is_infographic(img_cv):
    """判断是否为信息图（包含大量文字/高对比区域）"""
    if img_cv is None:
        return False

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算高对比区域（黑白交替频繁的区域）
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)

    # 信息图通常有较高的边缘密度和较多的水平/垂直线条
    # 使用形态学操作检测文字行
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    h_line_ratio = np.mean(h_lines > 0)

    return edge_density > 0.08 and h_line_ratio > 0.05


def is_detail_image(img_cv):
    """判断是否为细节图（高频信息密度高）"""
    if img_cv is None:
        return False

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Laplacian检测高频信息
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = np.var(laplacian)

    # 细节图通常有很高的高频信息密度
    return lap_var > 1000


def score_scene_images(image_files, images_dir):
    """
    场景图评分 (0-5)
    """
    scene_count = 0
    for f in image_files:
        if f.startswith("main"):
            continue  # 跳过主图
        img = load_image_cv(os.path.join(images_dir, f))
        if is_scene_image(img):
            scene_count += 1

    if scene_count >= 3:
        score = 5
    elif scene_count == 2:
        score = 4
    elif scene_count == 1:
        score = 2
    else:
        score = 0

    return score, scene_count


def score_infographics(image_files, images_dir):
    """
    信息图评分 (0-5)
    """
    info_count = 0
    for f in image_files:
        if f.startswith("main"):
            continue
        img = load_image_cv(os.path.join(images_dir, f))
        if is_infographic(img):
            info_count += 1

    if info_count >= 3:
        score = 5
    elif info_count == 2:
        score = 4
    elif info_count == 1:
        score = 2
    else:
        score = 0

    return score, info_count


def score_detail_images(image_files, images_dir):
    """
    细节图评分 (0-5)
    """
    detail_count = 0
    for f in image_files:
        if f.startswith("main"):
            continue
        img = load_image_cv(os.path.join(images_dir, f))
        if is_detail_image(img):
            detail_count += 1

    if detail_count >= 2:
        score = 5
    elif detail_count == 1:
        score = 3
    else:
        score = 0

    return score, detail_count


# ============================================================
# 3.4 设计质量
# ============================================================

def score_lighting_uniformity(img_cv):
    """
    光线均匀度 (1-5)
    亮度直方图标准差越小越均匀
    """
    if img_cv is None:
        return 1

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 将图片分成网格，计算每个区域的平均亮度
    h, w = gray.shape
    grid_size = 4
    cell_h, cell_w = h // grid_size, w // grid_size

    brightnesses = []
    for r in range(grid_size):
        for c in range(grid_size):
            cell = gray[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            brightnesses.append(np.mean(cell))

    std = np.std(brightnesses)

    if std < 15:
        score = 5
    elif std < 25:
        score = 4
    elif std < 40:
        score = 3
    elif std < 60:
        score = 2
    else:
        score = 1

    return score


def score_composition_balance(img_cv):
    """
    构图平衡 (1-5)
    质心偏移分析，越居中越平衡
    """
    if img_cv is None:
        return 1

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 对非白色区域计算质心
    mask = gray < 240  # 非白色区域
    if np.sum(mask) == 0:
        return 3  # 全白图

    # 计算质心
    y_coords, x_coords = np.where(mask)
    cx = np.mean(x_coords) / w  # 归一化到0-1
    cy = np.mean(y_coords) / h

    # 计算到中心的偏移
    offset = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    if offset < 0.05:
        score = 5
    elif offset < 0.10:
        score = 4
    elif offset < 0.15:
        score = 3
    elif offset < 0.25:
        score = 2
    else:
        score = 1

    return score


def score_color_richness(img_cv):
    """
    色彩表现 (1-5)
    HSV色彩空间饱和度分析
    """
    if img_cv is None:
        return 1

    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # 饱和度通道
    saturation = hsv[:, :, 1]

    # 排除白色/灰色背景区域
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    product_mask = gray < 240

    if np.sum(product_mask) == 0:
        return 2

    product_saturation = saturation[product_mask]
    mean_sat = np.mean(product_saturation)
    # 色相分布
    hue = hsv[:, :, 0][product_mask]
    hue_std = np.std(hue)

    # 综合评分
    if mean_sat > 80 and hue_std > 30:
        score = 5
    elif mean_sat > 60 or hue_std > 25:
        score = 4
    elif mean_sat > 40:
        score = 3
    elif mean_sat > 20:
        score = 2
    else:
        score = 1

    return score


def score_sharpness(img_cv):
    """
    清晰度 (1-5)
    Laplacian方差，越高越清晰
    """
    if img_cv is None:
        return 1

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = np.var(laplacian)

    if variance > 500:
        score = 5
    elif variance > 200:
        score = 4
    elif variance > 100:
        score = 3
    elif variance > 50:
        score = 2
    else:
        score = 1

    return score


# ============================================================
# 3.5 营销效果
# ============================================================

def score_selling_point_highlight(image_files, images_dir):
    """
    卖点突出度 (1-5)
    信息图文字密度 + 对比度
    """
    if not image_files:
        return 1

    total_edge_density = 0
    count = 0

    for f in image_files:
        if f.startswith("main"):
            continue
        img = load_image_cv(os.path.join(images_dir, f))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        total_edge_density += np.mean(edges > 0)
        count += 1

    avg_density = total_edge_density / max(count, 1)

    if avg_density > 0.12:
        return 5
    elif avg_density > 0.08:
        return 4
    elif avg_density > 0.05:
        return 3
    elif avg_density > 0.02:
        return 2
    else:
        return 1


def score_visual_appeal(img_cv):
    """
    视觉吸引力 (1-5)
    色彩丰富度 + 对比度综合评分
    """
    if img_cv is None:
        return 1

    # 对比度
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)

    # 色彩丰富度
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    sat_mean = np.mean(hsv[:, :, 1])
    hue_unique = len(np.unique(hsv[:, :, 0] // 10))  # 色相桶数

    combined = (contrast / 80) * 0.4 + (sat_mean / 128) * 0.3 + (hue_unique / 18) * 0.3

    if combined > 0.8:
        return 5
    elif combined > 0.6:
        return 4
    elif combined > 0.4:
        return 3
    elif combined > 0.2:
        return 2
    else:
        return 1


def score_scene_immersion(scene_count, total_sub_images):
    """
    场景代入感 (1-5)
    """
    if total_sub_images == 0:
        return 1

    ratio = scene_count / total_sub_images
    if ratio >= 0.4 and scene_count >= 2:
        return 5
    elif ratio >= 0.3 or scene_count >= 2:
        return 4
    elif scene_count >= 1:
        return 3
    elif total_sub_images >= 3:
        return 2
    else:
        return 1


def score_info_completeness(scene_count, info_count, detail_count, total_images):
    """
    信息完整性 (1-5)
    辅图覆盖类型数量
    """
    types_covered = 0
    if scene_count > 0:
        types_covered += 1
    if info_count > 0:
        types_covered += 1
    if detail_count > 0:
        types_covered += 1
    if total_images >= 5:
        types_covered += 1

    if types_covered >= 4:
        return 5
    elif types_covered == 3:
        return 4
    elif types_covered == 2:
        return 3
    elif types_covered == 1:
        return 2
    else:
        return 1


# ============================================================
# 综合分析
# ============================================================

def analyze_product_images(asin, images_dir=IMAGES_DIR):
    """
    分析单个商品的所有图片，返回完整评分结果
    """
    product_dir = os.path.join(images_dir, asin)

    if not os.path.isdir(product_dir):
        logger.warning(f"{asin}: 图片目录不存在")
        return None

    # 获取所有图片文件
    image_files = sorted([
        f for f in os.listdir(product_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])

    if not image_files:
        logger.warning(f"{asin}: 没有图片文件")
        return None

    # 找到主图
    main_file = None
    sub_files = []
    for f in image_files:
        if f.startswith("main"):
            main_file = f
        else:
            sub_files.append(f)

    if not main_file and image_files:
        main_file = image_files[0]
        sub_files = image_files[1:]

    main_path = os.path.join(product_dir, main_file) if main_file else None
    main_pil = load_image_pil(main_path) if main_path else None
    main_cv = load_image_cv(main_path) if main_path else None

    result = {
        "asin": asin,
        "image_count": len(image_files),
        "main_image": main_file,
        "sub_images": sub_files,
    }

    # ---- 3.1 技术指标 ----
    res_score, resolution = score_resolution(main_pil)
    size_score, file_size_kb = score_file_size(main_path) if main_path else (0, 0)
    count_score, count = score_image_count(image_files)
    format_score = score_format_quality(main_pil, main_path) if main_path else 0

    result["technical"] = {
        "resolution_score": res_score,
        "resolution": f"{resolution[0]}x{resolution[1]}" if resolution != (0, 0) else "N/A",
        "file_size_score": size_score,
        "file_size_kb": round(file_size_kb, 1),
        "image_count_score": count_score,
        "image_count": count,
        "format_score": format_score,
        "subtotal": round(res_score + size_score + count_score + format_score, 1)
    }

    # ---- 3.2 主图合规性 ----
    bg_score, white_ratio = score_white_background(main_cv)
    ratio_score, product_ratio = score_product_ratio(main_cv)
    text_score = score_text_logo_detection(main_cv)
    shadow_score = score_shadow_detection(main_cv)

    result["main_compliance"] = {
        "background_score": bg_score,
        "white_ratio": white_ratio,
        "product_ratio_score": ratio_score,
        "product_ratio": product_ratio,
        "text_logo_score": text_score,
        "shadow_score": shadow_score,
        "subtotal": round(bg_score + ratio_score + text_score + shadow_score, 1)
    }

    # ---- 3.3 辅图丰富度 ----
    angle_score, avg_diff = score_multi_angle(image_files, product_dir)
    scene_score, scene_count = score_scene_images(image_files, product_dir)
    info_score, info_count = score_infographics(image_files, product_dir)
    detail_score, detail_count = score_detail_images(image_files, product_dir)

    result["sub_richness"] = {
        "multi_angle_score": angle_score,
        "histogram_diff": avg_diff,
        "scene_score": scene_score,
        "scene_count": scene_count,
        "infographic_score": info_score,
        "infographic_count": info_count,
        "detail_score": detail_score,
        "detail_count": detail_count,
        "subtotal": round(angle_score + scene_score + info_score + detail_score, 1)
    }

    # ---- 3.4 设计质量 ----
    lighting = score_lighting_uniformity(main_cv)
    composition = score_composition_balance(main_cv)
    color = score_color_richness(main_cv)
    sharpness = score_sharpness(main_cv)

    result["design_quality"] = {
        "lighting_score": lighting,
        "composition_score": composition,
        "color_score": color,
        "sharpness_score": sharpness,
        "subtotal": round(lighting + composition + color + sharpness, 1)
    }

    # ---- 3.5 营销效果 ----
    selling_point = score_selling_point_highlight(image_files, product_dir)
    appeal = score_visual_appeal(main_cv)
    immersion = score_scene_immersion(scene_count, len(sub_files))
    completeness = score_info_completeness(scene_count, info_count, detail_count, len(image_files))

    result["marketing"] = {
        "selling_point_score": selling_point,
        "visual_appeal_score": appeal,
        "scene_immersion_score": immersion,
        "info_completeness_score": completeness,
        "subtotal": round(selling_point + appeal + immersion + completeness, 1)
    }

    # ---- 总分 ----
    total = (
        result["technical"]["subtotal"] +
        result["main_compliance"]["subtotal"] +
        result["sub_richness"]["subtotal"] +
        result["design_quality"]["subtotal"] +
        result["marketing"]["subtotal"]
    )
    result["total_score"] = round(total, 1)

    # 最高可能分数: 30 + 30 + 25 + 20 + 20 = 125 (大致)
    # 归一化到100分
    max_possible = 10 + 5 + 10 + 5 + 10 + 10 + 5 + 5 + 10 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5
    result["normalized_score"] = round(total / max_possible * 100, 1) if max_possible > 0 else 0

    return result


def analyze_all_products(products_file="products.json", images_dir=None, output_file="analysis_results.json"):
    """分析所有商品的图片质量"""
    if images_dir is None:
        images_dir = IMAGES_DIR

    products_path = os.path.join(BASE_DIR, products_file)
    output_path = os.path.join(BASE_DIR, output_file)

    # 加载商品数据
    if os.path.exists(products_path):
        with open(products_path, "r", encoding="utf-8") as f:
            products = json.load(f)
    else:
        # 如果没有products.json，直接扫描images目录
        products = []
        if os.path.isdir(images_dir):
            for asin_dir in os.listdir(images_dir):
                if os.path.isdir(os.path.join(images_dir, asin_dir)):
                    products.append({"asin": asin_dir})

    logger.info(f"开始分析 {len(products)} 个商品的图片质量...")

    results = []
    for i, product in enumerate(products):
        asin = product.get("asin", "")
        if not asin:
            continue

        logger.info(f"[{i+1}/{len(products)}] 分析 {asin}...")
        analysis = analyze_product_images(asin, images_dir)

        if analysis:
            # 合并商品信息
            analysis["title"] = product.get("title", "N/A")
            analysis["price"] = product.get("price", "N/A")
            analysis["rating"] = product.get("rating", "N/A")
            analysis["reviews"] = product.get("reviews", "0")
            analysis["rank"] = product.get("rank", 0)
            results.append(analysis)

    # 按总分排序
    results.sort(key=lambda x: x.get("total_score", 0), reverse=True)

    # 保存分析结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"分析完成! {len(results)} 个商品，结果保存到 {output_file}")
    return results


if __name__ == "__main__":
    results = analyze_all_products()
    if results:
        logger.info(f"\n===== 图片质量 Top 5 =====")
        for r in results[:5]:
            logger.info(f"  {r['asin']}: 总分 {r['total_score']} ({r['normalized_score']}分/100) - {r.get('title', 'N/A')[:40]}")
