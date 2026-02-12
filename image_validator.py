"""
图片质量验证器
- 复用 analyzer.py 对生成图片进行 5 维度评分
- 关键检查点验证
- 失败时分析具体维度 → 返回调整建议
- 主图白底不达标时使用 OpenCV 阈值处理兜底
"""

import json
import logging
import os
import shutil

import cv2
import numpy as np
from PIL import Image

from config import QUALITY_THRESHOLDS, VALIDATION, IMAGE_SPECS, SINGLE_IMAGE_QUALITY
from analyzer import (
    analyze_product_images,
    load_image_pil, load_image_cv,
    score_sharpness, score_resolution,
    score_white_background, score_product_ratio,
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 单张候选图片质量验证
# ============================================================

def validate_single_image(image_path, image_type="lifestyle"):
    """
    验证单张候选图片的质量。

    参数:
        image_path: 图片文件路径
        image_type: 图片类型 ("main"/"lifestyle"/"detail"/"infographic"/"multiangle")

    返回:
        {
            "passed": bool,
            "score": float,        # 综合得分 (0-1)
            "details": {...},      # 各项评分明细
            "reasons": [...]       # 未通过的原因列表
        }
    """
    thresholds = SINGLE_IMAGE_QUALITY
    reasons = []
    details = {}

    img_pil = load_image_pil(image_path)
    img_cv = load_image_cv(image_path)

    if img_pil is None or img_cv is None:
        return {
            "passed": False,
            "score": 0.0,
            "details": {},
            "reasons": ["无法加载图片"],
        }

    # 清晰度 (1-5)
    sharpness = score_sharpness(img_cv)
    details["sharpness"] = sharpness
    if sharpness < thresholds["min_sharpness"]:
        reasons.append(f"清晰度不足: {sharpness}/5 (要求>={thresholds['min_sharpness']})")

    # 分辨率 (0-10)
    resolution, dimensions = score_resolution(img_pil)
    details["resolution"] = resolution
    details["dimensions"] = f"{dimensions[0]}x{dimensions[1]}"
    if resolution < thresholds["min_resolution"]:
        reasons.append(f"分辨率不足: {resolution}/10 (要求>={thresholds['min_resolution']})")

    # 主图额外检查
    if image_type == "main":
        white_bg, white_ratio = score_white_background(img_cv)
        details["white_bg"] = white_bg
        details["white_ratio"] = white_ratio
        if white_bg < thresholds["main_min_white_bg"]:
            reasons.append(f"白底不达标: {white_bg}/10 (要求>={thresholds['main_min_white_bg']})")

        product_ratio, ratio_val = score_product_ratio(img_cv)
        details["product_ratio"] = product_ratio
        details["product_ratio_val"] = ratio_val
        if product_ratio < thresholds["main_min_product_ratio"]:
            reasons.append(f"产品占比不足: {product_ratio}/10 (要求>={thresholds['main_min_product_ratio']})")

    # 综合得分 (归一化到 0-1)
    if image_type == "main":
        # 主图: 清晰度(5) + 分辨率(10) + 白底(10) + 占比(10) = 满分35
        max_score = 35.0
        total = sharpness + resolution + details.get("white_bg", 0) + details.get("product_ratio", 0)
    else:
        # 辅图: 清晰度(5) + 分辨率(10) = 满分15
        max_score = 15.0
        total = sharpness + resolution

    score = round(total / max_score, 3) if max_score > 0 else 0.0
    details["total"] = round(total, 1)

    return {
        "passed": len(reasons) == 0,
        "score": score,
        "details": details,
        "reasons": reasons,
    }


# ============================================================
# 后处理：兜底方案
# ============================================================

def force_white_background(image_path, threshold=240):
    """
    兜底方案：用 OpenCV 阈值处理强制将主图背景变为纯白。
    适用于 AI 生成的白底不够纯的情况。
    """
    img = cv2.imread(image_path)
    if img is None:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        logger.error(f"无法加载图片: {image_path}")
        return False

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建背景掩膜（亮度高于阈值的区域视为背景）
    _, bg_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 对掩膜进行形态学操作，填补空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 将背景设为纯白
    img[bg_mask == 255] = [255, 255, 255]

    # 保存
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, IMAGE_SPECS["output_quality"]]
    else:
        encode_param = []

    # 处理中文路径
    success, encoded = cv2.imencode(ext, img, encode_param)
    if success:
        encoded.tofile(image_path)
        logger.info(f"强制白底处理完成: {image_path}")
        return True

    return False


# ============================================================
# 验证逻辑
# ============================================================

def prepare_validation_dir(output_dir, product_name):
    """
    将生成的图片按 analyzer.py 期望的目录结构组织。
    analyzer 期望: images_dir/{asin}/main.jpg, sub_1.jpg, sub_2.jpg ...
    """
    val_base = os.path.join(output_dir, "_validation")
    val_dir = os.path.join(val_base, product_name)
    os.makedirs(val_dir, exist_ok=True)

    # 复制生成的图片到验证目录
    plan = IMAGE_SPECS["generation_plan"]
    for i, item in enumerate(plan):
        src = os.path.join(output_dir, item["filename"])
        if not os.path.exists(src):
            continue

        if item["type"] == "main":
            dst_name = "main.jpg"
        else:
            dst_name = f"sub_{i}.jpg"

        dst = os.path.join(val_dir, dst_name)
        shutil.copy2(src, dst)

    return val_base, product_name


def validate_generated_images(output_dir, product_name="generated"):
    """
    使用 analyzer.py 评分系统验证生成图片的质量。

    返回:
        dict: {
            "passed": bool,
            "scores": {...},         # 完整评分
            "failures": [...],       # 不达标的维度
            "suggestions": [...],    # 改进建议
        }
    """
    # 准备验证目录
    val_base, val_name = prepare_validation_dir(output_dir, product_name)

    # 调用 analyzer
    result = analyze_product_images(val_name, val_base)

    if result is None:
        logger.error("验证失败：analyzer 返回 None")
        # 清理验证目录
        _cleanup_validation_dir(val_base)
        return {
            "passed": False,
            "scores": {},
            "failures": ["analyzer_error"],
            "suggestions": ["检查生成的图片是否存在"],
        }

    # 检查关键阈值
    thresholds = QUALITY_THRESHOLDS
    failures = []
    suggestions = []

    # 检查1：整体分数
    normalized = result.get("normalized_score", 0)
    if normalized < thresholds["target_normalized_score"]:
        failures.append("normalized_score")
        suggestions.append(
            f"整体评分 {normalized} 低于目标 {thresholds['target_normalized_score']}，"
            "建议增加图片多样性或改善主图质量"
        )

    # 检查2：主图白底
    bg_score = result.get("main_compliance", {}).get("background_score", 0)
    if bg_score < thresholds["min_background_score"]:
        failures.append("background_score")
        suggestions.append(
            f"主图白底评分 {bg_score} 低于阈值 {thresholds['min_background_score']}，"
            "建议使用 force_white_background 强制处理"
        )

    # 检查3：产品占比
    ratio_score = result.get("main_compliance", {}).get("product_ratio_score", 0)
    if ratio_score < thresholds["min_product_ratio_score"]:
        failures.append("product_ratio_score")
        product_ratio = result.get("main_compliance", {}).get("product_ratio", 0)
        suggestions.append(
            f"产品占比评分 {ratio_score} (实际占比 {product_ratio:.0%})，"
            "建议调整 prompt 中的产品大小指令"
        )

    passed = len(failures) == 0

    # 清理验证目录
    _cleanup_validation_dir(val_base)

    validation_result = {
        "passed": passed,
        "scores": {
            "normalized_score": normalized,
            "technical": result.get("technical", {}).get("subtotal", 0),
            "main_compliance": result.get("main_compliance", {}).get("subtotal", 0),
            "sub_richness": result.get("sub_richness", {}).get("subtotal", 0),
            "design_quality": result.get("design_quality", {}).get("subtotal", 0),
            "marketing": result.get("marketing", {}).get("subtotal", 0),
            "background_score": bg_score,
            "product_ratio_score": ratio_score,
        },
        "full_result": result,
        "failures": failures,
        "suggestions": suggestions,
    }

    if passed:
        logger.info(f"验证通过! 评分: {normalized}/100")
    else:
        logger.warning(f"验证未通过 (评分: {normalized}/100)")
        for s in suggestions:
            logger.warning(f"  - {s}")

    return validation_result


def _cleanup_validation_dir(val_base):
    """清理临时验证目录"""
    try:
        if os.path.exists(val_base):
            shutil.rmtree(val_base)
    except Exception as e:
        logger.warning(f"清理验证目录失败: {e}")


# ============================================================
# 自动修复
# ============================================================

def auto_fix(output_dir, validation_result):
    """
    根据验证结果自动修复可修复的问题。
    返回修复了哪些问题。
    """
    fixes = []

    if "background_score" in validation_result.get("failures", []):
        main_path = os.path.join(output_dir, "main.jpg")
        if os.path.exists(main_path):
            if force_white_background(main_path):
                fixes.append("background_score: 强制白底处理")

    if "product_ratio_score" in validation_result.get("failures", []):
        # 产品占比问题难以自动修复，记录建议
        fixes.append("product_ratio_score: 需要重新生成（调整 prompt）")

    return fixes


# ============================================================
# 完整验证+修复流程
# ============================================================

def validate_and_fix(output_dir, product_name="generated", max_retries=None):
    """
    执行验证 → 自动修复 → 重新验证的循环。
    不包括重新生成（那是 optimizer_main 的职责）。

    返回最终验证结果。
    """
    if max_retries is None:
        max_retries = 1  # 自动修复最多尝试1次

    result = validate_generated_images(output_dir, product_name)

    if result["passed"]:
        return result

    # 尝试自动修复
    for attempt in range(max_retries):
        logger.info(f"自动修复尝试 {attempt + 1}/{max_retries}")
        fixes = auto_fix(output_dir, result)

        if not fixes:
            logger.info("没有可自动修复的问题")
            break

        for fix in fixes:
            logger.info(f"  已修复: {fix}")

        # 重新验证
        result = validate_generated_images(output_dir, product_name)
        if result["passed"]:
            logger.info("修复后验证通过!")
            return result

    return result
