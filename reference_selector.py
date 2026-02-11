"""
最佳参考选择器
- 读取 analysis_results.json，按加权分数排序
- 选出 Top 3 参考商品
- 提取参考组图的构成模式（场景图/信息图/细节图的数量比例）
- 输出参考摘要，供 prompt 生成使用
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_FILE = os.path.join(BASE_DIR, "analysis_results.json")


def load_analysis_results(filepath=None):
    """加载分析结果"""
    filepath = filepath or ANALYSIS_FILE
    if not os.path.exists(filepath):
        logger.error(f"分析结果文件不存在: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_weighted_score(result):
    """
    计算加权评分，侧重对图片生成最关键的维度：
    - 辅图丰富度权重最高（数量和多样性最重要）
    - 营销效果权重次之
    - 主图合规性也重要
    - 技术指标和设计质量权重较低（可以靠后处理保证）
    """
    weights = {
        "sub_richness": 0.30,     # 辅图丰富度
        "marketing": 0.25,        # 营销效果
        "main_compliance": 0.20,  # 主图合规
        "design_quality": 0.15,   # 设计质量
        "technical": 0.10,        # 技术指标
    }

    # 各维度满分
    max_scores = {
        "technical": 30,
        "main_compliance": 30,
        "sub_richness": 25,
        "design_quality": 20,
        "marketing": 20,
    }

    weighted = 0
    for dim, weight in weights.items():
        subtotal = result.get(dim, {}).get("subtotal", 0)
        max_score = max_scores[dim]
        normalized = subtotal / max_score * 100 if max_score > 0 else 0
        weighted += normalized * weight

    return round(weighted, 1)


def extract_composition_pattern(result):
    """提取单个商品的组图构成模式"""
    sub_richness = result.get("sub_richness", {})
    return {
        "asin": result.get("asin", ""),
        "title": result.get("title", "")[:60],
        "normalized_score": result.get("normalized_score", 0),
        "weighted_score": compute_weighted_score(result),
        "image_count": result.get("image_count", 0),
        "scene_count": sub_richness.get("scene_count", 0),
        "infographic_count": sub_richness.get("infographic_count", 0),
        "detail_count": sub_richness.get("detail_count", 0),
        "multi_angle_score": sub_richness.get("multi_angle_score", 0),
        "background_score": result.get("main_compliance", {}).get("background_score", 0),
        "product_ratio": result.get("main_compliance", {}).get("product_ratio", 0),
    }


def select_top_references(results=None, top_n=3):
    """
    选出 Top N 参考商品
    返回参考摘要列表
    """
    if results is None:
        results = load_analysis_results()

    if not results:
        logger.warning("没有分析结果可供选择")
        return []

    # 按加权分数排序
    scored = []
    for r in results:
        pattern = extract_composition_pattern(r)
        scored.append(pattern)

    scored.sort(key=lambda x: x["weighted_score"], reverse=True)

    top = scored[:top_n]

    logger.info(f"Top {top_n} 参考商品:")
    for i, ref in enumerate(top, 1):
        logger.info(f"  #{i} {ref['asin']} - 加权分 {ref['weighted_score']}, "
                     f"标准分 {ref['normalized_score']}, "
                     f"图片 {ref['image_count']}张 "
                     f"(场景{ref['scene_count']} 信息{ref['infographic_count']} "
                     f"细节{ref['detail_count']})")

    return top


def compute_ideal_composition(references):
    """
    根据参考商品的组图模式，计算理想的图片构成
    取参考商品各类型图片数量的中位数/众数
    """
    if not references:
        # 默认构成（来自计划）
        return {
            "total": 9,
            "main": 1,
            "lifestyle": 2,
            "infographic": 2,
            "detail": 2,
            "multiangle": 2,
        }

    scene_counts = [r["scene_count"] for r in references]
    info_counts = [r["infographic_count"] for r in references]
    detail_counts = [r["detail_count"] for r in references]
    image_counts = [r["image_count"] for r in references]

    # 使用中位数
    def median(lst):
        s = sorted(lst)
        n = len(s)
        if n == 0:
            return 0
        return s[n // 2]

    ideal_scene = max(2, median(scene_counts))
    ideal_info = max(2, median(info_counts))
    ideal_detail = max(2, median(detail_counts))

    # 确保总数合理 (主图1 + 辅图)
    total_sub = ideal_scene + ideal_info + ideal_detail
    # 如果辅图太少，补充多角度图
    multiangle = max(1, 8 - total_sub)

    composition = {
        "total": 1 + ideal_scene + ideal_info + ideal_detail + multiangle,
        "main": 1,
        "lifestyle": int(ideal_scene),
        "infographic": int(ideal_info),
        "detail": int(ideal_detail),
        "multiangle": int(multiangle),
    }

    logger.info(f"理想组图构成: {composition}")
    return composition


def get_reference_summary(analysis_file=None):
    """
    获取完整的参考摘要（供 prompt_engine 使用）
    返回: {
        "references": [...],      # Top 3 参考商品详情
        "composition": {...},     # 理想组图构成
        "avg_scores": {...},      # 参考商品平均分
    }
    """
    results = load_analysis_results(analysis_file)
    references = select_top_references(results)
    composition = compute_ideal_composition(references)

    # 计算参考商品的平均分（作为生成目标）
    avg_scores = {}
    if references:
        for key in ["normalized_score", "weighted_score", "background_score"]:
            vals = [r[key] for r in references]
            avg_scores[key] = round(sum(vals) / len(vals), 1)

    summary = {
        "references": references,
        "composition": composition,
        "avg_scores": avg_scores,
    }

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    summary = get_reference_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
