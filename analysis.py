"""
数据分析与洞察
分析热销商品图片特征，发现高评分/高评论商品的图片共同特征
生成分析结论
"""

import json
import os
import logging
import math
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def safe_int(val, default=0):
    """安全转换为int"""
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    """安全转换为float"""
    try:
        s = str(val).replace(",", "").replace("$", "").strip()
        return float(s)
    except (ValueError, TypeError):
        return default


def pearson_correlation(x, y):
    """计算皮尔逊相关系数"""
    n = len(x)
    if n < 3:
        return 0.0

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    std_x = np.std(x)
    std_y = np.std(y)

    if std_x == 0 or std_y == 0:
        return 0.0

    correlation = np.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)
    return round(float(correlation), 4)


def split_groups(results, key_func, top_n=None, bottom_n=None):
    """按指标分为高低两组"""
    sorted_results = sorted(results, key=key_func, reverse=True)

    if top_n is None:
        top_n = max(len(results) // 4, 5)
    if bottom_n is None:
        bottom_n = max(len(results) // 4, 5)

    top_group = sorted_results[:top_n]
    bottom_group = sorted_results[-bottom_n:]

    return top_group, bottom_group


def compute_group_avg(group, metric_path):
    """
    计算一组数据的某个指标平均值
    metric_path 如 "technical.subtotal" 或 "normalized_score"
    """
    values = []
    for r in group:
        parts = metric_path.split(".")
        val = r
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part, 0)
            else:
                val = 0
                break
        values.append(safe_float(val))

    return round(sum(values) / max(len(values), 1), 2)


def analyze_correlations(results):
    """分析各评分维度与销量（评论数）的相关性"""
    reviews = [safe_int(r.get("reviews", 0)) for r in results]
    ratings = [safe_float(r.get("rating", 0)) for r in results]

    metrics = {
        "总分": [r.get("normalized_score", 0) for r in results],
        "技术指标": [r.get("technical", {}).get("subtotal", 0) for r in results],
        "主图合规": [r.get("main_compliance", {}).get("subtotal", 0) for r in results],
        "辅图丰富度": [r.get("sub_richness", {}).get("subtotal", 0) for r in results],
        "设计质量": [r.get("design_quality", {}).get("subtotal", 0) for r in results],
        "营销效果": [r.get("marketing", {}).get("subtotal", 0) for r in results],
        "图片数量": [r.get("image_count", 0) for r in results],
        "分辨率评分": [r.get("technical", {}).get("resolution_score", 0) for r in results],
        "白底评分": [r.get("main_compliance", {}).get("background_score", 0) for r in results],
        "多角度评分": [r.get("sub_richness", {}).get("multi_angle_score", 0) for r in results],
        "场景图数量": [r.get("sub_richness", {}).get("scene_count", 0) for r in results],
        "清晰度评分": [r.get("design_quality", {}).get("sharpness_score", 0) for r in results],
    }

    correlations = {}
    for name, values in metrics.items():
        corr_reviews = pearson_correlation(values, reviews)
        corr_ratings = pearson_correlation(values, ratings)
        correlations[name] = {
            "与评论数相关性": corr_reviews,
            "与评分相关性": corr_ratings,
        }

    return correlations


def analyze_top_vs_bottom(results):
    """高销量 vs 低销量商品的图片特征对比"""
    # 按评论数分组
    top_group, bottom_group = split_groups(
        results,
        key_func=lambda r: safe_int(r.get("reviews", 0))
    )

    metrics_to_compare = [
        ("normalized_score", "图片总分"),
        ("image_count", "图片数量"),
        ("technical.subtotal", "技术指标"),
        ("main_compliance.subtotal", "主图合规"),
        ("main_compliance.background_score", "白底评分"),
        ("main_compliance.product_ratio_score", "产品占比"),
        ("sub_richness.subtotal", "辅图丰富度"),
        ("sub_richness.scene_count", "场景图数"),
        ("sub_richness.infographic_count", "信息图数"),
        ("sub_richness.detail_count", "细节图数"),
        ("design_quality.subtotal", "设计质量"),
        ("design_quality.sharpness_score", "清晰度"),
        ("marketing.subtotal", "营销效果"),
    ]

    comparison = {}
    for metric_path, name in metrics_to_compare:
        top_avg = compute_group_avg(top_group, metric_path)
        bottom_avg = compute_group_avg(bottom_group, metric_path)
        diff_pct = round((top_avg - bottom_avg) / max(bottom_avg, 0.01) * 100, 1)

        comparison[name] = {
            "高销量组平均": top_avg,
            "低销量组平均": bottom_avg,
            "差异百分比": f"{diff_pct:+.1f}%",
        }

    return comparison


def identify_common_features(results):
    """识别热销商品图片的共同特征"""
    # 取评论数Top25%的商品
    sorted_by_reviews = sorted(
        results,
        key=lambda r: safe_int(r.get("reviews", 0)),
        reverse=True
    )
    n = max(len(sorted_by_reviews) // 4, 5)
    top_sellers = sorted_by_reviews[:n]

    features = []

    # 图片数量
    avg_imgs = sum(r.get("image_count", 0) for r in top_sellers) / max(len(top_sellers), 1)
    has_7_plus = sum(1 for r in top_sellers if r.get("image_count", 0) >= 7)
    features.append(
        f"图片数量: 热销商品平均 {avg_imgs:.1f} 张图片，"
        f"{has_7_plus}/{len(top_sellers)} ({has_7_plus/max(len(top_sellers),1)*100:.0f}%) 有7张以上图片"
    )

    # 白底合规
    avg_bg = sum(r.get("main_compliance", {}).get("background_score", 0) for r in top_sellers) / max(len(top_sellers), 1)
    features.append(f"主图白底: 热销商品平均白底评分 {avg_bg:.1f}/10")

    # 场景图
    avg_scene = sum(r.get("sub_richness", {}).get("scene_count", 0) for r in top_sellers) / max(len(top_sellers), 1)
    features.append(f"场景图: 热销商品平均有 {avg_scene:.1f} 张场景图")

    # 信息图
    avg_info = sum(r.get("sub_richness", {}).get("infographic_count", 0) for r in top_sellers) / max(len(top_sellers), 1)
    features.append(f"信息图: 热销商品平均有 {avg_info:.1f} 张信息图")

    # 清晰度
    avg_sharp = sum(r.get("design_quality", {}).get("sharpness_score", 0) for r in top_sellers) / max(len(top_sellers), 1)
    features.append(f"清晰度: 热销商品平均清晰度评分 {avg_sharp:.1f}/5")

    # 总分
    avg_total = sum(r.get("normalized_score", 0) for r in top_sellers) / max(len(top_sellers), 1)
    all_avg = sum(r.get("normalized_score", 0) for r in results) / max(len(results), 1)
    features.append(f"总分对比: 热销商品平均 {avg_total:.1f}分 vs 全部商品平均 {all_avg:.1f}分")

    return features


def generate_insights(analysis_file="analysis_results.json", output_file="insights_report.json"):
    """生成完整的分析洞察报告"""
    analysis_path = os.path.join(BASE_DIR, analysis_file)
    output_path = os.path.join(BASE_DIR, output_file)

    if not os.path.exists(analysis_path):
        logger.error(f"分析结果文件不存在: {analysis_path}")
        return None

    with open(analysis_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        logger.error("没有分析结果")
        return None

    logger.info(f"开始数据分析，共 {len(results)} 条记录...")

    # 1. 相关性分析
    logger.info("计算相关性...")
    correlations = analyze_correlations(results)

    # 2. 高低销量对比
    logger.info("高低销量对比分析...")
    comparison = analyze_top_vs_bottom(results)

    # 3. 热销商品共同特征
    logger.info("识别热销商品共同特征...")
    features = identify_common_features(results)

    # 4. 汇总统计
    scores = [r.get("normalized_score", 0) for r in results]
    summary_stats = {
        "总商品数": len(results),
        "平均评分": round(sum(scores) / max(len(scores), 1), 1),
        "最高评分": round(max(scores), 1) if scores else 0,
        "最低评分": round(min(scores), 1) if scores else 0,
        "中位评分": round(sorted(scores)[len(scores) // 2], 1) if scores else 0,
        "标准差": round(float(np.std(scores)), 1) if scores else 0,
    }

    # 5. 最值得关注的相关性
    key_insights = []
    for metric, corrs in correlations.items():
        r_val = corrs["与评论数相关性"]
        if abs(r_val) >= 0.2:
            direction = "正相关" if r_val > 0 else "负相关"
            strength = "强" if abs(r_val) >= 0.5 else "中等" if abs(r_val) >= 0.3 else "弱"
            key_insights.append(
                f"{metric} 与销量呈{strength}{direction} (r={r_val})"
            )

    # 整合报告
    report = {
        "summary": summary_stats,
        "correlations": correlations,
        "top_vs_bottom": comparison,
        "top_seller_features": features,
        "key_insights": key_insights,
    }

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印关键发现
    logger.info("\n" + "=" * 60)
    logger.info("关键发现:")
    logger.info("=" * 60)

    logger.info(f"\n总商品数: {summary_stats['总商品数']}")
    logger.info(f"平均图片质量评分: {summary_stats['平均评分']}/100")

    logger.info("\n--- 热销商品图片共同特征 ---")
    for f_item in features:
        logger.info(f"  - {f_item}")

    logger.info("\n--- 关键相关性 ---")
    for insight in key_insights:
        logger.info(f"  - {insight}")

    if not key_insights:
        logger.info("  (未发现显著相关性，可能样本量不足)")

    logger.info("\n--- 高销量 vs 低销量图片对比 ---")
    for name, vals in comparison.items():
        logger.info(f"  {name}: 高销量={vals['高销量组平均']} vs 低销量={vals['低销量组平均']} ({vals['差异百分比']})")

    logger.info(f"\n分析报告已保存到: {output_file}")
    return report


if __name__ == "__main__":
    generate_insights()
