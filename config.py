"""
配置系统
- 从 config.json 加载 API Key（OpenAI / Flux / SiliconFlow）
- 定义质量阈值（来自 insights_report.json 的高分商品特征）
- 定义图片生成规格（尺寸、数量、类型分配）
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


# ============================================================
# API 配置
# ============================================================

def load_api_config():
    """从 config.json 加载 API 配置"""
    if not os.path.exists(CONFIG_FILE):
        logger.warning(f"配置文件不存在: {CONFIG_FILE}")
        logger.warning("请创建 config.json，参考格式：")
        logger.warning('  {"openai_api_key": "sk-xxx", "openai_base_url": "", '
                        '"flux_api_key": "", "primary_model": "gpt-4o"}')
        return {}

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def get_openai_api_key():
    config = load_api_config()
    return config.get("openai_api_key", "")


def get_openai_base_url():
    config = load_api_config()
    url = config.get("openai_base_url", "")
    return url if url else None  # 空字符串返回 None，使用官方默认地址


def get_flux_api_key():
    config = load_api_config()
    return config.get("flux_api_key", "")


def get_siliconflow_api_key():
    config = load_api_config()
    return config.get("siliconflow_api_key", "")


def get_primary_model():
    config = load_api_config()
    return config.get("primary_model", "siliconflow")


# ============================================================
# 质量阈值（基于 100 商品分析的高分特征）
# ============================================================

# 来自 insights_report.json 的关键数据
QUALITY_THRESHOLDS = {
    # 整体目标分数（全部商品平均 73.9，目标超过平均）
    "target_normalized_score": 75,

    # 主图白底合规
    "min_background_score": 8,        # 白底纯度 ≥ 95% 对应 ~8-10 分
    "min_product_ratio_score": 6,     # 产品占比 60-85%

    # 图片数量
    "target_image_count": 9,          # 主图 + 8张辅图
    "min_image_count": 7,             # 最低 7 张

    # 辅图类型覆盖
    "min_scene_count": 2,             # 至少 2 张场景图
    "min_infographic_count": 2,       # 至少 2 张信息图
    "min_detail_count": 2,            # 至少 2 张细节图
}

# 验证重试配置
VALIDATION = {
    "max_retries": 3,                 # 单张图片最大重试次数
    "retry_on_fail_dimensions": [     # 哪些维度失败时需要重试
        "background_score",
        "product_ratio_score",
    ],
}


# ============================================================
# 图片生成规格
# ============================================================

IMAGE_SPECS = {
    # 输出尺寸（Amazon 推荐 2000x2000）
    "output_width": 2000,
    "output_height": 2000,
    "output_format": "JPEG",
    "output_quality": 95,             # JPEG 质量

    # 生成计划：9 张组图的类型分配
    "generation_plan": [
        {"slot": "main",            "type": "main",        "filename": "main.jpg"},
        {"slot": "PT01_lifestyle",  "type": "lifestyle",   "filename": "PT01_lifestyle.jpg"},
        {"slot": "PT02_lifestyle",  "type": "lifestyle",   "filename": "PT02_lifestyle.jpg"},
        {"slot": "PT03_infographic","type": "infographic",  "filename": "PT03_infographic.jpg"},
        {"slot": "PT04_infographic","type": "infographic",  "filename": "PT04_infographic.jpg"},
        {"slot": "PT05_detail",     "type": "detail",      "filename": "PT05_detail.jpg"},
        {"slot": "PT06_detail",     "type": "detail",      "filename": "PT06_detail.jpg"},
        {"slot": "PT07_multiangle", "type": "multiangle",  "filename": "PT07_multiangle.jpg"},
        {"slot": "PT08_multiangle", "type": "multiangle",  "filename": "PT08_multiangle.jpg"},
    ],
}


# ============================================================
# 信息图文字叠加配置
# ============================================================

INFOGRAPHIC_CONFIG = {
    # 文字渲染
    "font_size_title": 60,            # 标题字号
    "font_size_body": 40,             # 正文字号
    "font_size_min": 36,              # 最小字号（手机端可读）
    "font_color_primary": (255, 255, 255),   # 白色文字
    "font_color_dark": (33, 33, 33),         # 深色文字
    "accent_color": (255, 153, 0),           # Amazon 橙色强调色
    "bg_overlay_color": (0, 0, 0),           # 黑色蒙层
    "bg_overlay_opacity": 160,               # 蒙层透明度 (0-255)

    # 布局
    "padding": 60,                    # 边距
    "text_shadow": True,              # 文字阴影
    "text_shadow_offset": (2, 2),     # 阴影偏移
    "text_shadow_color": (0, 0, 0),   # 阴影颜色

    # 字体回退列表（按优先级）
    "font_candidates": [
        "arial.ttf",
        "Arial.ttf",
        "msyh.ttc",          # 微软雅黑
        "simhei.ttf",        # 黑体
        "DejaVuSans.ttf",
    ],
}


# ============================================================
# 费用追踪
# ============================================================

# GPT-4o 图片生成估算价格（美元）
COST_ESTIMATES = {
    "gpt-4o": {
        "text_to_image": 0.04,    # 纯文字生成
        "image_edit": 0.08,       # 带参考图编辑
    },
    "flux": {
        "text_to_image": 0.04,
        "image_edit": 0.06,
    },
    "siliconflow": {
        "text_to_image": 0.00,    # Flux.1-schnell 免费
        "image_edit": 0.00,
    },
}
