"""
Prompt 模板系统
基于 5 个评分维度设计 Prompt，支持动态填充产品信息。

5 种图片类型：
1. 主图 (main) - 纯白背景、产品居中、无文字无阴影、专业摄影质感
2. 场景图 (lifestyle) - 产品在真实使用场景中、自然暖色调、故事感
3. 信息图 (infographic) - AI生成底图 + Pillow叠加文字层
4. 细节图 (detail) - 微距风格、材质纹理、浅景深、品质感
5. 多角度图 (multiangle) - 侧面/背面/俯视、白底、揭示主图看不到的特征
"""

import logging

logger = logging.getLogger(__name__)


# ============================================================
# 基础 Prompt 组件（来自评分维度的要求）
# ============================================================

# 技术指标 → 输出规格指令
TECH_SPEC = (
    "Output a high-resolution product photograph, at least 2000x2000 pixels, "
    "sharp focus throughout, saved as high-quality JPEG."
)

# 主图合规性 → 白底要求
MAIN_COMPLIANCE = (
    "Pure white background RGB(255,255,255). Product centered, occupying 70-85% of the frame. "
    "No text, no logos, no watermarks, no shadows, no props. "
    "Clean, professional e-commerce product photography style."
)

# 设计质量 → 光线/构图/色彩
DESIGN_QUALITY = (
    "Even, soft studio lighting with no harsh shadows. "
    "Centered, balanced composition. "
    "Rich, natural colors with accurate white balance. "
    "Tack-sharp focus on the product."
)

# 营销效果 → 场景/卖点
MARKETING_SCENE = (
    "Show the product in a realistic, aspirational use scenario. "
    "Warm, inviting lighting. Tell a visual story that makes the viewer "
    "imagine owning and using this product."
)

MARKETING_INFO = (
    "Highlight the product's key selling points and unique features. "
    "Professional, clean layout that communicates value clearly."
)


# ============================================================
# 产品信息格式化
# ============================================================

def format_product_description(product_info):
    """
    将产品信息字典格式化为 prompt 可用的描述。

    product_info 示例:
    {
        "name": "Car Trunk Organizer",
        "category": "car accessories",
        "features": ["foldable", "waterproof", "72L capacity"],
        "material": "Oxford fabric",
        "color": "black",
        "selling_points": ["Large capacity", "Easy to install", "Durable material"],
        "target_audience": "car owners",
    }
    """
    parts = []

    name = product_info.get("name", "product")
    parts.append(f"Product: {name}")

    category = product_info.get("category", "")
    if category:
        parts.append(f"Category: {category}")

    features = product_info.get("features", [])
    if features:
        parts.append(f"Key features: {', '.join(features)}")

    material = product_info.get("material", "")
    if material:
        parts.append(f"Material: {material}")

    color = product_info.get("color", "")
    if color:
        parts.append(f"Color: {color}")

    return ". ".join(parts) + "."


# ============================================================
# Prompt 模板
# ============================================================

def build_main_prompt(product_info):
    """
    主图 Prompt - 纯白背景产品照
    用于 image-to-image 编辑（以原图为参考）
    """
    desc = format_product_description(product_info)

    prompt = (
        f"{desc}\n\n"
        f"{MAIN_COMPLIANCE}\n\n"
        f"{DESIGN_QUALITY}\n\n"
        f"{TECH_SPEC}\n\n"
        "Style: Professional Amazon product listing main image. "
        "Photorealistic, studio quality. The product should look premium and trustworthy."
    )
    return prompt


def build_lifestyle_prompt(product_info, scene_variant=1):
    """
    场景图 Prompt - 产品在真实使用场景中
    scene_variant: 1-3, 不同场景变体
    """
    desc = format_product_description(product_info)
    name = product_info.get("name", "the product")
    target = product_info.get("target_audience", "the user")
    category = product_info.get("category", "")

    # 根据产品类别和变体编号选择不同场景
    scene_descriptions = {
        1: (
            f"Show {name} being actively used in its natural environment. "
            f"A person ({target}) is interacting with the product in a real, everyday setting. "
            "Natural daylight, warm tones, lifestyle photography feel. "
            "The scene tells a story of convenience and satisfaction."
        ),
        2: (
            f"Show {name} installed/placed in its intended location. "
            "Clean, well-organized environment. Slightly elevated angle. "
            "Soft natural lighting, modern aesthetic. "
            "The image conveys quality and a premium lifestyle."
        ),
        3: (
            f"Show {name} in an outdoor or travel context. "
            "Dynamic, active scene with appealing background. "
            "Golden hour lighting, vibrant colors. "
            "The image evokes adventure and reliability."
        ),
    }

    scene = scene_descriptions.get(scene_variant, scene_descriptions[1])

    prompt = (
        f"{desc}\n\n"
        f"{scene}\n\n"
        f"{MARKETING_SCENE}\n\n"
        f"{TECH_SPEC}\n\n"
        "Style: High-end lifestyle product photography for Amazon listing. "
        "Photorealistic, editorial quality. NOT a white background studio shot."
    )
    return prompt


def build_infographic_base_prompt(product_info, info_variant=1):
    """
    信息图底图 Prompt（第一步：AI 生成产品展示底图，不含文字）
    文字层将由 Pillow 在第二步叠加
    """
    desc = format_product_description(product_info)
    name = product_info.get("name", "the product")
    features = product_info.get("features", [])

    angle_descriptions = {
        1: (
            f"A single {name} photographed from a 3/4 top-down angle, "
            "centered on a clean solid light gray gradient background. "
            "The product occupies the center 50% of the frame. "
            "Large empty margins on left, right, and top for later use. "
            "Minimalist, clean product photography."
        ),
        2: (
            f"A grid of 6-8 small photographs of {name} from different angles, "
            "arranged neatly on a clean white background. "
            "Each photo in a rounded rectangle frame. "
            "Large empty space at top and bottom of the image. "
            "Clean, organized catalog-style product collage."
        ),
    }

    angle = angle_descriptions.get(info_variant, angle_descriptions[1])

    # 用最强硬的措辞禁止 AI 生成文字
    no_text_rule = (
        "ABSOLUTE RULE: This image must contain ZERO text. "
        "No letters, no words, no numbers, no labels, no logos, no brand names, "
        "no watermarks, no captions, no annotations. "
        "The image is ONLY the physical product and the background. Nothing else. "
        "If you add any text or typography, the image is rejected."
    )

    prompt = (
        f"{no_text_rule}\n\n"
        f"{desc}\n\n"
        f"{angle}\n\n"
        f"{TECH_SPEC}\n\n"
        "Style: Clean product-only photograph for an infographic base layer. "
        "Photorealistic, no graphic design elements, no text of any kind."
    )
    return prompt


def build_infographic_text_plan(product_info, info_variant=1):
    """
    信息图文字层计划（供 Pillow 渲染使用）
    返回要叠加的文字元素列表
    """
    name = product_info.get("name", "Product")
    selling_points = product_info.get("selling_points", [])
    features = product_info.get("features", [])

    # 如果没有 selling_points，用 features 代替
    points = selling_points if selling_points else features

    if info_variant == 1:
        # 布局1：左右两侧标注卖点
        text_elements = [
            {
                "type": "title",
                "text": name,
                "position": "top_center",
                "font_size": "title",
            },
        ]

        # 卖点分列两侧
        for i, point in enumerate(points[:6]):
            side = "left" if i % 2 == 0 else "right"
            text_elements.append({
                "type": "feature",
                "text": point if isinstance(point, str) else str(point),
                "position": f"{side}_{i // 2}",
                "font_size": "body",
                "icon": "check",
            })

    else:
        # 布局2：底部横排特性条
        text_elements = [
            {
                "type": "title",
                "text": name,
                "position": "top_center",
                "font_size": "title",
            },
        ]

        for i, point in enumerate(points[:4]):
            text_elements.append({
                "type": "feature_box",
                "text": point if isinstance(point, str) else str(point),
                "position": f"bottom_{i}",
                "font_size": "body",
                "style": "box",
            })

    return text_elements


def build_detail_prompt(product_info, detail_variant=1):
    """
    细节图 Prompt - 微距风格、材质纹理
    """
    desc = format_product_description(product_info)
    name = product_info.get("name", "the product")
    material = product_info.get("material", "the material")
    features = product_info.get("features", [])

    detail_descriptions = {
        1: (
            f"Extreme close-up macro photograph of {name}, focusing on the {material} texture "
            "and build quality. Shallow depth of field (f/2.8), the focused area is tack-sharp "
            "while the background gently blurs. Show stitching, material grain, or surface finish details."
        ),
        2: (
            f"Close-up photograph of {name}'s key functional detail or mechanism. "
            "Show how a specific feature works — buckle, zipper, hinge, connector, or interface. "
            "Clean, well-lit, with subtle background."
        ),
        3: (
            f"Close-up showing the craftsmanship and premium quality of {name}. "
            "Highlight the finish, color accuracy, and material quality. "
            "Professional macro photography with ring light reflection."
        ),
    }

    detail = detail_descriptions.get(detail_variant, detail_descriptions[1])

    prompt = (
        f"{desc}\n\n"
        f"{detail}\n\n"
        f"{TECH_SPEC}\n\n"
        "Style: Macro product detail photography. "
        "The close-up should communicate premium quality and attention to detail. "
        "Shallow depth of field, crisp focus on the key area."
    )
    return prompt


def build_multiangle_prompt(product_info, angle_variant=1):
    """
    多角度图 Prompt - 侧面/背面/俯视
    """
    desc = format_product_description(product_info)
    name = product_info.get("name", "the product")

    angle_descriptions = {
        1: (
            f"Side view of {name} on a pure white background. "
            "Show the product's profile, thickness, and side details that are not visible "
            "from the front. Clean, studio-lit, same visual style as the main image."
        ),
        2: (
            f"Back/rear view of {name} on a pure white background. "
            "Reveal the back design, labels, ports, or construction details. "
            "Professional studio lighting, consistent with the main image style."
        ),
        3: (
            f"Top-down/bird's-eye view of {name} on a pure white background. "
            "Show the product from directly above, revealing its footprint, layout, "
            "and top surface details."
        ),
    }

    angle = angle_descriptions.get(angle_variant, angle_descriptions[1])

    prompt = (
        f"{desc}\n\n"
        f"{angle}\n\n"
        f"{MAIN_COMPLIANCE}\n\n"
        f"{TECH_SPEC}\n\n"
        "Style: Multi-angle product photography for Amazon listing. "
        "White background, studio quality, consistent with the main product image."
    )
    return prompt


# ============================================================
# 统一接口：生成完整的 prompt 计划
# ============================================================

def generate_prompt_plan(product_info, generation_plan=None):
    """
    根据 generation_plan 生成所有图片的 prompt。

    参数:
        product_info: 产品信息字典
        generation_plan: 图片生成计划列表（来自 config.IMAGE_SPECS）

    返回:
        list[dict]: 每个元素包含 slot, type, filename, prompt, text_plan(仅信息图)
    """
    from config import IMAGE_SPECS

    if generation_plan is None:
        generation_plan = IMAGE_SPECS["generation_plan"]

    prompt_plan = []

    # 各类型的变体计数器
    variant_counters = {
        "lifestyle": 0,
        "infographic": 0,
        "detail": 0,
        "multiangle": 0,
    }

    for item in generation_plan:
        img_type = item["type"]
        slot = item["slot"]
        filename = item["filename"]

        entry = {
            "slot": slot,
            "type": img_type,
            "filename": filename,
        }

        if img_type == "main":
            entry["prompt"] = build_main_prompt(product_info)

        elif img_type == "lifestyle":
            variant_counters["lifestyle"] += 1
            variant = variant_counters["lifestyle"]
            entry["prompt"] = build_lifestyle_prompt(product_info, variant)

        elif img_type == "infographic":
            variant_counters["infographic"] += 1
            variant = variant_counters["infographic"]
            entry["prompt"] = build_infographic_base_prompt(product_info, variant)
            entry["text_plan"] = build_infographic_text_plan(product_info, variant)

        elif img_type == "detail":
            variant_counters["detail"] += 1
            variant = variant_counters["detail"]
            entry["prompt"] = build_detail_prompt(product_info, variant)

        elif img_type == "multiangle":
            variant_counters["multiangle"] += 1
            variant = variant_counters["multiangle"]
            entry["prompt"] = build_multiangle_prompt(product_info, variant)

        prompt_plan.append(entry)

    logger.info(f"生成 {len(prompt_plan)} 张图片的 prompt 计划:")
    type_counts = {}
    for p in prompt_plan:
        t = p["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in type_counts.items():
        logger.info(f"  {t}: {c}张")

    return prompt_plan


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 测试用产品信息
    test_product = {
        "name": "Car Trunk Organizer",
        "category": "car accessories",
        "features": ["foldable", "waterproof", "72L capacity", "multi-compartment"],
        "material": "600D Oxford fabric",
        "color": "black",
        "selling_points": [
            "72L Large Capacity",
            "Waterproof & Durable",
            "Foldable Design",
            "Multi-Compartment Storage",
            "Easy Installation",
            "Premium 600D Oxford Fabric",
        ],
        "target_audience": "car owners who need organized trunk space",
    }

    plan = generate_prompt_plan(test_product)

    print("\n" + "=" * 60)
    for i, p in enumerate(plan):
        print(f"\n--- [{p['slot']}] {p['type']} ---")
        print(p["prompt"][:200] + "...")
        if "text_plan" in p:
            print(f"  文字元素: {len(p['text_plan'])} 个")
