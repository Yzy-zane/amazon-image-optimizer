"""
Prompt 模板系统 v2
基于 Amazon 平台合规标准、商家效率需求、买家心理预期三维度优化。

v1 → v2 改进点:
- 主图填充比 70-85% → 85-95% (Amazon 搜索权重要求)
- 主图增加禁止包装/车辆部件指令
- 场景图增加车型感知 (SUV/Sedan/Truck)，防止空间扭曲
- 全部 prompt 增加负向指令 (AVOID)，防止常见 AI 生成错误
- 细节图增加更具体的汽车饰品材质描述
- 信息图文字计划增加尺寸参考维度

5 种图片类型:
1. 主图 (main) - 纯白背景、产品占 85-95%、无文字无包装
2. 场景图 (lifestyle) - 车内/使用场景、空间比例准确、温暖氛围
3. 信息图 (infographic) - AI 底图 + Pillow 叠加文字 + 尺寸参考
4. 细节图 (detail) - 微距材质、缝线工艺、功能部件特写
5. 多角度图 (multiangle) - 补充视角、白底、一致视觉风格
"""

import logging

logger = logging.getLogger(__name__)


# ============================================================
# Prompt 组件 v2
# ============================================================

# 技术规格 (精简，每条 prompt 结尾统一附加)
TECH_SPEC = (
    "Output: 2000x2000 pixel photograph, photorealistic, studio-grade, "
    "tack-sharp focus, high-quality JPEG."
)

# 主图合规 (Amazon 2026 标准)
MAIN_COMPLIANCE = (
    "Pure white background RGB(255,255,255) — absolutely no gray, no gradient, "
    "no shadow cast on the background. "
    "Product centered, filling 85-95% of the frame with minimal white border. "
    "PROHIBITED: text, logos, watermarks, shadows on background, props, "
    "packaging, boxes, tags, price labels, hangers. "
    "Product only — do NOT include vehicle parts, car seats, steering wheels, "
    "or any non-product elements as background. "
    "Amazon e-commerce main image standard."
)

# 设计质量 (专业摄影术语)
DESIGN_QUALITY = (
    "Professional three-point studio lighting: "
    "key light at 45 degrees for gentle highlight modeling, "
    "fill light opposite to eliminate harsh shadows, "
    "rim light from behind for crisp edge separation from the white background. "
    "Centered, symmetrical composition. "
    "Vivid, true-to-life colors with precise white balance. "
    "Tack-sharp focus across the entire product surface."
)

# 场景营销 (汽车饰品专用)
MARKETING_SCENE = (
    "Realistic, aspirational automotive lifestyle photograph. "
    "Warm, natural lighting — soft daylight or golden hour, never harsh or artificial. "
    "The vehicle interior must have physically correct, realistic proportions and geometry. "
    "The viewer should immediately imagine this product in their own vehicle."
)

# 信息图营销
MARKETING_INFO = (
    "Highlight the product's key selling points and unique value proposition. "
    "Professional, clean layout that communicates quality and trust."
)

# 负向指令 — 嵌入每条 prompt，防止 AI 生成常见错误
NEGATIVE_GENERAL = (
    "AVOID: blurry areas, soft focus, artificial-looking plastic textures, "
    "distorted proportions, any text or letters or numbers, watermarks, logos, "
    "unnatural color casts, low-resolution artifacts, over-saturation."
)

NEGATIVE_CAR_SCENE = (
    "AVOID: distorted car interior geometry, impossible trunk or cabin dimensions, "
    "warped seats or dashboard, incorrect vehicle scale relative to product, "
    "unrealistic lighting inside vehicle, product floating or poorly placed in space, "
    "generic stock-photo feel."
)


# ============================================================
# 产品信息工具函数
# ============================================================

def format_product_description(product_info):
    """将产品信息字典格式化为 prompt 描述段。"""
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


def _detect_product_context(product_info):
    """
    判断产品使用场景类型，决定 lifestyle prompt 方向。

    返回:
        "car_interior"  - 车内配件 (收纳箱、座套、手机支架等) → 放进车里展示
        "car_exterior"  - 车外配件 (车贴、天线、雨刮等) → 装在车外展示
        "display"       - 展示型产品 (车模、模型、收藏品等) → 桌面/展柜展示
        "general"       - 通用产品 → 通用生活场景
    """
    text = " ".join([
        product_info.get("name", ""),
        product_info.get("category", ""),
        " ".join(product_info.get("features", [])),
        product_info.get("target_audience", ""),
    ]).lower()

    # 展示型/收藏型产品 (优先检测，因为 "car model" 也包含 "car")
    display_keywords = [
        "model", "模型", "die-cast", "diecast", "toy", "replica",
        "collectible", "figurine", "scale", "1:18", "1:24", "1:43",
        "decoration", "decor", "ornament", "display", "statue",
    ]
    if any(kw in text for kw in display_keywords):
        return "display"

    # 车内配件
    interior_keywords = [
        "organizer", "seat cover", "seat cushion", "phone holder", "mount",
        "floor mat", "trunk", "storage", "console", "visor", "sunshade",
        "charger", "dash cam", "air freshener", "steering wheel",
        "armrest", "headrest", "cup holder", "trash can", "tissue",
        "收纳", "座套", "坐垫", "脚垫", "遮阳",
    ]
    if any(kw in text for kw in interior_keywords):
        return "car_interior"

    # 车外配件
    exterior_keywords = [
        "decal", "sticker", "antenna", "wiper", "mirror", "light",
        "bumper", "grille", "fender", "spoiler", "roof rack", "mud flap",
        "车贴", "雨刮", "后视镜",
    ]
    if any(kw in text for kw in exterior_keywords):
        return "car_exterior"

    return "general"


def _detect_vehicle_type(product_info):
    """从产品信息推断车型 (仅当产品确实是车内/车外配件时有意义)。"""
    text = " ".join([
        product_info.get("name", ""),
        product_info.get("category", ""),
        " ".join(product_info.get("features", [])),
        product_info.get("target_audience", ""),
    ]).lower()

    if "suv" in text:
        return "SUV"
    elif "truck" in text or "pickup" in text:
        return "truck"
    elif "sedan" in text:
        return "sedan"
    return "car"


def _extract_size_info(product_info):
    """从产品特点/卖点中提取尺寸相关信息，用于信息图尺寸参考。"""
    all_items = (
        product_info.get("features", [])
        + product_info.get("selling_points", [])
    )
    size_keywords = [
        "capacity", "liter", "gallon", "inch", "cm", "mm",
        "size", "fit", "hold", "large", "big", "L ",
    ]
    for item in all_items:
        item_str = item if isinstance(item, str) else str(item)
        if any(kw.lower() in item_str.lower() for kw in size_keywords):
            return item_str
    return None


# ============================================================
# Prompt 模板 v2
# ============================================================

def build_main_prompt(product_info):
    """
    主图 — 纯白背景产品照 (Amazon 合规)

    v2 改进:
    - 填充比 85-95%（Amazon 搜索权重）
    - 禁止包装、标签、车辆部件
    - 增加负向指令
    """
    desc = format_product_description(product_info)

    prompt = (
        f"{desc}\n\n"
        f"{MAIN_COMPLIANCE}\n\n"
        f"{DESIGN_QUALITY}\n\n"
        f"{TECH_SPEC}\n\n"
        f"{NEGATIVE_GENERAL}\n\n"
        "Style: Premium Amazon product listing main image. "
        "The product must look high-quality, trustworthy, and worth purchasing. "
        "Photorealistic studio photography, not a 3D render."
    )
    return prompt


def build_lifestyle_prompt(product_info, scene_variant=1):
    """
    场景图 — 产品在真实使用/展示场景

    v3 修复:
    - 先判断产品类型 (车内配件/展示品/通用)，再匹配对应场景
    - 车模/模型 → 桌面/展柜展示场景，不会被塞进后备箱
    - 车内配件 → 车内安装/使用场景
    - 通用产品 → 通用生活场景
    """
    desc = format_product_description(product_info)
    name = product_info.get("name", "the product")
    target = product_info.get("target_audience", "the customer")
    context = _detect_product_context(product_info)

    if context == "display":
        scene, style_note, spatial_note = _build_display_scene(name, target, scene_variant)
    elif context == "car_interior":
        scene, style_note, spatial_note = _build_car_interior_scene(name, target, product_info, scene_variant)
    elif context == "car_exterior":
        scene, style_note, spatial_note = _build_car_exterior_scene(name, target, scene_variant)
    else:
        scene, style_note, spatial_note = _build_general_scene(name, target, product_info, scene_variant)

    prompt = (
        f"{desc}\n\n"
        f"SCENE: {scene}\n\n"
        f"{spatial_note}\n\n"
        f"{TECH_SPEC}\n\n"
        f"{NEGATIVE_GENERAL}\n\n"
        f"Style: {style_note}"
    )
    return prompt


def _build_display_scene(name, target, variant):
    """展示型产品场景 (车模、模型、收藏品、装饰品)"""
    scenes = {
        1: (
            f"A {name} displayed on a clean, dark wooden desk or shelf. "
            "Soft warm side lighting from the left creates gentle highlights on the product surface. "
            "Background: slightly blurred modern living room or study, bookshelves and warm decor visible. "
            f"A person's hand ({target}) is reaching towards the product as if admiring it. "
            "The product is the clear hero — it looks premium and collectible. "
            "The scene evokes pride of ownership and aesthetic appreciation."
        ),
        2: (
            f"A {name} placed on a glass display shelf or clean white surface. "
            "Professional display lighting — a small spotlight from above creates a focused glow. "
            "Background: a softly blurred collection or showroom setting. "
            "The product catches the light beautifully, showing its details and finish. "
            "The image conveys exclusivity, craftsmanship, and collector's value."
        ),
    }
    scene = scenes.get(variant, scenes[1])
    style = (
        "High-end product lifestyle photography. Photorealistic, editorial quality. "
        "The product is a collectible display piece shown on furniture or a display shelf. "
        "This is a showcase scene in a home or showroom setting."
    )
    spatial = (
        "IMPORTANT: This product is a display/collectible item. "
        "Show it on a desk, shelf, or display surface in a home or showroom. "
        "The product should be the focal point, beautifully lit and clearly visible."
    )
    return scene, style, spatial


def _build_car_interior_scene(name, target, product_info, variant):
    """车内配件场景 (收纳箱、座套、手机支架等)"""
    vehicle = _detect_vehicle_type(product_info)

    suv_scenes = {
        1: (
            f"{name} placed inside the spacious trunk of a modern SUV (like Honda CR-V or Toyota RAV4). "
            f"A person ({target}) is reaching into the organized trunk, loading shopping bags. "
            "Camera shoots from behind the open tailgate at a slight downward angle. "
            "The SUV is parked in a suburban driveway, soft afternoon daylight. "
            "The trunk space is clean and well-organized, the product fits naturally. "
            "The scene conveys everyday convenience and an organized lifestyle."
        ),
        2: (
            f"{name} neatly installed in the rear cargo area of a premium SUV. "
            "The tailgate is open, showing the product perfectly fitted in the trunk space. "
            "Background: a tree-lined suburban street visible through the rear window. "
            "Late afternoon golden light streams in from behind. "
            "The image suggests weekend family trips and outdoor adventures."
        ),
    }

    truck_scenes = {
        1: (
            f"{name} in the rear seat area or extended cab storage of a pickup truck. "
            f"A person ({target}) is accessing the product from the passenger side. "
            "Outdoor setting — a clean worksite or trailhead parking area. "
            "Warm golden hour lighting with long shadows. "
            "The scene conveys rugged reliability and smart organization."
        ),
        2: (
            f"{name} installed in a truck's rear seat organizer space. "
            "Work gear and outdoor equipment neatly arranged alongside the product. "
            "Early morning light with a slight cool-to-warm gradient. "
            "The image communicates utility, durability, and readiness for action."
        ),
    }

    sedan_scenes = {
        1: (
            f"{name} placed inside the trunk of a modern sedan (like Honda Accord or Toyota Camry). "
            f"A person ({target}) is leaning in to retrieve something from the product. "
            "Clean underground parking or home garage setting. "
            "Soft overhead lighting mixed with natural light from the entrance. "
            "The trunk is organized and tidy. "
            "The scene tells a story of daily convenience and smart organization."
        ),
        2: (
            f"{name} neatly set up in a car's rear trunk compartment. "
            "The trunk lid is open, showing the product fitting perfectly in the available space. "
            "Bright daylight exterior visible behind the car. "
            "Shot from a 3/4 rear angle so both the car and the product interior are visible. "
            "The image conveys a tidy, quality-conscious lifestyle."
        ),
    }

    scene_map = {"SUV": suv_scenes, "truck": truck_scenes, "sedan": sedan_scenes, "car": sedan_scenes}
    scenes = scene_map.get(vehicle, sedan_scenes)
    scene = scenes.get(variant, scenes[1])

    style = (
        "High-end automotive lifestyle photography for Amazon listing. "
        "Photorealistic, editorial quality. This is NOT a white background studio shot — "
        "it is an in-context vehicle interior lifestyle image."
    )
    spatial = (
        "SPATIAL ACCURACY RULE: The vehicle interior dimensions must be physically correct. "
        "The product must fit naturally in the space — no impossible sizing, no distorted geometry. "
        f"Vehicle type: {vehicle}.\n"
        f"{NEGATIVE_CAR_SCENE}"
    )
    return scene, style, spatial


def _build_car_exterior_scene(name, target, variant):
    """车外配件场景 (车贴、天线、灯饰等)"""
    scenes = {
        1: (
            f"A {name} installed on a clean, modern car in a well-lit parking area. "
            f"A person ({target}) is standing nearby, admiring the installed product. "
            "The car is clean and polished, shot from a 3/4 angle that shows the product clearly. "
            "Soft afternoon daylight, the product is the clear focal point on the vehicle."
        ),
        2: (
            f"Close-up of a {name} mounted on a vehicle exterior. "
            "The car is parked in a scenic location — clean urban street or suburban driveway. "
            "Golden hour lighting creates appealing highlights on both the car body and the product. "
            "The image shows how the product enhances the vehicle's appearance."
        ),
    }
    scene = scenes.get(variant, scenes[1])
    style = (
        "Automotive product photography showing the product installed on a vehicle. "
        "Photorealistic, lifestyle quality. The product should look like a natural upgrade."
    )
    spatial = (
        "SPATIAL ACCURACY: The vehicle proportions must be realistic. "
        "The product must be correctly sized relative to the car.\n"
        f"{NEGATIVE_CAR_SCENE}"
    )
    return scene, style, spatial


def _build_general_scene(name, target, product_info, variant):
    """通用产品场景"""
    scenes = {
        1: (
            f"{name} being used in its natural environment by {target}. "
            "Real, everyday setting with warm natural daylight. "
            "The person is interacting with the product naturally — "
            "the scene tells a story of convenience and satisfaction. "
            "Clean, uncluttered background with soft bokeh."
        ),
        2: (
            f"{name} placed in a clean, well-lit environment that matches its purpose. "
            "Slightly elevated camera angle. Soft natural lighting, modern aesthetic. "
            "The product is the clear hero of the image. "
            "The scene conveys quality and a premium lifestyle."
        ),
    }
    scene = scenes.get(variant, scenes[1])
    style = (
        "High-end lifestyle product photography for Amazon listing. "
        "Photorealistic, editorial quality. NOT a white background studio shot."
    )
    spatial = (
        "The product must be shown at correct, realistic proportions. "
        "The scene should feel natural and believable."
    )
    return scene, style, spatial


def build_infographic_base_prompt(product_info, info_variant=1):
    """
    信息图底图 — AI 生成无文字底图，后续 Pillow 叠加文字

    v2 改进:
    - 更强的禁止文字指令
    - 改进构图描述，留出更清晰的文字叠加空间
    - 增加负向指令
    """
    desc = format_product_description(product_info)
    name = product_info.get("name", "the product")

    no_text_rule = (
        "ABSOLUTE RULE — ZERO TEXT IN THIS IMAGE: "
        "No letters, no words, no numbers, no labels, no logos, no brand names, "
        "no watermarks, no captions, no annotations, no symbols that resemble text. "
        "The image contains ONLY the physical product and a clean background. "
        "Any text or typography of any kind = automatic rejection."
    )

    if info_variant == 1:
        # Layout 1: 产品居中，左右留白给卖点文字
        composition = (
            f"A single {name} photographed from a 3/4 elevated angle (about 30 degrees from above). "
            "Product centered on a clean, smooth light-gray to white gradient background. "
            "The product occupies ONLY the center 40-50% of the frame. "
            "CRITICAL: Leave large, clean, empty margins on the left side (25%), "
            "right side (25%), and top (15%) of the image — these spaces will be used "
            "for text overlay in post-production. "
            "Even, soft shadowless lighting. Minimalist, premium product photography."
        )
    else:
        # Layout 2: 产品居上，底部留白给特性条
        composition = (
            f"A feature-highlight view of {name}. "
            "Product positioned in the upper 60% of the frame on a clean white background. "
            "If the product has multiple compartments or parts, show them slightly separated "
            "to reveal internal structure and craftsmanship. "
            "CRITICAL: Leave the bottom 30% of the image completely empty and white — "
            "this space will be used for feature labels in post-production. "
            "Also leave the top 10% empty for a title. "
            "Clean, organized, catalog-style product visualization."
        )

    prompt = (
        f"{no_text_rule}\n\n"
        f"{desc}\n\n"
        f"COMPOSITION: {composition}\n\n"
        f"{TECH_SPEC}\n\n"
        f"{NEGATIVE_GENERAL}\n\n"
        "Style: Clean product photograph optimized as an infographic base layer. "
        "Photorealistic, no graphic design elements, absolutely no text of any kind."
    )
    return prompt


def build_infographic_text_plan(product_info, info_variant=1):
    """
    信息图文字叠加计划 (供 Pillow 渲染)

    v2 改进:
    - 增加尺寸参考维度 (如有容量/尺寸信息)
    """
    name = product_info.get("name", "Product")
    selling_points = product_info.get("selling_points", [])
    features = product_info.get("features", [])

    # 优先用 selling_points，没有就用 features
    points = selling_points if selling_points else features
    size_info = _extract_size_info(product_info)

    if info_variant == 1:
        # Layout 1: 标题顶部 + 卖点左右分列
        text_elements = [
            {
                "type": "title",
                "text": name,
                "position": "top_center",
                "font_size": "title",
            },
        ]

        # 卖点分列两侧（最多 6 个）
        display_points = list(points[:6])

        # 如果有尺寸信息且不在卖点中，替换最后一个
        if size_info and not any(size_info.lower() in p.lower() for p in display_points):
            if len(display_points) >= 6:
                display_points[5] = f"Size: {size_info}"
            else:
                display_points.append(f"Size: {size_info}")

        for i, point in enumerate(display_points):
            side = "left" if i % 2 == 0 else "right"
            text_elements.append({
                "type": "feature",
                "text": point if isinstance(point, str) else str(point),
                "position": f"{side}_{i // 2}",
                "font_size": "body",
                "icon": "check",
            })

    else:
        # Layout 2: 标题顶部 + 底部特性框
        text_elements = [
            {
                "type": "title",
                "text": name,
                "position": "top_center",
                "font_size": "title",
            },
        ]

        display_points = list(points[:4])

        for i, point in enumerate(display_points):
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
    细节图 — 微距/材质/功能特写

    v2 改进:
    - 更具体的汽车饰品材质描述
    - 更专业的微距摄影术语
    - 增加负向指令
    """
    desc = format_product_description(product_info)
    name = product_info.get("name", "the product")
    material = product_info.get("material", "the material")

    details = {
        1: (
            # 材质微距 — 展示做工品质
            f"Extreme close-up macro photograph of {name}. "
            f"Focus on the {material} texture, stitching precision, and surface finish quality. "
            "Camera settings: f/2.8 aperture for shallow depth of field — "
            "the focal area is razor-sharp while the background dissolves into creamy bokeh. "
            "Show the weave of the fabric, the straightness and evenness of the seams, "
            "the quality of edge binding or reinforcement stitching. "
            "Lighting: soft diffused ring light from above, creating subtle texture shadows "
            "that reveal material depth and quality. "
            "The close-up should make the viewer feel they can almost touch the material."
        ),
        2: (
            # 功能特写 — 展示关键机制
            f"Close-up photograph of {name}'s most important functional mechanism. "
            "Show one specific feature in action: a buckle being clasped, "
            "a zipper being pulled open, a velcro strap being attached, "
            "a handle being gripped, or a compartment divider being adjusted. "
            "The mechanism should be captured mid-action to show how it works. "
            "Clean, slightly blurred neutral background (light gray or white). "
            "Professional product detail lighting — bright, even, no harsh shadows. "
            "The image should answer the buyer's question: 'How does this part work?'"
        ),
    }

    detail = details.get(detail_variant, details[1])

    prompt = (
        f"{desc}\n\n"
        f"SHOT: {detail}\n\n"
        f"{TECH_SPEC}\n\n"
        f"{NEGATIVE_GENERAL}\n\n"
        "Style: Professional macro product photography for e-commerce. "
        "Communicate premium build quality and craftsmanship through close-up detail. "
        "Shallow depth of field, precise focus, inviting tactile quality. "
        "The viewer should feel confident about the product's build quality."
    )
    return prompt


def build_multiangle_prompt(product_info, angle_variant=1):
    """
    多角度图 — 补充主图看不到的角度

    v2 改进:
    - 只保留 1 张（45度俯视角，信息量最大）
    - 强调与主图视觉一致性
    - 增加负向指令
    """
    desc = format_product_description(product_info)
    name = product_info.get("name", "the product")

    angles = {
        1: (
            f"45-degree elevated angle view of {name} on a pure white background. "
            "Show the product's top surface, depth, and full three-dimensional form. "
            "This angle reveals features not visible in the straight-on main image: "
            "the top opening or surface, internal compartments (if visible from above), "
            "overall depth and proportions. "
            "Same professional studio lighting as the main image — consistent visual style. "
            "The viewer should feel they are looking down at the product on a table."
        ),
    }

    angle = angles.get(angle_variant, angles[1])

    prompt = (
        f"{desc}\n\n"
        f"ANGLE: {angle}\n\n"
        f"{MAIN_COMPLIANCE}\n\n"
        f"{TECH_SPEC}\n\n"
        f"{NEGATIVE_GENERAL}\n\n"
        "Style: Multi-angle product photography for Amazon listing. "
        "Pure white background, studio quality, visually consistent with the main product image."
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
        "target_audience": "car and SUV owners who need organized trunk space",
    }

    plan = generate_prompt_plan(test_product)

    print("\n" + "=" * 60)
    for i, p in enumerate(plan, 1):
        print(f"\n--- [{i}/{len(plan)}] {p['slot']} ({p['type']}) ---")
        print(f"Prompt ({len(p['prompt'])} chars):")
        print(p["prompt"])
        if "text_plan" in p:
            print(f"\nText overlay ({len(p['text_plan'])} elements):")
            for t in p["text_plan"]:
                print(f"  [{t['type']}] \"{t['text']}\" @ {t['position']}")
        print()
