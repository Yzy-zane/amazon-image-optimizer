"""
Amazon 商品图片优化 Demo - 流水线入口

用法:
    # 基本用法
    python optimizer_main.py --photo "my_product.jpg" --product-info "product_info.json"

    # 指定模型
    python optimizer_main.py --photo "my_product.jpg" --product-info "product_info.json" --model gpt-4o

    # 仅生成 prompt 不调用 API（调试用）
    python optimizer_main.py --photo "my_product.jpg" --product-info "product_info.json" --dry-run

流水线：选参考 → 组装 Prompt → 调用 AI → 验证评分 → [不达标则重试] → 后处理 → 输出
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimizer.log"),
            encoding="utf-8"
        ),
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_product_info(product_info_path):
    """
    加载产品信息。
    支持 JSON 文件或直接返回默认示例。
    """
    if product_info_path and os.path.exists(product_info_path):
        with open(product_info_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if product_info_path:
        logger.warning(f"产品信息文件不存在: {product_info_path}")

    return None


def interactive_input():
    """
    交互式输入产品信息。
    逐步引导用户填写，回车跳过可选项。
    """
    print("\n" + "=" * 50)
    print("  商品信息录入（交互模式）")
    print("=" * 50)
    print("提示：带 * 的为必填项，其他回车可跳过\n")

    # 1. 商品名称（必填）
    while True:
        name = input("* 商品英文名称 (如 Car Trunk Organizer): ").strip()
        if name:
            break
        print("  商品名称不能为空，请重新输入")

    # 2. 品类
    category = input("  商品品类 (如 car accessories): ").strip()
    if not category:
        category = "general product"

    # 3. 颜色
    color = input("  颜色 (如 black/blue/red): ").strip()

    # 4. 材质
    material = input("  材质 (如 600D Oxford fabric): ").strip()

    # 5. 产品特点（多个，逗号分隔）
    print("\n  产品特点（用逗号分隔多个特点）")
    print("  示例: foldable, waterproof, 72L capacity, lightweight")
    features_raw = input("  特点: ").strip()
    features = [f.strip() for f in features_raw.split(",") if f.strip()] if features_raw else []

    # 6. 卖点（最重要 — 会显示在信息图上）
    print("\n  核心卖点（用逗号分隔，会显示在信息图上，建议3-6个）")
    print("  示例: 72L Large Capacity, Waterproof & Durable, Foldable Design")
    selling_raw = input("* 卖点: ").strip()

    if selling_raw:
        selling_points = [s.strip() for s in selling_raw.split(",") if s.strip()]
    else:
        # 如果没填卖点，用 features 代替
        selling_points = features[:6] if features else [name]

    # 7. 目标用户
    target = input("  目标用户 (如 car owners): ").strip()
    if not target:
        target = f"people who need {name}"

    # 组装结果
    product_info = {"name": name, "category": category}
    if color:
        product_info["color"] = color
    if material:
        product_info["material"] = material
    if features:
        product_info["features"] = features
    product_info["selling_points"] = selling_points
    product_info["target_audience"] = target

    # 确认
    print("\n" + "-" * 50)
    print("  录入信息确认：")
    print(f"  商品名称: {name}")
    print(f"  品类: {category}")
    if color:
        print(f"  颜色: {color}")
    if material:
        print(f"  材质: {material}")
    if features:
        print(f"  特点: {', '.join(features)}")
    print(f"  卖点: {', '.join(selling_points)}")
    print(f"  目标用户: {target}")
    print("-" * 50)

    confirm = input("\n确认以上信息？(回车确认 / n重新输入): ").strip().lower()
    if confirm == "n":
        return interactive_input()  # 重新输入

    # 保存到文件
    save_path = os.path.join(BASE_DIR, "product_info.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(product_info, f, ensure_ascii=False, indent=4)
    print(f"\n已保存到: {save_path}")

    return product_info


def interactive_photo_input():
    """交互式输入商品原图路径"""
    print("\n请输入商品原图路径（支持拖拽文件到终端）")
    while True:
        photo = input("* 图片路径: ").strip().strip('"').strip("'")
        if not photo:
            print("  图片路径不能为空")
            continue
        if os.path.exists(photo):
            return photo
        print(f"  文件不存在: {photo}，请重新输入")


def run_pipeline(photo_path, product_info_path=None, model=None,
                 dry_run=False, output_name=None, max_retries=3):
    """
    执行完整流水线。

    参数:
        photo_path: 商品原图路径
        product_info_path: 产品信息 JSON 文件路径
        model: AI 模型名
        dry_run: 仅输出 prompt 不调用 API
        output_name: 输出目录名（默认用产品名）
        max_retries: 不达标时最大重试次数
    """
    start_time = time.time()

    logger.info("*" * 60)
    logger.info("Amazon 商品图片优化 Demo")
    logger.info("*" * 60)

    # ---- Step 0: 验证输入 ----
    if not dry_run:
        if not photo_path or not os.path.exists(photo_path):
            logger.error(f"原图不存在: {photo_path}")
            logger.info("提示: 使用 --dry-run 可以跳过实际生成，仅查看 prompt")
            return None

    product_info = load_product_info(product_info_path)
    product_name = output_name or product_info.get("name", "product").replace(" ", "_")

    from config import OUTPUT_DIR
    output_dir = os.path.join(OUTPUT_DIR, product_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"商品原图: {photo_path}")
    logger.info(f"产品名称: {product_info.get('name', 'N/A')}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"AI 模型: {model or '配置默认'}")
    logger.info(f"Dry Run: {dry_run}")

    # ---- Step 1: 选参考 ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: 选择参考商品")
    logger.info("=" * 60)

    from reference_selector import get_reference_summary
    ref_summary = get_reference_summary()

    if ref_summary["references"]:
        logger.info(f"选出 {len(ref_summary['references'])} 个参考商品")
        logger.info(f"理想组图构成: {ref_summary['composition']}")
    else:
        logger.info("无参考数据，使用默认组图构成")

    # ---- Step 2: 组装 Prompt ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: 生成 Prompt 计划")
    logger.info("=" * 60)

    from prompt_engine import generate_prompt_plan
    prompt_plan = generate_prompt_plan(product_info)

    if dry_run:
        _print_dry_run(prompt_plan, output_dir, product_info)
        return {"dry_run": True, "prompt_plan": prompt_plan}

    # ---- Step 3: 调用 AI 生成 ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: AI 图片生成")
    logger.info("=" * 60)

    from image_generator import generate_product_images
    gen_result = generate_product_images(
        prompt_plan=prompt_plan,
        output_dir=output_dir,
        source_photo=photo_path,
        model=model,
        dry_run=False,
    )

    if not gen_result["generated"]:
        logger.error("没有成功生成任何图片!")
        return gen_result

    # ---- Step 4: 验证评分 ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: 质量验证")
    logger.info("=" * 60)

    from image_validator import validate_and_fix
    val_result = validate_and_fix(output_dir, product_name)

    # ---- Step 4b: 不达标则重试 ----
    retry_count = 0
    while not val_result["passed"] and retry_count < max_retries:
        retry_count += 1
        logger.info(f"\n--- 重试 {retry_count}/{max_retries} ---")

        # 分析失败原因，决定重新生成哪些图
        items_to_retry = _decide_retry_items(prompt_plan, val_result)

        if not items_to_retry:
            logger.info("没有可重试的项目，接受当前结果")
            break

        logger.info(f"重新生成 {len(items_to_retry)} 张图片...")
        retry_result = generate_product_images(
            prompt_plan=items_to_retry,
            output_dir=output_dir,
            source_photo=photo_path,
            model=model,
        )

        # 重新验证
        val_result = validate_and_fix(output_dir, product_name)

    # ---- Step 5: 后处理 + 输出 ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: 后处理与输出")
    logger.info("=" * 60)

    _post_process(output_dir)

    # 保存生成日志
    elapsed = time.time() - start_time
    generation_log = {
        "product_name": product_info.get("name", ""),
        "product_info": product_info,
        "model": model or "default",
        "source_photo": os.path.basename(photo_path) if photo_path else None,
        "generation_result": gen_result,
        "validation_result": {
            "passed": val_result["passed"],
            "scores": val_result["scores"],
            "failures": val_result["failures"],
        },
        "retry_count": retry_count,
        "elapsed_seconds": round(elapsed, 1),
        "cost_usd": gen_result["stats"]["total_cost_usd"],
    }

    log_path = os.path.join(output_dir, "generation_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(generation_log, f, ensure_ascii=False, indent=2)

    # ---- 最终总结 ----
    logger.info("\n" + "*" * 60)
    logger.info("完成!")
    logger.info("*" * 60)

    logger.info(f"输出目录: {output_dir}")
    _list_output_files(output_dir)

    logger.info(f"\n验证结果: {'通过' if val_result['passed'] else '未通过'}")
    logger.info(f"整体评分: {val_result['scores'].get('normalized_score', 'N/A')}/100")
    logger.info(f"总耗时: {elapsed:.1f}s")
    logger.info(f"API 费用: ${gen_result['stats']['total_cost_usd']:.2f}")
    logger.info(f"生成日志: {log_path}")

    return generation_log


def _print_dry_run(prompt_plan, output_dir, product_info):
    """Dry Run 模式：打印所有 prompt"""
    logger.info("\n[DRY RUN MODE] 以下是将要使用的 Prompt:")

    for i, item in enumerate(prompt_plan, 1):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[{i}/{len(prompt_plan)}] {item['slot']} ({item['type']})")
        logger.info(f"输出文件: {item['filename']}")
        logger.info(f"Prompt ({len(item['prompt'])} chars):")
        logger.info(item['prompt'])

        if "text_plan" in item:
            logger.info(f"\n文字叠加计划 ({len(item['text_plan'])} 元素):")
            for te in item['text_plan']:
                logger.info(f"  [{te['type']}] {te['text'][:50]}  @ {te['position']}")

    logger.info(f"\n{'─' * 50}")
    logger.info(f"总计: {len(prompt_plan)} 张图片待生成")
    logger.info(f"输出目录: {output_dir}")

    # 保存 prompt 计划到文件
    plan_path = os.path.join(output_dir, "prompt_plan.json")
    serializable = []
    for item in prompt_plan:
        entry = {k: v for k, v in item.items()}
        serializable.append(entry)

    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"Prompt 计划已保存: {plan_path}")


def _decide_retry_items(prompt_plan, validation_result):
    """根据验证失败原因决定需要重新生成的图片"""
    failures = validation_result.get("failures", [])
    items_to_retry = []

    for item in prompt_plan:
        if "background_score" in failures and item["type"] == "main":
            # 主图白底问题 - 在 auto_fix 中已尝试修复，
            # 如果还失败，调整 prompt 强调白底
            adjusted = dict(item)
            adjusted["prompt"] = (
                "CRITICAL: The background MUST be PURE WHITE RGB(255,255,255). "
                "Absolutely no gray, no shadow, no gradient. "
                "Check every pixel in the background area. \n\n" + item["prompt"]
            )
            items_to_retry.append(adjusted)

        if "product_ratio_score" in failures and item["type"] == "main":
            if item not in [i for i in items_to_retry if i["slot"] == item["slot"]]:
                adjusted = dict(item)
                adjusted["prompt"] = item["prompt"].replace(
                    "70-85%", "75-85%"
                ).replace(
                    "occupying", "prominently occupying at least"
                )
                items_to_retry.append(adjusted)

    return items_to_retry


def _post_process(output_dir):
    """后处理：确保所有图片符合输出规格"""
    from PIL import Image as PILImage
    from image_generator import resize_to_spec
    from config import IMAGE_SPECS

    for filename in os.listdir(output_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if filename.startswith("_"):
            continue  # 跳过临时文件

        filepath = os.path.join(output_dir, filename)
        try:
            img = PILImage.open(filepath)
            w, h = img.size
            target_w = IMAGE_SPECS["output_width"]
            target_h = IMAGE_SPECS["output_height"]

            if w != target_w or h != target_h:
                resize_to_spec(filepath)
                logger.info(f"已 resize: {filename} ({w}x{h} → {target_w}x{target_h})")
        except Exception as e:
            logger.warning(f"后处理失败 {filename}: {e}")


def _list_output_files(output_dir):
    """列出输出目录中的文件"""
    files = sorted(os.listdir(output_dir))
    for f in files:
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            logger.info(f"  {f} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Amazon 商品图片优化 Demo - 从原图生成 Amazon 标准组图"
    )
    parser.add_argument(
        "--photo", "-p",
        required=False,
        help="商品原图路径 (JPEG/PNG)"
    )
    parser.add_argument(
        "--product-info", "-i",
        required=False,
        help="产品信息 JSON 文件路径"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        choices=["siliconflow", "gpt-4o", "flux"],
        help="AI 模型 (默认: siliconflow 免费)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="仅生成 prompt 不调用 API"
    )
    parser.add_argument(
        "--output-name", "-o",
        default=None,
        help="输出目录名 (默认: 产品名)"
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int, default=3,
        help="不达标时最大重试次数 (默认: 3)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式输入模式（逐步引导填写商品信息）"
    )

    args = parser.parse_args()

    # 判断是否进入交互模式：
    # 1. 显式 --interactive 参数
    # 2. 没有提供任何参数时自动进入
    no_args = not args.photo and not args.product_info and not args.dry_run
    use_interactive = args.interactive or no_args

    if use_interactive:
        print("\n" + "*" * 50)
        print("  Amazon 商品图片优化 Demo")
        print("  交互式模式 — 跟着提示一步步输入即可")
        print("*" * 50)

        # 交互式获取产品信息
        product_info = interactive_input()

        # 交互式获取图片路径（dry-run 时跳过）
        if args.dry_run:
            photo_path = None
        else:
            photo_path = interactive_photo_input()

        # 保存后直接用内存中的 product_info 运行
        run_pipeline(
            photo_path=photo_path,
            product_info_path=os.path.join(BASE_DIR, "product_info.json"),
            model=args.model,
            dry_run=args.dry_run,
            output_name=args.output_name,
            max_retries=args.max_retries,
        )
    else:
        # 命令行参数模式
        if not args.dry_run and not args.photo:
            parser.error("--photo 是必需的 (除非使用 --dry-run 或 --interactive)")

        run_pipeline(
            photo_path=args.photo,
            product_info_path=args.product_info,
            model=args.model,
            dry_run=args.dry_run,
            output_name=args.output_name,
            max_retries=args.max_retries,
        )


if __name__ == "__main__":
    main()
