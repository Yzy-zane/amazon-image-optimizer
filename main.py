"""
Amazon汽车装饰品图片质量分析 - 主入口
串联所有步骤：爬虫 → 下载 → 分析 → 报告 → 洞察
"""

import os
import sys
import json
import time
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.log"),
            encoding="utf-8"
        ),
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def step_scrape(search_term="car accessories", max_products=100):
    """Step 1: 爬取Amazon商品数据"""
    logger.info("=" * 60)
    logger.info("Step 1: 爬取Amazon商品数据")
    logger.info("=" * 60)

    from scraper import scrape_amazon
    products = scrape_amazon(
        search_term=search_term,
        max_products=max_products,
        output_file="products.json"
    )
    logger.info(f"Step 1 完成: 获取 {len(products)} 个商品")
    return products


def step_download():
    """Step 2: 下载商品图片"""
    logger.info("=" * 60)
    logger.info("Step 2: 下载商品图片")
    logger.info("=" * 60)

    from downloader import download_all_images, get_download_stats
    total = download_all_images()
    stats = get_download_stats()
    logger.info(f"Step 2 完成: {stats['total_products']} 个商品, {stats['total_images']} 张图片")
    return stats


def step_analyze():
    """Step 3: 图片质量分析"""
    logger.info("=" * 60)
    logger.info("Step 3: 图片质量分析")
    logger.info("=" * 60)

    from analyzer import analyze_all_products
    results = analyze_all_products()
    logger.info(f"Step 3 完成: 分析 {len(results)} 个商品")
    return results


def step_report():
    """Step 4: 生成Excel报告"""
    logger.info("=" * 60)
    logger.info("Step 4: 生成Excel报告")
    logger.info("=" * 60)

    from report import generate_report
    path = generate_report()
    logger.info(f"Step 4 完成: 报告已生成 → {path}")
    return path


def step_insights():
    """Step 5: 数据分析与洞察"""
    logger.info("=" * 60)
    logger.info("Step 5: 数据分析与洞察")
    logger.info("=" * 60)

    from analysis import generate_insights
    report = generate_insights()
    logger.info("Step 5 完成: 洞察报告已生成")
    return report


def run_all(search_term="car accessories", max_products=100, skip_steps=None):
    """运行全部流程"""
    if skip_steps is None:
        skip_steps = set()

    start_time = time.time()

    logger.info("*" * 60)
    logger.info("Amazon汽车装饰品图片质量分析 - 开始")
    logger.info(f"搜索词: {search_term}")
    logger.info(f"目标商品数: {max_products}")
    logger.info("*" * 60)

    try:
        # Step 1: 爬虫
        if 1 not in skip_steps:
            step_scrape(search_term, max_products)
        else:
            logger.info("跳过 Step 1: 爬虫")

        # Step 2: 下载图片
        if 2 not in skip_steps:
            step_download()
        else:
            logger.info("跳过 Step 2: 下载图片")

        # Step 3: 分析图片
        if 3 not in skip_steps:
            step_analyze()
        else:
            logger.info("跳过 Step 3: 分析图片")

        # Step 4: 生成报告
        if 4 not in skip_steps:
            step_report()
        else:
            logger.info("跳过 Step 4: 生成报告")

        # Step 5: 数据洞察
        if 5 not in skip_steps:
            step_insights()
        else:
            logger.info("跳过 Step 5: 数据洞察")

        elapsed = time.time() - start_time
        logger.info("\n" + "*" * 60)
        logger.info(f"全部完成! 总耗时: {elapsed/60:.1f} 分钟")
        logger.info("*" * 60)

        # 打印输出文件列表
        output_files = [
            ("products.json", "商品数据"),
            ("images/", "图片文件夹"),
            ("analysis_results.json", "分析结果"),
            ("汽车装饰品图片质量分析.xlsx", "Excel报告"),
            ("insights_report.json", "洞察报告"),
        ]

        logger.info("\n输出文件:")
        for fname, desc in output_files:
            fpath = os.path.join(BASE_DIR, fname)
            exists = os.path.exists(fpath)
            status = "[OK]" if exists else "[NO]"
            logger.info(f"  {status} {fname} - {desc}")

    except KeyboardInterrupt:
        logger.info("\n用户中断，已安全退出")
    except Exception as e:
        logger.error(f"\n运行出错: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Amazon汽车装饰品图片质量分析")
    parser.add_argument(
        "--search", "-s",
        default="car accessories",
        help="搜索关键词 (默认: car accessories)"
    )
    parser.add_argument(
        "--count", "-n",
        type=int, default=100,
        help="爬取商品数量 (默认: 100)"
    )
    parser.add_argument(
        "--skip", "-k",
        type=int, nargs="*", default=[],
        help="跳过的步骤编号 (1=爬虫 2=下载 3=分析 4=报告 5=洞察)"
    )
    parser.add_argument(
        "--step",
        type=int,
        help="只运行指定步骤"
    )

    args = parser.parse_args()

    if args.step:
        # 只运行指定步骤
        step_map = {
            1: lambda: step_scrape(args.search, args.count),
            2: step_download,
            3: step_analyze,
            4: step_report,
            5: step_insights,
        }
        if args.step in step_map:
            step_map[args.step]()
        else:
            print(f"无效步骤: {args.step} (有效: 1-5)")
    else:
        run_all(args.search, args.count, skip_steps=set(args.skip))


if __name__ == "__main__":
    main()
