"""
修复脚本：重新提取评论数和评分
访问每个商品的详情页，提取真实的评论数和评分
"""

import json
import os
import re
import time
import random
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def random_delay(min_sec=1, max_sec=3):
    time.sleep(random.uniform(min_sec, max_sec))


def fix_reviews():
    products_path = os.path.join(BASE_DIR, "products.json")

    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    # 只处理缺少评论数的商品
    to_process = [(i, p) for i, p in enumerate(products) if p.get("reviews", "0") == "0"]
    logger.info(f"共 {len(products)} 个商品，{len(to_process)} 个需要更新评论数...")

    if not to_process:
        logger.info("所有商品都已有评论数据，无需更新")
        return

    from scraper import create_driver
    from bs4 import BeautifulSoup

    driver = create_driver()

    try:
        updated = 0
        for idx, (i, product) in enumerate(to_process):
            asin = product["asin"]
            url = product.get("url") or f"https://www.amazon.com/dp/{asin}"
            if not url.startswith("http"):
                url = f"https://www.amazon.com/dp/{asin}"

            logger.info(f"[{idx+1}/{len(to_process)}] {asin} (#{i+1})...")

            try:
                driver.get(url)
                random_delay(2, 4)

                soup = BeautifulSoup(driver.page_source, "html.parser")

                # 提取评论数
                old_reviews = product.get("reviews", "0")
                new_reviews = "0"

                review_selectors = [
                    "#acrCustomerReviewText",
                    "#averageCustomerReviews .a-size-base",
                    "#acrCustomerReviewLink span",
                    "#reviewsMedley [data-hook='total-review-count'] span",
                    "#social-proofing-faceout-title-tk_bought span",
                ]
                for sel in review_selectors:
                    elem = soup.select_one(sel)
                    if elem:
                        text = elem.get_text(strip=True)
                        match = re.search(r'([\d,]+)', text)
                        if match:
                            new_reviews = match.group(1).replace(",", "")
                            break

                # 提取评分
                old_rating = product.get("rating", "N/A")
                new_rating = old_rating
                rating_elem = soup.select_one(
                    "#acrPopover .a-icon-alt, "
                    "#averageCustomerReviews .a-icon-alt, "
                    "#acrPopover span.a-icon-alt"
                )
                if rating_elem:
                    match = re.search(r'([\d.]+)', rating_elem.get_text(strip=True))
                    if match:
                        new_rating = match.group(1)

                product["reviews"] = new_reviews
                product["rating"] = new_rating

                if new_reviews != old_reviews or new_rating != old_rating:
                    updated += 1
                    logger.info(f"  -> reviews: {old_reviews} -> {new_reviews}, rating: {old_rating} -> {new_rating}")
                else:
                    logger.info(f"  -> reviews={new_reviews}, rating={new_rating} (unchanged)")

            except Exception as e:
                logger.warning(f"  {asin} 访问失败: {e}")

            # 每10个保存一次
            if (idx + 1) % 10 == 0:
                with open(products_path, "w", encoding="utf-8") as f:
                    json.dump(products, f, ensure_ascii=False, indent=2)
                logger.info(f"  已保存进度 ({idx+1}/{len(to_process)})")

            random_delay(1, 3)

        # 最终保存
        with open(products_path, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

        logger.info(f"\n完成! 更新了 {updated}/{len(products)} 个商品的评论数据")

        # 统计
        non_zero = sum(1 for p in products if p.get("reviews", "0") != "0")
        logger.info(f"有评论数据的商品: {non_zero}/{len(products)}")

    finally:
        driver.quit()


if __name__ == "__main__":
    fix_reviews()
