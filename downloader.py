"""
图片下载器
读取products.json，按ASIN文件夹下载所有商品图片
支持断点续传和高分辨率下载
"""

import json
import os
import re
import time
import random
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# 下载配置
MAX_RETRIES = 3
TIMEOUT = 30
CONCURRENT_DOWNLOADS = 3

HEADERS_LIST = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.amazon.com/",
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Accept": "image/avif,image/webp,*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.amazon.com/",
    },
]


def upgrade_image_url(url):
    """将Amazon图片URL升级为高分辨率版本"""
    if not url:
        return url
    # 替换尺寸参数为SL1500（高分辨率）
    upgraded = re.sub(r'\._[A-Z]{2}\d+[A-Z_]*_\.', '._SL1500_.', url)
    return upgraded


def get_image_extension(url, response=None):
    """从URL或响应头判断图片格式"""
    if response and "Content-Type" in response.headers:
        ct = response.headers["Content-Type"].lower()
        if "jpeg" in ct or "jpg" in ct:
            return ".jpg"
        elif "png" in ct:
            return ".png"
        elif "webp" in ct:
            return ".webp"
        elif "gif" in ct:
            return ".gif"

    # 从URL中提取
    url_lower = url.lower().split("?")[0]
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        if url_lower.endswith(ext):
            return ext if ext != ".jpeg" else ".jpg"

    return ".jpg"  # 默认jpg


def download_single_image(url, save_path, retries=MAX_RETRIES):
    """下载单张图片，支持重试"""
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True, save_path, "已存在(跳过)"

    for attempt in range(retries):
        try:
            headers = random.choice(HEADERS_LIST).copy()
            response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
            response.raise_for_status()

            content_length = int(response.headers.get("Content-Length", 0))
            if content_length > 0 and content_length < 500:
                return False, save_path, "图片太小，可能是占位图"

            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 写入文件
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(save_path)
            if file_size < 500:
                os.remove(save_path)
                return False, save_path, f"文件太小 ({file_size}B)"

            return True, save_path, f"成功 ({file_size // 1024}KB)"

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 2 + random.uniform(0, 1)
                time.sleep(wait)
            else:
                return False, save_path, f"下载失败: {e}"

    return False, save_path, "超过最大重试次数"


def download_product_images(product, images_dir=IMAGES_DIR):
    """下载单个商品的所有图片"""
    asin = product.get("asin", "")
    if not asin:
        return 0

    product_dir = os.path.join(images_dir, asin)
    os.makedirs(product_dir, exist_ok=True)

    images = product.get("images", [])
    if not images:
        # 如果没有详情页图片，使用搜索页的图片
        search_img = product.get("search_image_url", "")
        if search_img:
            images = [upgrade_image_url(search_img)]

    if not images:
        logger.warning(f"  {asin}: 没有图片URL")
        return 0

    downloaded = 0
    for idx, img_url in enumerate(images):
        if not img_url:
            continue

        # 升级为高分辨率
        hr_url = upgrade_image_url(img_url)

        # 命名规则：第一张为主图，其余为辅图
        if idx == 0:
            filename = "main.jpg"
        else:
            filename = f"sub_{idx}.jpg"

        save_path = os.path.join(product_dir, filename)

        # 先尝试高分辨率
        success, path, msg = download_single_image(hr_url, save_path)
        if not success and hr_url != img_url:
            # 高分辨率失败则尝试原始URL
            success, path, msg = download_single_image(img_url, save_path)

        if success:
            downloaded += 1
            logger.debug(f"  {asin}/{filename}: {msg}")
        else:
            logger.warning(f"  {asin}/{filename}: {msg}")

    return downloaded


def download_all_images(products_file="products.json", images_dir=None):
    """下载所有商品图片"""
    if images_dir is None:
        images_dir = IMAGES_DIR

    products_path = os.path.join(BASE_DIR, products_file)
    if not os.path.exists(products_path):
        logger.error(f"找不到商品数据文件: {products_path}")
        return

    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    logger.info(f"共 {len(products)} 个商品待下载图片")

    total_downloaded = 0
    total_products = len(products)

    for i, product in enumerate(products):
        asin = product.get("asin", "unknown")
        logger.info(f"[{i+1}/{total_products}] 下载 {asin} 的图片...")

        count = download_product_images(product, images_dir)
        total_downloaded += count
        logger.info(f"  {asin}: 下载了 {count} 张图片")

        # 随机延迟，避免被封
        if (i + 1) % 10 == 0:
            random_delay = random.uniform(2, 5)
            logger.info(f"  休息 {random_delay:.1f}s...")
            time.sleep(random_delay)

    logger.info(f"下载完成! 共下载 {total_downloaded} 张图片")
    return total_downloaded


def get_download_stats(images_dir=None):
    """统计已下载图片"""
    if images_dir is None:
        images_dir = IMAGES_DIR

    if not os.path.exists(images_dir):
        return {"total_products": 0, "total_images": 0, "products": {}}

    stats = {"total_products": 0, "total_images": 0, "products": {}}

    for asin_dir in os.listdir(images_dir):
        asin_path = os.path.join(images_dir, asin_dir)
        if not os.path.isdir(asin_path):
            continue

        images = [f for f in os.listdir(asin_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        if images:
            stats["total_products"] += 1
            stats["total_images"] += len(images)
            stats["products"][asin_dir] = {
                "image_count": len(images),
                "has_main": "main.jpg" in images or "main.png" in images,
                "sub_count": len([f for f in images if f.startswith("sub_")]),
                "total_size_kb": sum(
                    os.path.getsize(os.path.join(asin_path, f)) // 1024
                    for f in images
                )
            }

    return stats


if __name__ == "__main__":
    download_all_images()
    stats = get_download_stats()
    logger.info(f"下载统计: {stats['total_products']} 个商品, {stats['total_images']} 张图片")
