"""
Amazon汽车装饰品爬虫
使用Selenium爬取Amazon.com上car accessories的Top 100商品信息
"""

import json
import time
import random
import re
import os
import logging
from urllib.parse import urljoin, quote_plus

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException
)
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# User-Agent池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def random_delay(min_sec=2, max_sec=5):
    """随机延迟，模拟人类行为"""
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)


def create_driver():
    """创建Chrome WebDriver"""
    options = Options()
    ua = random.choice(USER_AGENTS)
    options.add_argument(f"user-agent={ua}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    # 不使用headless，以减少被检测的风险
    # options.add_argument("--headless=new")

    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)
    # 隐藏webdriver标志
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.navigator.chrome = {runtime: {}};
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """
    })
    driver.implicitly_wait(10)
    return driver


def extract_asin_from_url(url):
    """从URL中提取ASIN"""
    patterns = [
        r'/dp/([A-Z0-9]{10})',
        r'/gp/product/([A-Z0-9]{10})',
        r'/product/([A-Z0-9]{10})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_asin_from_element(element):
    """从商品元素中提取ASIN"""
    try:
        return element.get_attribute("data-asin")
    except Exception:
        return None


def scroll_page_slowly(driver):
    """缓慢滚动页面，模拟人类浏览行为"""
    total_height = driver.execute_script("return document.body.scrollHeight")
    current = 0
    step = random.randint(300, 500)
    while current < total_height:
        current += step
        driver.execute_script(f"window.scrollTo(0, {current});")
        time.sleep(random.uniform(0.3, 0.8))


def get_product_list(driver, search_term="car accessories", max_products=100):
    """
    在Amazon上搜索并获取商品列表
    """
    products = []
    seen_asins = set()

    # 访问Amazon主页
    logger.info("正在访问Amazon.com...")
    driver.get("https://www.amazon.com")
    random_delay(3, 6)

    # 搜索商品
    logger.info(f"搜索: {search_term}")
    try:
        search_box = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "twotabsearchtextbox"))
        )
        search_box.clear()
        # 模拟逐字输入
        for char in search_term:
            search_box.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))
        random_delay(0.5, 1.5)
        search_box.send_keys(Keys.RETURN)
    except TimeoutException:
        logger.error("无法找到搜索框，可能遇到验证码")
        return products

    random_delay(3, 5)

    # 尝试按Best Sellers排序（使用JS操作，避免元素遮挡）
    try:
        sort_select = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "s-result-sort-select"))
        )
        # 使用JavaScript直接设置排序值
        driver.execute_script("""
            var select = document.getElementById('s-result-sort-select');
            if (select) {
                select.value = 'review-rank';
                select.dispatchEvent(new Event('change', {bubbles: true}));
            }
        """)
        random_delay(3, 5)
    except Exception as e:
        logger.warning(f"无法排序，使用默认排序: {e}")

    page = 1
    while len(products) < max_products:
        logger.info(f"正在解析第 {page} 页 (已获取 {len(products)} 个商品)...")

        # 缓慢滚动加载所有图片
        scroll_page_slowly(driver)
        random_delay(1, 2)

        # 获取页面源码并用BeautifulSoup解析
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # 查找商品卡片
        items = soup.select('[data-component-type="s-search-result"]')
        if not items:
            items = soup.select('.s-result-item[data-asin]')

        for item in items:
            if len(products) >= max_products:
                break

            asin = item.get("data-asin", "").strip()
            if not asin or asin in seen_asins or len(asin) != 10:
                continue

            seen_asins.add(asin)

            # 提取标题
            title_elem = item.select_one("h2 a span") or item.select_one("h2 span")
            title = title_elem.get_text(strip=True) if title_elem else "N/A"

            # 提取价格
            price = "N/A"
            price_whole = item.select_one(".a-price-whole")
            price_frac = item.select_one(".a-price-fraction")
            if price_whole:
                price_text = price_whole.get_text(strip=True).replace(",", "")
                frac_text = price_frac.get_text(strip=True) if price_frac else "00"
                price = f"${price_text}{frac_text}"

            # 提取评分
            rating = "N/A"
            rating_elem = item.select_one(".a-icon-alt")
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                match = re.search(r'([\d.]+)', rating_text)
                if match:
                    rating = match.group(1)

            # 提取评论数 (多种选择器兼容Amazon不同布局)
            reviews = "0"
            review_selectors = [
                'a[href*="#customerReviews"] span.a-size-base',
                'a[href*="customerReviews"] span',
                '.a-row.a-size-small a span.a-size-base',
                '.s-underline-text',
                '[aria-label*="stars"] ~ span a span',
                '[aria-label*="stars"] + span.a-size-base',
            ]
            for sel in review_selectors:
                reviews_elem = item.select_one(sel)
                if reviews_elem:
                    text = reviews_elem.get_text(strip=True).replace(",", "")
                    # 确保是数字（评论数），而不是评分文字
                    if text and re.match(r'^\d+$', text):
                        reviews = text
                        break

            # 提取商品链接
            link_elem = item.select_one("h2 a")
            product_url = ""
            if link_elem and link_elem.get("href"):
                href = link_elem["href"]
                if href.startswith("/"):
                    product_url = f"https://www.amazon.com{href}"
                else:
                    product_url = href

            # 提取搜索页的主图URL
            img_elem = item.select_one(".s-image")
            search_img_url = img_elem.get("src", "") if img_elem else ""

            product = {
                "asin": asin,
                "title": title,
                "price": price,
                "rating": rating,
                "reviews": reviews,
                "url": product_url,
                "search_image_url": search_img_url,
                "images": [],
                "rank": len(products) + 1
            }
            products.append(product)
            logger.info(f"  [{len(products)}] {asin}: {title[:50]}...")

        if len(products) >= max_products:
            break

        # 翻到下一页
        try:
            next_btn = driver.find_element(
                By.CSS_SELECTOR,
                ".s-pagination-next:not(.s-pagination-disabled)"
            )
            # 先滚动到翻页按钮位置，再用JS点击，避免被搜索框遮挡
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
            random_delay(0.5, 1)
            try:
                next_btn.click()
            except Exception:
                # 如果常规点击失败，尝试JS点击
                driver.execute_script("arguments[0].click();", next_btn)
            page += 1
            random_delay(3, 6)
        except NoSuchElementException:
            # 尝试直接构造下一页URL
            next_page_url = f"https://www.amazon.com/s?k={quote_plus(search_term)}&page={page+1}"
            try:
                driver.get(next_page_url)
                page += 1
                random_delay(3, 5)
                # 检查是否真的有搜索结果
                test_soup = BeautifulSoup(driver.page_source, "html.parser")
                test_items = test_soup.select('[data-component-type="s-search-result"]')
                if not test_items:
                    logger.info("没有更多页面了")
                    break
            except Exception:
                logger.info("没有更多页面了")
                break

    logger.info(f"共获取 {len(products)} 个商品列表")
    return products


def get_product_detail_images(driver, product):
    """进入商品详情页获取所有图片URL"""
    asin = product["asin"]
    url = product.get("url", "")

    if not url:
        url = f"https://www.amazon.com/dp/{asin}"

    logger.info(f"  获取 {asin} 的详情页图片...")

    try:
        driver.get(url)
        random_delay(2, 4)

        # 等待图片区域加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "imgTagWrapperId"))
        )
    except TimeoutException:
        logger.warning(f"  {asin} 详情页加载超时")
    except Exception as e:
        logger.warning(f"  {asin} 访问失败: {e}")

    images = []
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # 尝试从详情页提取评论数和评分（比搜索页更准确）
    if product.get("reviews", "0") == "0" or product.get("reviews") == "N/A":
        detail_review_selectors = [
            "#acrCustomerReviewText",
            "#averageCustomerReviews .a-size-base",
            "#acrCustomerReviewLink span",
            "#reviewsMedley [data-hook='total-review-count'] span",
        ]
        for sel in detail_review_selectors:
            elem = soup.select_one(sel)
            if elem:
                text = elem.get_text(strip=True)
                match = re.search(r'([\d,]+)', text)
                if match:
                    product["reviews"] = match.group(1).replace(",", "")
                    break

    if product.get("rating", "N/A") == "N/A":
        rating_elem = soup.select_one("#acrPopover .a-icon-alt, #averageCustomerReviews .a-icon-alt")
        if rating_elem:
            match = re.search(r'([\d.]+)', rating_elem.get_text(strip=True))
            if match:
                product["rating"] = match.group(1)

    # 方法1: 从JS变量中提取图片数据 (最可靠)
    scripts = soup.find_all("script", string=re.compile(r"ImageBlockATF|colorImages"))
    for script in scripts:
        text = script.string or ""
        # 匹配 hiRes 图片URL
        hi_res = re.findall(r'"hiRes"\s*:\s*"(https://[^"]+)"', text)
        if hi_res:
            images.extend(hi_res)
            continue
        # 匹配 large 图片URL
        large = re.findall(r'"large"\s*:\s*"(https://[^"]+)"', text)
        if large:
            images.extend(large)

    # 方法2: 从缩略图列表提取并转换为高分辨率
    if not images:
        thumb_list = soup.select("#altImages .imageThumbnail img, #altImages .a-button-thumbnail img")
        for thumb in thumb_list:
            src = thumb.get("src", "")
            if src and "amazon" in src:
                # 替换尺寸参数获取高分辨率版本
                hi_res_url = re.sub(
                    r'\._[A-Z]{2}\d+_\.',
                    '._SL1500_.',
                    src
                )
                if hi_res_url not in images:
                    images.append(hi_res_url)

    # 方法3: 主图
    if not images:
        main_img = soup.select_one("#landingImage, #imgBlkFront")
        if main_img:
            # 尝试data-old-hires属性
            hi_res = main_img.get("data-old-hires", "")
            if hi_res:
                images.append(hi_res)
            else:
                src = main_img.get("src", "")
                if src:
                    images.append(src)

    # 去重并过滤无效URL
    seen = set()
    unique_images = []
    for img_url in images:
        if not img_url or img_url in seen:
            continue
        # 过滤播放按钮等非商品图片
        if "play-button" in img_url or "sprite" in img_url:
            continue
        seen.add(img_url)
        unique_images.append(img_url)

    # 尝试将所有图片转为最高分辨率
    high_res_images = []
    for img_url in unique_images:
        hr = re.sub(r'\._[A-Z]{2}\d+[A-Z_]*_\.', '._SL1500_.', img_url)
        high_res_images.append(hr)

    product["images"] = high_res_images if high_res_images else unique_images
    logger.info(f"  {asin}: 获取到 {len(product['images'])} 张图片")

    return product


def scrape_amazon(search_term="car accessories", max_products=100, output_file="products.json"):
    """主爬虫函数"""
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)

    # 加载已有数据（断点续传）
    existing_products = []
    existing_asins = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_products = json.load(f)
            existing_asins = {p["asin"] for p in existing_products if p.get("images")}
            logger.info(f"已有 {len(existing_asins)} 个完整商品数据")
        except Exception:
            existing_products = []

    driver = create_driver()

    try:
        # Step 1: 获取商品列表
        if len(existing_products) < max_products:
            products = get_product_list(driver, search_term, max_products)
            # 合并新旧数据
            for p in products:
                if p["asin"] not in existing_asins:
                    existing_products.append(p)
        else:
            products = existing_products

        # Step 2: 获取每个商品的详情页图片
        for i, product in enumerate(existing_products):
            if product["asin"] in existing_asins and product.get("images"):
                logger.info(f"[{i+1}/{len(existing_products)}] {product['asin']} 已有图片数据，跳过")
                continue

            logger.info(f"[{i+1}/{len(existing_products)}] 获取 {product['asin']} 的图片...")
            get_product_detail_images(driver, product)

            # 每获取5个商品就保存一次
            if (i + 1) % 5 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(existing_products, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存进度 ({i+1}/{len(existing_products)})")

            random_delay(2, 5)

        # 最终保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_products, f, ensure_ascii=False, indent=2)

        logger.info(f"爬虫完成! 共 {len(existing_products)} 个商品，数据已保存到 {output_file}")
        return existing_products

    except Exception as e:
        logger.error(f"爬虫出错: {e}")
        # 保存已获取的数据
        if existing_products:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing_products, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(existing_products)} 个已获取的商品数据")
            return existing_products
        raise
    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_amazon()
