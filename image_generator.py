"""
AI 图片生成引擎
- 策略模式：GPT4oGenerator（主）+ FluxGenerator（备）
- 主图：image-to-image 编辑（以原图为参考，换白底+增强）
- 场景图/细节图/多角度：text-to-image + 原图参考
- 信息图：混合方案（AI 生成底图 + Pillow 叠加文字层）
- 费用追踪
"""

import base64
import io
import json
import logging
import os
import time
from abc import ABC, abstractmethod

import requests
from PIL import Image, ImageDraw, ImageFont

from config import (
    get_openai_api_key, get_openai_base_url,
    get_flux_api_key, get_siliconflow_api_key, get_primary_model,
    COST_ESTIMATES, INFOGRAPHIC_CONFIG, IMAGE_SPECS,
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 辅助函数
# ============================================================

def encode_image_to_base64(image_path):
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_image_from_base64(b64_data, output_path):
    """将 base64 图片数据保存为文件"""
    img_data = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(img_data)


def save_image_from_url(url, output_path):
    """从 URL 下载图片并保存"""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)


def resize_to_spec(image_path, width=None, height=None, quality=None):
    """将图片 resize 到指定尺寸并保存"""
    if width is None:
        width = IMAGE_SPECS["output_width"]
    if height is None:
        height = IMAGE_SPECS["output_height"]
    if quality is None:
        quality = IMAGE_SPECS["output_quality"]

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((width, height), Image.LANCZOS)
    img.save(image_path, "JPEG", quality=quality)


# ============================================================
# 信息图文字叠加（Pillow 精确渲染）
# ============================================================

def find_font(size):
    """查找可用字体"""
    candidates = INFOGRAPHIC_CONFIG["font_candidates"]

    # 系统字体目录
    font_dirs = [
        "C:/Windows/Fonts/",
        "/usr/share/fonts/truetype/",
        "/System/Library/Fonts/",
        os.path.join(BASE_DIR, "fonts/"),
    ]

    for font_name in candidates:
        # 先尝试直接加载
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            pass

        # 再尝试完整路径
        for font_dir in font_dirs:
            font_path = os.path.join(font_dir, font_name)
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except (OSError, IOError):
                    pass

    # 回退到默认字体
    logger.warning("未找到合适字体，使用默认字体")
    return ImageFont.load_default()


def overlay_infographic_text(base_image_path, text_plan, output_path):
    """
    在底图上叠加文字层，生成信息图。

    text_plan: 来自 prompt_engine.build_infographic_text_plan() 的文字元素列表
    """
    cfg = INFOGRAPHIC_CONFIG
    img = Image.open(base_image_path).convert("RGBA")
    w, h = img.size
    padding = cfg["padding"]

    # 创建文字叠加层
    txt_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    font_title = find_font(cfg["font_size_title"])
    font_body = find_font(cfg["font_size_body"])

    for element in text_plan:
        text = element.get("text", "")
        position = element.get("position", "")
        el_type = element.get("type", "")
        style = element.get("style", "")

        font = font_title if element.get("font_size") == "title" else font_body

        # 获取文字尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 根据 position 计算坐标
        x, y = _compute_text_position(position, w, h, tw, th, padding, el_type)

        # 绘制背景框（如果是 box 样式）
        if style == "box":
            box_padding = 15
            box_rect = [x - box_padding, y - box_padding,
                        x + tw + box_padding, y + th + box_padding]
            draw.rounded_rectangle(box_rect, radius=10,
                                   fill=(*cfg["bg_overlay_color"], cfg["bg_overlay_opacity"]))

        # 绘制文字阴影
        if cfg["text_shadow"]:
            sx, sy = cfg["text_shadow_offset"]
            draw.text((x + sx, y + sy), text, font=font,
                      fill=(*cfg["text_shadow_color"], 180))

        # 绘制图标（简单的勾号）
        if element.get("icon") == "check":
            icon_x = x - 30
            icon_y = y + th // 2 - 8
            draw.ellipse([icon_x, icon_y, icon_x + 20, icon_y + 20],
                         fill=cfg["accent_color"])
            # 在圆中画勾
            draw.text((icon_x + 3, icon_y - 2), "✓", font=find_font(16),
                      fill=(255, 255, 255, 255))

        # 绘制文字
        if el_type == "title":
            color = cfg["font_color_dark"]
        else:
            color = cfg["font_color_primary"] if style == "box" else cfg["font_color_dark"]

        draw.text((x, y), text, font=font, fill=(*color, 255))

    # 合并图层
    result = Image.alpha_composite(img, txt_layer)
    result = result.convert("RGB")
    result.save(output_path, "JPEG", quality=IMAGE_SPECS["output_quality"])
    logger.info(f"信息图文字叠加完成: {output_path}")


def _compute_text_position(position, img_w, img_h, text_w, text_h, padding, el_type):
    """根据位置标记计算文字坐标"""
    if position == "top_center":
        return ((img_w - text_w) // 2, padding)

    if position.startswith("left_"):
        row = int(position.split("_")[1])
        x = padding
        y = padding + 80 + row * (text_h + 40)
        return (x, y)

    if position.startswith("right_"):
        row = int(position.split("_")[1])
        x = img_w - text_w - padding
        y = padding + 80 + row * (text_h + 40)
        return (x, y)

    if position.startswith("bottom_"):
        col = int(position.split("_")[1])
        total_cols = 4
        col_width = (img_w - 2 * padding) // total_cols
        x = padding + col * col_width + (col_width - text_w) // 2
        y = img_h - padding - text_h - 20
        return (x, y)

    # 默认居中
    return ((img_w - text_w) // 2, (img_h - text_h) // 2)


# ============================================================
# 生成器基类
# ============================================================

class ImageGenerator(ABC):
    """图片生成器抽象基类"""

    def __init__(self):
        self.total_cost = 0.0
        self.call_count = 0

    @abstractmethod
    def generate(self, prompt, output_path, reference_image=None):
        """
        生成图片。

        参数:
            prompt: 文字 prompt
            output_path: 输出文件路径
            reference_image: 参考图片路径（可选）

        返回:
            bool: 是否成功
        """
        pass

    def get_stats(self):
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "call_count": self.call_count,
        }


# ============================================================
# GPT-4o 生成器
# ============================================================

class GPT4oGenerator(ImageGenerator):
    """OpenAI GPT-4o 图片生成器"""

    def __init__(self):
        super().__init__()
        self.api_key = get_openai_api_key()
        self.base_url = get_openai_base_url()
        self.client = None

    def _get_client(self):
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("请安装 openai 库: pip install openai>=1.0")

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self.client = OpenAI(**kwargs)
        return self.client

    def generate(self, prompt, output_path, reference_image=None):
        """使用 GPT-4o 生成图片"""
        if not self.api_key:
            logger.error("未配置 OpenAI API Key")
            return False

        client = self._get_client()
        self.call_count += 1

        try:
            if reference_image and os.path.exists(reference_image):
                return self._generate_with_reference(client, prompt, output_path, reference_image)
            else:
                return self._generate_text_only(client, prompt, output_path)
        except Exception as e:
            logger.error(f"GPT-4o 生成失败: {e}")
            return False

    def _generate_text_only(self, client, prompt, output_path):
        """纯文字生成（使用 dall-e-3）"""
        logger.info(f"DALL-E-3 text-to-image: {output_path}")

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="hd",
        )

        # 处理响应
        image_data = response.data[0]
        if hasattr(image_data, "b64_json") and image_data.b64_json:
            save_image_from_base64(image_data.b64_json, output_path)
        elif hasattr(image_data, "url") and image_data.url:
            save_image_from_url(image_data.url, output_path)
        else:
            logger.error("DALL-E-3 响应中无图片数据")
            return False

        resize_to_spec(output_path)
        self.total_cost += COST_ESTIMATES["gpt-4o"]["text_to_image"]
        logger.info(f"生成成功: {output_path}")
        return True

    def _generate_with_reference(self, client, prompt, output_path, reference_image):
        """带参考图的编辑生成（dall-e-2），失败则回退到纯文字生成"""
        logger.info(f"DALL-E-2 image-edit: {output_path} (参考: {reference_image})")

        try:
            # 读取参考图并转为 PNG bytes（OpenAI edit API 要求）
            ref_img = Image.open(reference_image)
            if ref_img.mode != "RGBA":
                ref_img = ref_img.convert("RGBA")
            ref_img = ref_img.resize((1024, 1024), Image.LANCZOS)

            buf = io.BytesIO()
            ref_img.save(buf, format="PNG")
            buf.seek(0)

            response = client.images.edit(
                model="dall-e-2",
                image=buf,
                prompt=prompt[:1000],  # dall-e-2 edit prompt 限制较短
                n=1,
                size="1024x1024",
            )

            image_data = response.data[0]
            if hasattr(image_data, "b64_json") and image_data.b64_json:
                save_image_from_base64(image_data.b64_json, output_path)
            elif hasattr(image_data, "url") and image_data.url:
                save_image_from_url(image_data.url, output_path)
            else:
                raise ValueError("edit 响应中无图片数据")

            resize_to_spec(output_path)
            self.total_cost += COST_ESTIMATES["gpt-4o"]["image_edit"]
            logger.info(f"编辑生成成功: {output_path}")
            return True

        except Exception as e:
            logger.warning(f"image-edit 失败 ({e})，回退到 DALL-E-3 text-to-image")
            return self._generate_text_only(client, prompt, output_path)


# ============================================================
# Flux 生成器
# ============================================================

class FluxGenerator(ImageGenerator):
    """Black Forest Labs Flux 2 图片生成器"""

    API_BASE = "https://api.bfl.ml/v1"

    def __init__(self):
        super().__init__()
        self.api_key = get_flux_api_key()

    def generate(self, prompt, output_path, reference_image=None):
        """使用 Flux 2 生成图片"""
        if not self.api_key:
            logger.error("未配置 Flux API Key")
            return False

        self.call_count += 1

        try:
            if reference_image and os.path.exists(reference_image):
                return self._generate_with_reference(prompt, output_path, reference_image)
            else:
                return self._generate_text_only(prompt, output_path)
        except Exception as e:
            logger.error(f"Flux 生成失败: {e}")
            return False

    def _generate_text_only(self, prompt, output_path):
        """Flux text-to-image"""
        logger.info(f"Flux text-to-image: {output_path}")

        # 提交生成任务
        resp = requests.post(
            f"{self.API_BASE}/flux-pro-1.1",
            headers={"X-Key": self.api_key},
            json={
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        task_id = resp.json().get("id")

        if not task_id:
            logger.error("Flux 未返回任务 ID")
            return False

        # 轮询结果
        result_url = self._poll_result(task_id)
        if not result_url:
            return False

        save_image_from_url(result_url, output_path)
        resize_to_spec(output_path)
        self.total_cost += COST_ESTIMATES["flux"]["text_to_image"]
        logger.info(f"生成成功: {output_path}")
        return True

    def _generate_with_reference(self, prompt, output_path, reference_image):
        """Flux image-to-image (使用 Redux 端点)"""
        logger.info(f"Flux image-edit: {output_path}")

        b64_ref = encode_image_to_base64(reference_image)

        resp = requests.post(
            f"{self.API_BASE}/flux-pro-1.1",
            headers={"X-Key": self.api_key},
            json={
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "image": b64_ref,
                "strength": 0.65,
            },
            timeout=30,
        )
        resp.raise_for_status()
        task_id = resp.json().get("id")

        if not task_id:
            return False

        result_url = self._poll_result(task_id)
        if not result_url:
            return False

        save_image_from_url(result_url, output_path)
        resize_to_spec(output_path)
        self.total_cost += COST_ESTIMATES["flux"]["image_edit"]
        logger.info(f"编辑生成成功: {output_path}")
        return True

    def _poll_result(self, task_id, max_wait=120, interval=2):
        """轮询 Flux 异步任务结果"""
        start = time.time()
        while time.time() - start < max_wait:
            resp = requests.get(
                f"{self.API_BASE}/get_result",
                params={"id": task_id},
                headers={"X-Key": self.api_key},
                timeout=15,
            )
            data = resp.json()
            status = data.get("status")

            if status == "Ready":
                return data.get("result", {}).get("sample")
            elif status in ("Error", "Failed"):
                logger.error(f"Flux 任务失败: {data}")
                return None

            time.sleep(interval)

        logger.error(f"Flux 任务超时 ({max_wait}s)")
        return None


# ============================================================
# SiliconFlow 生成器（免费，使用 Flux.1-schnell）
# ============================================================

class SiliconFlowGenerator(ImageGenerator):
    """硅基流动 SiliconFlow 图片生成器（免费 Flux.1-schnell 模型）"""

    API_URL = "https://api.siliconflow.cn/v1/images/generations"

    def __init__(self):
        super().__init__()
        self.api_key = get_siliconflow_api_key()

    def generate(self, prompt, output_path, reference_image=None):
        """使用 SiliconFlow 生成图片"""
        if not self.api_key:
            logger.error("未配置 SiliconFlow API Key")
            return False

        self.call_count += 1

        try:
            # SiliconFlow 统一使用 text-to-image（Flux.1-schnell 不支持 image edit）
            # 如果有参考图，在 prompt 前加上描述性引导
            final_prompt = prompt
            if reference_image and os.path.exists(reference_image):
                logger.info(f"SiliconFlow 不支持参考图编辑，使用纯文字生成: {output_path}")
            else:
                logger.info(f"SiliconFlow text-to-image: {output_path}")

            resp = requests.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "black-forest-labs/FLUX.1-schnell",
                    "prompt": final_prompt,
                    "image_size": "1024x1024",
                    "num_inference_steps": 4,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            # 解析响应 - SiliconFlow 返回格式: {"images": [{"url": "..."}]}
            images = data.get("images", [])
            if not images:
                logger.error(f"SiliconFlow 未返回图片: {data}")
                return False

            image_url = images[0].get("url", "")
            if not image_url:
                logger.error(f"SiliconFlow 响应中无图片 URL: {data}")
                return False

            save_image_from_url(image_url, output_path)
            resize_to_spec(output_path)
            self.total_cost += COST_ESTIMATES["siliconflow"]["text_to_image"]
            logger.info(f"生成成功: {output_path}")
            return True

        except Exception as e:
            logger.error(f"SiliconFlow 生成失败: {e}")
            return False


# ============================================================
# 工厂函数
# ============================================================

def create_generator(model=None):
    """创建图片生成器实例"""
    if model is None:
        model = get_primary_model()

    if model in ("siliconflow", "silicon", "sf"):
        return SiliconFlowGenerator()
    elif model in ("gpt-4o", "gpt4o", "openai"):
        return GPT4oGenerator()
    elif model in ("flux", "flux2", "bfl"):
        return FluxGenerator()
    else:
        logger.warning(f"未知模型 '{model}'，回退到 SiliconFlow")
        return SiliconFlowGenerator()


def generate_product_images(prompt_plan, output_dir, source_photo,
                            model=None, dry_run=False):
    """
    执行完整的图片生成流程。

    参数:
        prompt_plan: 来自 prompt_engine.generate_prompt_plan() 的计划
        output_dir: 输出目录
        source_photo: 用户提供的商品原图路径
        model: AI 模型名（None 则使用配置默认值）
        dry_run: 仅输出 prompt 不实际调用 API

    返回:
        dict: 生成结果 {
            "generated": [...],  # 成功生成的文件列表
            "failed": [...],     # 失败的列表
            "stats": {...},      # 费用统计
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    generator = create_generator(model)
    generated = []
    failed = []

    for item in prompt_plan:
        slot = item["slot"]
        img_type = item["type"]
        filename = item["filename"]
        prompt = item["prompt"]
        output_path = os.path.join(output_dir, filename)

        logger.info(f"\n{'='*50}")
        logger.info(f"生成 [{slot}] ({img_type})")

        if dry_run:
            logger.info(f"[DRY RUN] Prompt ({len(prompt)} chars):")
            logger.info(prompt[:300] + ("..." if len(prompt) > 300 else ""))
            generated.append({"slot": slot, "filename": filename, "dry_run": True})
            continue

        # 决定是否使用参考图
        use_reference = img_type in ("main", "lifestyle", "detail", "multiangle")
        ref_image = source_photo if (use_reference and source_photo
                                      and os.path.exists(source_photo)) else None

        # 生成底图
        if img_type == "infographic":
            # 信息图：先生成底图，再叠加文字
            base_path = os.path.join(output_dir, f"_base_{filename}")
            success = generator.generate(prompt, base_path, reference_image=ref_image)

            if success and "text_plan" in item:
                overlay_infographic_text(base_path, item["text_plan"], output_path)
                # 清理中间文件
                if os.path.exists(base_path):
                    os.remove(base_path)
            elif success:
                # 无文字计划，直接用底图
                os.rename(base_path, output_path)
            else:
                failed.append({"slot": slot, "filename": filename, "error": "generation_failed"})
                continue
        else:
            success = generator.generate(prompt, output_path, reference_image=ref_image)

        if success or (img_type == "infographic" and os.path.exists(output_path)):
            generated.append({
                "slot": slot,
                "filename": filename,
                "type": img_type,
                "prompt_length": len(prompt),
            })
        else:
            failed.append({"slot": slot, "filename": filename, "error": "generation_failed"})

    stats = generator.get_stats()
    logger.info(f"\n{'='*50}")
    logger.info(f"生成完成: {len(generated)} 成功, {len(failed)} 失败")
    logger.info(f"API 调用: {stats['call_count']} 次, 预估费用: ${stats['total_cost_usd']:.2f}")

    return {
        "generated": generated,
        "failed": failed,
        "stats": stats,
    }
