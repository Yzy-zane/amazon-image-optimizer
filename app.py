"""
Amazon 商品图片优化 - FastAPI 后端

启动:
    python app.py
    # 或
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

API 文档: http://localhost:8000/docs
前端页面: http://localhost:8000/static/index.html
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional
from threading import Lock

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Amazon 商品图片优化 API", version="2.0.0")

# CORS - 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 线程池 (图片生成是 CPU/IO 混合型，用线程池即可)
executor = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# 任务存储 (内存字典，单机够用)
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}
jobs_lock = Lock()


def get_job(job_id: str) -> dict | None:
    with jobs_lock:
        return jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# 后台生成任务
# ---------------------------------------------------------------------------

def _run_generation(job_id: str, photo_path: str, product_info: dict,
                    model: str | None, output_dir: str):
    """在线程池中运行的生成任务"""
    try:
        update_job(job_id, status="generating", progress="0/8")

        # Step 1: 生成 prompt 计划
        from prompt_engine import generate_prompt_plan
        prompt_plan = generate_prompt_plan(product_info)
        logger.info(f"[{job_id[:8]}] Prompt 计划已生成: {len(prompt_plan)} 张")

        # Step 2: 逐张调用 AI 生成图片（便于更新进度）
        from image_generator import generate_product_images

        total = len(prompt_plan)
        all_generated = []
        all_failed = []
        total_cost = 0.0
        call_count = 0

        for i, item in enumerate(prompt_plan):
            update_job(job_id, progress=f"{i}/{total}",
                       current_slot=item["slot"])

            result = generate_product_images(
                prompt_plan=[item],
                output_dir=output_dir,
                source_photo=photo_path,
                model=model,
            )
            all_generated.extend(result["generated"])
            all_failed.extend(result["failed"])
            total_cost += result["stats"]["total_cost_usd"]
            call_count += result["stats"]["call_count"]

        update_job(job_id, progress=f"{total}/{total}")

        if not all_generated:
            update_job(job_id, status="failed", error="没有成功生成任何图片")
            return

        # Step 3: 质量验证
        update_job(job_id, status="validating", progress="评分中...")
        from image_validator import validate_and_fix
        product_name = product_info.get("name", "product").replace(" ", "_")
        val_result = validate_and_fix(output_dir, product_name)

        # 汇总生成的图片（排除上传的原图）
        files = []
        for f in sorted(os.listdir(output_dir)):
            if (f.lower().endswith((".jpg", ".jpeg", ".png"))
                    and not f.startswith("_")
                    and not f.startswith("source_")):
                fpath = os.path.join(output_dir, f)
                files.append({
                    "filename": f,
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                })

        update_job(
            job_id,
            status="done",
            progress="完成",
            files=files,
            scores=val_result["scores"],
            passed=val_result["passed"],
            failures=val_result.get("failures", []),
            suggestions=val_result.get("suggestions", []),
            cost_usd=round(total_cost, 4),
            generated_count=len(all_generated),
            failed_count=len(all_failed),
        )
        logger.info(f"[{job_id[:8]}] 任务完成, 评分: "
                     f"{val_result['scores'].get('normalized_score', 'N/A')}")

    except Exception as e:
        logger.exception(f"[{job_id[:8]}] 任务失败")
        update_job(job_id, status="failed", error=str(e))


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate(
    photos: List[UploadFile] = File(..., description="商品实拍图 (支持多张，第一张为主参考图)"),
    product_info: str = Form(..., description='商品信息 JSON'),
    model: str = Form(None, description="AI 模型: siliconflow(默认) / gpt-4o / flux"),
):
    """上传实拍图 + 商品信息 → 启动后台生成任务（1+7 组图）"""
    # 解析 product_info JSON
    try:
        info = json.loads(product_info)
    except json.JSONDecodeError as e:
        return {"error": f"product_info JSON 格式错误: {e}"}

    if not info.get("name"):
        return {"error": "product_info 必须包含 name 字段"}

    if not photos:
        return {"error": "请至少上传一张商品实拍图"}

    # 创建任务
    job_id = uuid.uuid4().hex
    product_name = info.get("name", "product").replace(" ", "_")

    from config import OUTPUT_DIR
    output_dir = os.path.join(OUTPUT_DIR, f"{product_name}_{job_id[:8]}")
    os.makedirs(output_dir, exist_ok=True)

    # 保存所有上传的原图，第一张作为主参考图
    primary_photo_path = None
    for idx, photo in enumerate(photos):
        photo_filename = f"source_{idx}_{photo.filename}"
        photo_path = os.path.join(output_dir, photo_filename)
        content = await photo.read()
        with open(photo_path, "wb") as f:
            f.write(content)
        if idx == 0:
            primary_photo_path = photo_path

    # 注册任务
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": "排队中",
            "product_name": product_name,
            "model": model,
            "output_dir": output_dir,
            "photo_count": len(photos),
            "created_at": time.time(),
        }

    # 提交到线程池
    executor.submit(_run_generation, job_id, primary_photo_path, info, model, output_dir)
    logger.info(f"[{job_id[:8]}] 任务已创建: {product_name} ({len(photos)} 张原图)")

    return {"job_id": job_id, "status": "pending", "photo_count": len(photos)}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """查询任务进度"""
    job = get_job(job_id)
    if not job:
        return {"error": "任务不存在", "job_id": job_id}

    resp = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", ""),
        "product_name": job.get("product_name", ""),
    }
    if job["status"] == "generating":
        resp["current_slot"] = job.get("current_slot", "")
    if job["status"] == "failed":
        resp["error"] = job.get("error", "未知错误")
    return resp


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """获取生成结果（图片列表 + 评分 + 费用）"""
    job = get_job(job_id)
    if not job:
        return {"error": "任务不存在", "job_id": job_id}
    if job["status"] != "done":
        return {
            "error": f"任务尚未完成，当前状态: {job['status']}",
            "status": job["status"],
        }

    return {
        "job_id": job_id,
        "status": "done",
        "product_name": job.get("product_name", ""),
        "files": job.get("files", []),
        "scores": job.get("scores", {}),
        "passed": job.get("passed", False),
        "failures": job.get("failures", []),
        "suggestions": job.get("suggestions", []),
        "cost_usd": job.get("cost_usd", 0),
        "generated_count": job.get("generated_count", 0),
        "failed_count": job.get("failed_count", 0),
    }


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """下载单张图片"""
    job = get_job(job_id)
    if not job:
        return {"error": "任务不存在"}

    filepath = os.path.join(job["output_dir"], filename)
    if not os.path.isfile(filepath):
        return {"error": f"文件不存在: {filename}"}

    return FileResponse(
        filepath,
        media_type="image/jpeg",
        filename=filename,
    )


@app.get("/api/download/{job_id}")
async def download_all(job_id: str):
    """打包下载全部生成图片 (ZIP)"""
    job = get_job(job_id)
    if not job:
        return {"error": "任务不存在"}

    output_dir = job["output_dir"]
    if not os.path.isdir(output_dir):
        return {"error": "输出目录不存在"}

    # 在内存中创建 ZIP（只打包生成的图，不包含 source_ 原图）
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(output_dir)):
            if (f.lower().endswith((".jpg", ".jpeg", ".png"))
                    and not f.startswith("_")
                    and not f.startswith("source_")):
                zf.write(os.path.join(output_dir, f), f)

    buf.seek(0)
    zip_name = f"{job.get('product_name', 'images')}_{job_id[:8]}.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.get("/api/models")
async def list_models():
    """返回可用 AI 模型列表"""
    from config import COST_ESTIMATES
    models = []
    for name, costs in COST_ESTIMATES.items():
        models.append({
            "id": name,
            "name": {
                "siliconflow": "SiliconFlow (FLUX.1-schnell, 免费)",
                "gpt-4o": "GPT-4o (DALL-E 3, 付费)",
                "flux": "Flux Pro 1.1 (BFL API, 付费)",
            }.get(name, name),
            "cost_per_image": costs.get("text_to_image", 0),
            "supports_reference": name != "siliconflow",
        })
    return {"models": models}


# ---------------------------------------------------------------------------
# 首页 → 重定向到前端 UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return '<html><head><meta http-equiv="refresh" content="0;url=/static/index.html"></head></html>'


# ---------------------------------------------------------------------------
# 静态文件 (放在路由之后，避免覆盖 API 路由)
# ---------------------------------------------------------------------------

static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("启动 FastAPI 服务...")
    logger.info("前端页面: http://localhost:8000")
    logger.info("API 文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
