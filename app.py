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

import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional
from threading import Lock

from fastapi import FastAPI, File, Form, Request, UploadFile
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
                    model: str | None, output_dir: str,
                    candidates_per_slot: int = 2):
    """在线程池中运行的生成任务（多候选模式）"""
    try:
        update_job(job_id, status="generating", progress="0/8")

        # Step 1: 生成 prompt 计划
        from prompt_engine import generate_prompt_plan
        prompt_plan = generate_prompt_plan(product_info)
        logger.info(f"[{job_id[:8]}] Prompt 计划已生成: {len(prompt_plan)} 张")

        # Step 2: 为每个槽位生成 N 张候选
        from image_generator import generate_product_images
        from image_validator import validate_single_image

        total_tasks = len(prompt_plan) * candidates_per_slot
        all_generated = []
        all_failed = []
        total_cost = 0.0
        task_idx = 0

        # candidates_map: slot -> [{filename, score, passed, details, reasons}, ...]
        candidates_map = {}

        for item in prompt_plan:
            slot = item["slot"]
            img_type = item["type"]
            original_filename = item["filename"]
            base_name, ext = os.path.splitext(original_filename)

            slot_candidates = []

            for c_idx in range(1, candidates_per_slot + 1):
                task_idx += 1
                candidate_filename = f"{base_name}_c{c_idx}{ext}"
                update_job(job_id, progress=f"{task_idx}/{total_tasks}",
                           current_slot=f"{slot} (候选 {c_idx}/{candidates_per_slot})")

                # 构造带候选文件名的 plan item
                candidate_item = deepcopy(item)
                candidate_item["filename"] = candidate_filename

                result = generate_product_images(
                    prompt_plan=[candidate_item],
                    output_dir=output_dir,
                    source_photo=photo_path,
                    model=model,
                )
                all_generated.extend(result["generated"])
                all_failed.extend(result["failed"])
                total_cost += result["stats"]["total_cost_usd"]

                # 对生成的候选图片进行单张质量验证
                candidate_path = os.path.join(output_dir, candidate_filename)
                if os.path.exists(candidate_path):
                    val = validate_single_image(candidate_path, img_type)
                    slot_candidates.append({
                        "filename": candidate_filename,
                        "candidate_index": c_idx,
                        "score": val["score"],
                        "passed": val["passed"],
                        "details": val["details"],
                        "reasons": val["reasons"],
                        "size_kb": round(os.path.getsize(candidate_path) / 1024, 1),
                    })
                else:
                    slot_candidates.append({
                        "filename": candidate_filename,
                        "candidate_index": c_idx,
                        "score": 0.0,
                        "passed": False,
                        "details": {},
                        "reasons": ["生成失败"],
                        "size_kb": 0,
                    })

            candidates_map[slot] = {
                "type": img_type,
                "original_filename": original_filename,
                "candidates": slot_candidates,
            }

        update_job(job_id, progress=f"{total_tasks}/{total_tasks}")

        if not all_generated:
            update_job(job_id, status="failed", error="没有成功生成任何图片")
            return

        # Step 3: 进入选择阶段（不再直接 done）
        update_job(
            job_id,
            status="selecting",
            progress="等待选择",
            candidates=candidates_map,
            cost_usd=round(total_cost, 4),
            generated_count=len(all_generated),
            failed_count=len(all_failed),
        )
        logger.info(f"[{job_id[:8]}] 候选生成完成, 等待用户选择")

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
    candidates_per_slot: int = Form(None, description="每个槽位生成候选数 (默认2)"),
):
    """上传实拍图 + 商品信息 → 启动后台生成任务（1+7 组图，每个槽位多候选）"""
    # 解析 product_info JSON
    try:
        info = json.loads(product_info)
    except json.JSONDecodeError as e:
        return {"error": f"product_info JSON 格式错误: {e}"}

    if not info.get("name"):
        return {"error": "product_info 必须包含 name 字段"}

    if not photos:
        return {"error": "请至少上传一张商品实拍图"}

    # 候选数量
    from config import CANDIDATES_PER_SLOT
    n_candidates = candidates_per_slot if candidates_per_slot and candidates_per_slot >= 1 else CANDIDATES_PER_SLOT

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
            "candidates_per_slot": n_candidates,
            "created_at": time.time(),
        }

    # 提交到线程池
    executor.submit(_run_generation, job_id, primary_photo_path, info, model,
                    output_dir, n_candidates)
    logger.info(f"[{job_id[:8]}] 任务已创建: {product_name} "
                f"({len(photos)} 张原图, 每槽 {n_candidates} 候选)")

    return {"job_id": job_id, "status": "pending", "photo_count": len(photos),
            "candidates_per_slot": n_candidates}


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
    if job["status"] == "selecting":
        resp["candidates_per_slot"] = job.get("candidates_per_slot", 2)
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


@app.get("/api/candidates/{job_id}")
async def get_candidates(job_id: str):
    """获取所有候选图片（按 slot 分组），含质量分数"""
    job = get_job(job_id)
    if not job:
        return {"error": "任务不存在", "job_id": job_id}
    if job["status"] not in ("selecting", "done"):
        return {
            "error": f"候选尚未就绪，当前状态: {job['status']}",
            "status": job["status"],
        }

    candidates = job.get("candidates", {})
    return {
        "job_id": job_id,
        "status": job["status"],
        "candidates": candidates,
        "cost_usd": job.get("cost_usd", 0),
        "generated_count": job.get("generated_count", 0),
        "failed_count": job.get("failed_count", 0),
    }


@app.post("/api/select/{job_id}")
async def select_candidates(job_id: str, request: Request):
    """
    用户提交选择：每个 slot 选一张候选图片。
    请求体 JSON: {"selections": {"main": "main_c1.jpg", "PT01_lifestyle": "PT01_lifestyle_c2.jpg", ...}}
    选中图片复制为最终文件名，运行整体验证，状态变为 done。
    """
    job = get_job(job_id)
    if not job:
        return {"error": "任务不存在", "job_id": job_id}
    if job["status"] != "selecting":
        return {"error": f"当前状态不允许选择: {job['status']}"}

    body = await request.json()
    selections = body.get("selections", {})
    if not selections:
        return {"error": "selections 不能为空"}

    output_dir = job["output_dir"]
    candidates = job.get("candidates", {})

    # 验证选择并复制为最终文件名
    for slot, slot_data in candidates.items():
        selected_file = selections.get(slot)
        if not selected_file:
            # 自动选择分数最高的候选
            best = max(slot_data["candidates"], key=lambda c: c["score"])
            selected_file = best["filename"]

        src_path = os.path.join(output_dir, selected_file)
        if not os.path.exists(src_path):
            return {"error": f"候选文件不存在: {selected_file}"}

        # 复制为最终文件名 (original_filename)
        dst_path = os.path.join(output_dir, slot_data["original_filename"])
        shutil.copy2(src_path, dst_path)

    # 运行整体验证
    update_job(job_id, status="validating", progress="最终评分中...")
    from image_validator import validate_and_fix
    product_name = job.get("product_name", "generated")
    val_result = validate_and_fix(output_dir, product_name)

    # 汇总最终图片（排除候选 _c*.jpg 和 source_）
    files = []
    for f in sorted(os.listdir(output_dir)):
        if (f.lower().endswith((".jpg", ".jpeg", ".png"))
                and not f.startswith("_")
                and not f.startswith("source_")
                and "_c" not in f):
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
        selections=selections,
    )
    logger.info(f"[{job_id[:8]}] 用户选择完成, 评分: "
                f"{val_result['scores'].get('normalized_score', 'N/A')}")

    return {
        "job_id": job_id,
        "status": "done",
        "files": files,
        "scores": val_result["scores"],
        "passed": val_result["passed"],
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

    # 在内存中创建 ZIP（只打包最终图，不包含 source_ 原图和候选 _c 图）
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(output_dir)):
            if (f.lower().endswith((".jpg", ".jpeg", ".png"))
                    and not f.startswith("_")
                    and not f.startswith("source_")
                    and "_c" not in f):
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
# 自动部署 (GitHub Webhook)
# ---------------------------------------------------------------------------

DEPLOY_SECRET = os.environ.get("DEPLOY_SECRET", "amazon-image-optimizer-deploy-2026")


def _verify_github_signature(payload: bytes, signature: str) -> bool:
    """验证 GitHub Webhook 签名"""
    if not signature:
        return False
    mac = hmac.new(DEPLOY_SECRET.encode(), payload, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/api/deploy")
async def deploy(request: Request):
    """GitHub Webhook: push 事件触发自动部署"""
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not _verify_github_signature(body, signature):
        return {"error": "签名验证失败"}, 403

    # 只处理 push 事件
    event = request.headers.get("X-GitHub-Event", "")
    if event != "push":
        return {"message": f"忽略事件: {event}"}

    logger.info("收到 GitHub push 事件，开始自动部署...")

    try:
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=BASE_DIR,
            capture_output=True, text=True, timeout=30,
        )
        logger.info(f"git pull: {result.stdout.strip()}")
        if result.returncode != 0:
            logger.error(f"git pull 失败: {result.stderr}")
            return {"error": f"git pull 失败: {result.stderr}"}
    except Exception as e:
        logger.error(f"git pull 异常: {e}")
        return {"error": str(e)}

    # 重启自身 (systemd 会自动拉起)
    logger.info("代码已更新，正在重启服务...")
    os._exit(0)


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
