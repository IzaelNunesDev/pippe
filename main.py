"""
Anim8 Studio FastAPI Server
===========================
Backend REST & SSE API + Web Application Server para o Editor Vetorial Anim8.
Execução:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import asyncio
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from anim8_backend_engine import VectorizerConfig
from anim8_service import Anim8Service, JobStatus

app = FastAPI(
    title="Anim8 Studio API",
    description="Conversor de Vídeo para SVG Animado e Lottie com Empilhamento Topológico e Morphing",
    version="2.0.0"
)

# Habilitar CORS para permitir consumo do frontend e integrações
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o serviço de jobs
service = Anim8Service()

# Diretório estático do frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RecolorRequest(BaseModel):
    palette: Dict[str, str]  # ex: {"0": "#ff0000", "1": "#00ff00"}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Anim8 Studio: Frontend em construção...</h1>")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0", "engine": "Anim8ProEngine"}


@app.post("/api/vectorize")
async def vectorize_video(
    file: UploadFile = File(...),
    num_colors: int = Form(8),
    detail_level: str = Form("balanced"),
    smoothing: float = Form(40.0),
    path_budget: int = Form(128),
    target_fps: int = Form(30),
    remove_background: bool = Form(False),
    use_stacking: bool = Form(True),
    use_morphing: bool = Form(True),
    max_dimension: int = Form(512),
):
    """
    Inicia o processamento de vetorização assíncrono para o vídeo enviado.
    Retorna imediatamente o job_id para acompanhamento em tempo real.
    """
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo de vídeo vazio.")

    config = VectorizerConfig(
        num_colors=num_colors,
        detail_level=detail_level,
        smoothing=smoothing,
        path_budget=path_budget,
        target_fps=target_fps,
        remove_background=remove_background,
        max_dimension=max_dimension,
    )

    job_id = service.create_job(
        video_bytes=content,
        config=config,
        use_stacking=use_stacking,
        use_morphing=use_morphing,
    )

    return {"job_id": job_id, "status": "queued", "message": "Vídeo recebido com sucesso."}


@app.post("/api/demo")
async def generate_demo_mascot(
    num_colors: int = 8,
    detail_level: str = "balanced",
    smoothing: float = 40.0,
    path_budget: int = 128,
    target_fps: int = 24,
    remove_background: bool = False,
    use_stacking: bool = True,
    use_morphing: bool = True,
):
    """
    Gera um mascote animado procedural de teste e executa a vetorização diretamente.
    Permite validar a aplicação em 1 clique sem precisar subir arquivos locais.
    """
    config = VectorizerConfig(
        num_colors=num_colors,
        detail_level=detail_level,
        smoothing=smoothing,
        path_budget=path_budget,
        target_fps=target_fps,
        remove_background=remove_background,
    )

    job_id = service.create_demo_job(
        config=config,
        use_stacking=use_stacking,
        use_morphing=use_morphing,
    )

    return {"job_id": job_id, "status": "queued", "message": "Demo procedural gerada."}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Retorna o status atual do job, métricas de processamento e dados do vetor se concluído.
    """
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")

    response = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "details": job.details,
        "error": job.error,
    }

    if job.status == JobStatus.COMPLETED and job.result:
        # Inclui os dados completos do resultado
        response["result"] = {
            "frames_processed": job.result["frames_processed"],
            "total_shapes": job.result["total_shapes"],
            "palette_hex": job.result["palette_hex"],
            "canvas_size": job.result["canvas_size"],
            "target_fps": job.result["target_fps"],
            "duration": job.result["duration"],
            "use_stacking": job.result["use_stacking"],
            "use_morphing": job.result["use_morphing"],
            "frame_stats": job.result["frame_stats"],
            "svg_content": job.result["svg_content"],
            "lottie_content": job.result["lottie_content"],
            "shapes_by_frame": job.result["shapes_by_frame"],
        }

    return response


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request):
    """
    Endpoint Server-Sent Events (SSE) para atualização fluida da barra de progresso.
    """
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")

    queue_sub = job.subscribe()

    async def event_generator():
        try:
            # Envia status inicial imediatamente
            initial_data = json.dumps({
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "stage": job.stage,
                "details": job.details
            })
            yield f"data: {initial_data}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    # Checagem não bloqueante com timeout curto
                    msg = queue_sub.get(block=False)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("status") in (JobStatus.COMPLETED, JobStatus.FAILED):
                        break
                except Exception:
                    await asyncio.sleep(0.15)

                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    break

        finally:
            job.unsubscribe(queue_sub)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/jobs/{job_id}/svg")
async def download_svg(job_id: str):
    """
    Download direto do arquivo SVG Animado.
    """
    job = service.get_job(job_id)
    if not job or not job.result or not job.result.get("svg_path"):
        raise HTTPException(status_code=404, detail="Arquivo SVG indisponível.")

    return FileResponse(
        path=job.result["svg_path"],
        media_type="image/svg+xml",
        filename=f"anim8_{job_id}.svg"
    )


@app.get("/api/jobs/{job_id}/lottie")
async def download_lottie(job_id: str):
    """
    Download direto da especificação Lottie JSON.
    """
    job = service.get_job(job_id)
    if not job or not job.result or not job.result.get("lottie_path"):
        raise HTTPException(status_code=404, detail="Arquivo Lottie indisponível.")

    return FileResponse(
        path=job.result["lottie_path"],
        media_type="application/json",
        filename=f"anim8_{job_id}.json"
    )


@app.get("/api/jobs/{job_id}/video")
async def get_original_video(job_id: str):
    """
    Serve o vídeo original para visualização comparativa (Split-Screen) no frontend.
    """
    job = service.get_job(job_id)
    if not job or not job.result or not job.result.get("original_video_path"):
        raise HTTPException(status_code=404, detail="Vídeo original não encontrado.")

    return FileResponse(
        path=job.result["original_video_path"],
        media_type="video/mp4"
    )


@app.post("/api/jobs/{job_id}/recolor")
async def recolor_animation(job_id: str, req: RecolorRequest):
    """
    Altera as cores no SVG em tempo de execução via CSS Variables.
    """
    job = service.get_job(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Job não encontrado ou incompleto.")

    palette_map = {int(k): v for k, v in req.palette.items()}
    new_svg = service.recolor_svg(job.result["svg_content"], palette_map)

    # Atualiza o arquivo em disco se existir
    if job.result.get("svg_path") and os.path.exists(job.result["svg_path"]):
        with open(job.result["svg_path"], "w", encoding="utf-8") as f:
            f.write(new_svg)

    return {"status": "ok", "svg": new_svg}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
