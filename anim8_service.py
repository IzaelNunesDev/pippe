"""
Anim8 Job Service & Background Task Manager
===========================================
Gerencia o processamento assíncrono de vetorização de vídeos,
notificações em tempo real (SSE) e manipulação dinâmica de paleta.
"""

import os
import time
import uuid
import queue
import shutil
import tempfile
import threading
from typing import Dict, Any, Optional, List
from anim8_backend_engine import VectorizerConfig, generate_demo_video
from anim8_advanced_morph_stacking import Anim8ProEngine


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Anim8Job:
    def __init__(self, job_id: str, config: VectorizerConfig, use_stacking: bool = True, use_morphing: bool = True):
        self.job_id = job_id
        self.config = config
        self.use_stacking = use_stacking
        self.use_morphing = use_morphing
        self.status = JobStatus.QUEUED
        self.progress = 0
        self.stage = "Na fila..."
        self.details: Dict[str, Any] = {}
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()

    def update_progress(self, pct: int, stage: str, details: Optional[Dict[str, Any]] = None):
        with self._lock:
            self.progress = pct
            self.stage = stage
            if details:
                self.details.update(details)
            msg = {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "stage": self.stage,
                "details": self.details,
            }
            # Notifica os ouvintes do SSE
            dead_subs = []
            for q in self.subscribers:
                try:
                    q.put_nowait(msg)
                except Exception:
                    dead_subs.append(q)
            for q in dead_subs:
                self.subscribers.remove(q)

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=100)
        with self._lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


class Anim8Service:
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.join(tempfile.gettempdir(), "anim8_jobs")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.jobs: Dict[str, Anim8Job] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        video_bytes: bytes,
        config: VectorizerConfig,
        use_stacking: bool = True,
        use_morphing: bool = True,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Anim8Job(job_id, config, use_stacking, use_morphing)

        with self._lock:
            self.jobs[job_id] = job

        job_dir = os.path.join(self.storage_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        video_path = os.path.join(job_dir, "input.mp4")

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Inicia processamento em thread separada
        thread = threading.Thread(
            target=self._run_vectorization_worker,
            args=(job, video_path, job_dir),
            daemon=True,
        )
        thread.start()

        return job_id

    def create_demo_job(
        self,
        config: VectorizerConfig,
        use_stacking: bool = True,
        use_morphing: bool = True,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Anim8Job(job_id, config, use_stacking, use_morphing)

        with self._lock:
            self.jobs[job_id] = job

        job_dir = os.path.join(self.storage_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        video_path = os.path.join(job_dir, "sample_ai_mascot.mp4")

        # Gera o mascote sintético procedural
        generate_demo_video(video_path)

        thread = threading.Thread(
            target=self._run_vectorization_worker,
            args=(job, video_path, job_dir),
            daemon=True,
        )
        thread.start()

        return job_id

    def get_job(self, job_id: str) -> Optional[Anim8Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def _run_vectorization_worker(self, job: Anim8Job, video_path: str, job_dir: str):
        try:
            job.status = JobStatus.PROCESSING
            job.update_progress(2, "Iniciando engine Anim8 Pro...")

            engine = Anim8ProEngine(job.config)
            svg_out = os.path.join(job_dir, "animation.svg")
            lottie_out = os.path.join(job_dir, "animation.json")

            def on_progress(pct: int, stage: str, details: Optional[Dict] = None):
                job.update_progress(pct, stage, details)

            result = engine.process_video_advanced(
                video_path=video_path,
                output_svg_path=svg_out,
                output_lottie_path=lottie_out,
                use_stacking=job.use_stacking,
                use_morphing=job.use_morphing,
                progress_callback=on_progress,
            )

            # Também guardamos o caminho do vídeo original para preview / split-screen
            result["original_video_path"] = video_path

            job.result = result
            job.status = JobStatus.COMPLETED
            job.update_progress(100, "Vetorização concluída com sucesso!")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.update_progress(job.progress, f"Erro: {str(e)}")

    def recolor_svg(self, original_svg: str, palette_map: Dict[int, str]) -> str:
        """
        Substitui as variáveis CSS :root { --color-i: ... } no código SVG.
        """
        updated_svg = original_svg
        for idx, new_hex in palette_map.items():
            var_name = f"--color-{idx}"
            # Procura --color-X: #xxxxxx; e substitui
            import re
            pattern = re.compile(rf"{var_name}\s*:\s*#[0-9a-fA-F]{{3,8}}\s*;")
            updated_svg = pattern.sub(f"{var_name}: {new_hex};", updated_svg)
        return updated_svg
