"""
Anim8-Style Video to Animated Vector (SVG & Lottie) Engine
===========================================================
Backend processing pipeline in Python covering:
1. Video Frame Extraction & Resampling (OpenCV)
2. Background Removal via Border Flood-Fill & LAB Color Distance
3. Temporal-Consistent Global Color Quantization (MiniBatchKMeans in CIELAB space)
4. Hierarchical Contour Extraction with Holes (RETR_CCOMP) & Path Budget Filtering
5. Polygon Simplification (Ramer-Douglas-Peucker) & Least-Squares Cubic Bezier Curve Fitting (Schneider's Algorithm)
6. Temporal Shape Tracking (IoU + Centroid Matching) & Animated SVG / Lottie JSON Export

Dependencies:
    pip install numpy opencv-python scikit-learn scipy
"""

import cv2
import json
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import MiniBatchKMeans


# =====================================================================
# CONFIGURAÇÃO DA ENGINE (Espelhando os parâmetros da UI do Anim8)
# =====================================================================
@dataclass
class VectorizerConfig:
    num_colors: int = 8              # "Cores": Quantidade de cores na paleta global (2 a 32)
    detail_level: str = "balanced"   # "Detalhe": 'low', 'balanced', 'high' (controla área mínima)
    smoothing: float = 40.0          # "Suavização": 0 a 100 (controla RDP + erro do ajuste de Bézier)
    path_budget: int = 128           # "Orçamento do caminho": Máximo de shapes por frame
    target_fps: int = 30             # "Taxa de quadros": FPS de saída da animação
    remove_background: bool = False  # "Antecedentes": Desligado (True = remove) / Sobre (False = mantém)
    bg_tolerance: float = 18.0       # Tolerância no espaço LAB para remoção de fundo nas bordas
    max_dimension: int = 512         # Resolução máxima de processamento para manter performance

    @property
    def min_region_area(self) -> int:
        mapping = {"low": 64, "balanced": 20, "high": 6}
        return mapping.get(self.detail_level, 20)

    @property
    def bezier_error_threshold(self) -> float:
        # Mapeia suavização (0..100) para erro quadrático tolerado em pixels (1.0 .. 25.0)
        return 1.0 + (self.smoothing / 100.0) * 24.0

    @property
    def rdp_epsilon(self) -> float:
        # Simplificação prévia de vértices colineares
        return 0.5 + (self.smoothing / 100.0) * 2.5


@dataclass
class VectorShape:
    color_index: int
    hex_color: str
    svg_path_d: str
    area: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[float, float]
    bezier_subpaths: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = field(default_factory=list)
    track_id: int = -1


# =====================================================================
# MATEMÁTICA GEOMÉTRICA: AJUSTE DE CURVAS DE BÉZIER CÚBICAS (SCHNEIDER)
# =====================================================================
class CubicBezierFitter:
    """
    Implementação vetorial do algoritmo de Philip J. Schneider ("An Algorithm for
    Automatically Fitting Digitized Curves", Graphics Gems) para converter polígonos
    de pixels em curvas de Bézier cúbicas suaves e editáveis.
    """

    @staticmethod
    def fit_closed_contour(points: np.ndarray, max_error: float, rdp_eps: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Recebe um contorno fechado (N, 2) e retorna uma lista de segmentos de Bézier Cúbicos:
        [(P0, P1, P2, P3), ...]
        """
        if len(points) < 3:
            return []

        # 1. Simplificação poligonal rápida via Ramer-Douglas-Peucker para remover ruído de escadinha (pixel staircase)
        approx = cv2.approxPolyDP(points.reshape(-1, 1, 2).astype(np.float32), epsilon=rdp_eps, closed=True)
        pts = approx.reshape(-1, 2).astype(np.float64)

        if len(pts) < 3:
            pts = points.astype(np.float64)

        # Garante fechamento do contorno
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

        # Reamostra ao longo do perímetro para permitir ajuste suave de mínimos quadrados
        dense_pts = CubicBezierFitter._densify_polyline(pts, step=2.0)
        if len(dense_pts) < 4:
            p0, p3 = pts[0], pts[-1]
            return [(p0, p0 + (p3 - p0) / 3.0, p0 + 2.0 * (p3 - p0) / 3.0, p3)]

        # Tangentes nas extremidades do contorno fechado
        left_tangent = CubicBezierFitter._normalize(dense_pts[1] - dense_pts[-2])
        right_tangent = -left_tangent

        return CubicBezierFitter._fit_cubic(dense_pts, left_tangent, right_tangent, max_error)

    @staticmethod
    def _densify_polyline(pts: np.ndarray, step: float = 2.0) -> np.ndarray:
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p_start, p_end = pts[i], pts[i + 1]
            dist = np.linalg.norm(p_end - p_start)
            if dist > step:
                num_steps = int(math.ceil(dist / step))
                for s in range(1, num_steps + 1):
                    t = s / num_steps
                    out.append(p_start * (1.0 - t) + p_end * t)
            elif dist > 1e-4:
                out.append(p_end)
        return np.array(out, dtype=np.float64)

    @staticmethod
    def _fit_cubic(pts: np.ndarray, t_hat1: np.ndarray, t_hat2: np.ndarray, error_thresh: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        n_pts = len(pts)
        if n_pts == 2:
            dist = np.linalg.norm(pts[1] - pts[0]) / 3.0
            p0, p3 = pts[0], pts[1]
            return [(p0, p0 + t_hat1 * dist, p3 + t_hat2 * dist, p3)]

        # Parametrização por comprimento de corda (Chord-Length Parameterization)
        u = CubicBezierFitter._chord_length_parameterize(pts)
        bez_curve = CubicBezierFitter._generate_bezier(pts, u, t_hat1, t_hat2)
        max_err, split_point = CubicBezierFitter._compute_max_error(pts, bez_curve, u)

        if max_err < error_thresh:
            return [bez_curve]

        # Se o erro for moderado, refina os parâmetros u via Newton-Raphson
        if max_err < error_thresh * 4.0:
            for _ in range(4):
                u_prime = CubicBezierFitter._reparameterize(pts, u, bez_curve)
                bez_curve = CubicBezierFitter._generate_bezier(pts, u_prime, t_hat1, t_hat2)
                max_err, split_point = CubicBezierFitter._compute_max_error(pts, bez_curve, u_prime)
                if max_err < error_thresh:
                    return [bez_curve]
                u = u_prime

        # Subdivisão recursiva no ponto de erro máximo
        split_point = max(1, min(split_point, n_pts - 2))
        t_center = CubicBezierFitter._normalize(pts[split_point - 1] - pts[split_point + 1])

        left_curves = CubicBezierFitter._fit_cubic(pts[:split_point + 1], t_hat1, t_center, error_thresh)
        right_curves = CubicBezierFitter._fit_cubic(pts[split_point:], -t_center, t_hat2, error_thresh)
        return left_curves + right_curves

    @staticmethod
    def _generate_bezier(pts: np.ndarray, u: np.ndarray, t_hat1: np.ndarray, t_hat2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p0, p3 = pts[0], pts[-1]
        b0 = (1.0 - u) ** 3
        b1 = 3.0 * u * ((1.0 - u) ** 2)
        b2 = 3.0 * (u ** 2) * (1.0 - u)
        b3 = u ** 3

        a1 = np.outer(b1, t_hat1)
        a2 = np.outer(b2, t_hat2)

        c11 = np.sum(a1 * a1)
        c12 = np.sum(a1 * a2)
        c22 = np.sum(a2 * a2)

        tmp = pts - (np.outer(b0 + b1, p0) + np.outer(b2 + b3, p3))
        x1 = np.sum(a1 * tmp)
        x2 = np.sum(a2 * tmp)

        det_c1_c2 = c11 * c22 - c12 * c12
        det_c1_x = c11 * x2 - c12 * x1
        det_x_c2 = x1 * c22 - x2 * c12

        alpha_l = det_x_c2 / det_c1_c2 if abs(det_c1_c2) > 1e-6 else 0.0
        alpha_r = det_c1_x / det_c1_c2 if abs(det_c1_c2) > 1e-6 else 0.0

        seg_length = np.linalg.norm(p3 - p0)
        epsilon = 1.0e-6 * seg_length
        if alpha_l < epsilon or alpha_r < epsilon:
            dist = seg_length / 3.0
            return (p0, p0 + t_hat1 * dist, p3 + t_hat2 * dist, p3)

        return (p0, p0 + t_hat1 * alpha_l, p3 + t_hat2 * alpha_r, p3)

    @staticmethod
    def _reparameterize(pts: np.ndarray, u: np.ndarray, bez: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        p0, p1, p2, p3 = bez
        u_prime = np.copy(u)
        for i in range(1, len(pts) - 1):
            t = u[i]
            q_u = ((1 - t)**3)*p0 + 3*t*((1 - t)**2)*p1 + 3*(t**2)*(1 - t)*p2 + (t**3)*p3
            q1_u = 3*((1 - t)**2)*(p1 - p0) + 6*(1 - t)*t*(p2 - p1) + 3*(t**2)*(p3 - p2)
            q2_u = 6*(1 - t)*(p2 - 2*p1 + p0) + 6*t*(p3 - 2*p2 + p1)

            numerator = np.dot(q_u - pts[i], q1_u)
            denominator = np.dot(q1_u, q1_u) + np.dot(q_u - pts[i], q2_u)
            if abs(denominator) > 1e-8:
                u_prime[i] = np.clip(t - numerator / denominator, u_prime[i - 1] + 1e-5, 1.0 - 1e-5)
        return u_prime

    @staticmethod
    def _chord_length_parameterize(pts: np.ndarray) -> np.ndarray:
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(dists)])
        total = cum[-1]
        return cum / total if total > 1e-8 else np.linspace(0.0, 1.0, len(pts))

    @staticmethod
    def _compute_max_error(pts: np.ndarray, bez: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], u: np.ndarray) -> Tuple[float, int]:
        p0, p1, p2, p3 = bez
        b0 = (1.0 - u) ** 3
        b1 = 3.0 * u * ((1.0 - u) ** 2)
        b2 = 3.0 * (u ** 2) * (1.0 - u)
        b3 = u ** 3
        eval_pts = np.outer(b0, p0) + np.outer(b1, p1) + np.outer(b2, p2) + np.outer(b3, p3)
        sq_dists = np.sum((eval_pts - pts) ** 2, axis=1)
        split_idx = int(np.argmax(sq_dists))
        return float(sq_dists[split_idx]), split_idx

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else np.array([1.0, 0.0])

    @staticmethod
    def curves_to_svg_path(curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> str:
        if not curves:
            return ""
        p0 = curves[0][0]
        cmds = [f"M{p0[0]:.2f},{p0[1]:.2f}"]
        for _, p1, p2, p3 in curves:
            cmds.append(f"C{p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}")
        cmds.append("Z")
        return " ".join(cmds)


# =====================================================================
# ENGINE PRINCIPAL: VÍDEO -> QUANTIZAÇÃO GLOBAL -> TRAÇADO -> SVG/LOTTIE
# =====================================================================
class Anim8Engine:
    def __init__(self, config: VectorizerConfig):
        self.config = config
        self.palette_lab: Optional[np.ndarray] = None
        self.palette_rgb: Optional[np.ndarray] = None
        self.palette_hex: List[str] = []
        self.bg_color_index: Optional[int] = None

    # -----------------------------------------------------------------
    # ETAPA 1: Decodificação e Extração de Quadros do Vídeo
    # -----------------------------------------------------------------
    def extract_video_frames(self, video_path: str) -> Tuple[List[np.ndarray], Tuple[int, int]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1.0, src_fps / self.config.target_fps)

        frames_rgb = []
        frame_idx = 0
        next_target = 0.0
        target_size = (0, 0)

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx >= int(round(next_target)):
                h, w = frame_bgr.shape[:2]
                scale = min(1.0, self.config.max_dimension / max(h, w))
                if scale < 1.0:
                    new_w, new_h = int(round(w * scale)), int(round(h * scale))
                    frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                target_size = (frame_bgr.shape[1], frame_bgr.shape[0])
                # Filtro bilateral leve para reduzir granulação de IA preservando bordas nítidas
                filtered = cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=25, sigmaSpace=25)
                frames_rgb.append(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
                next_target += frame_interval

            frame_idx += 1

        cap.release()
        return frames_rgb, target_size

    # -----------------------------------------------------------------
    # ETAPA 2: Paleta Global Temporalmente Consistente (CIELAB K-Means)
    # -----------------------------------------------------------------
    def compute_global_palette(self, frames_rgb: List[np.ndarray]) -> None:
        """
        Amostra pixels de todos os quadros simultaneamente no espaço perceptual CIELAB.
        Isso garante que as cores não mudem de um frame para o outro (zero flickering).
        """
        sampled_pixels_lab = []
        step = max(1, len(frames_rgb) // 15)
        for frame in frames_rgb[::step]:
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            pixels = lab[::2, ::2].reshape(-1, 3)
            sampled_pixels_lab.append(pixels)

        all_samples = np.vstack(sampled_pixels_lab).astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=self.config.num_colors,
            random_state=42,
            batch_size=4096,
            n_init=3
        )
        kmeans.fit(all_samples)

        self.palette_lab = kmeans.cluster_centers_.astype(np.float32)

        # Converte centros LAB de volta para RGB e Hex
        centers_lab_uint8 = np.clip(np.round(self.palette_lab), 0, 255).astype(np.uint8).reshape(1, -1, 3)
        centers_rgb = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2RGB).reshape(-1, 3)
        self.palette_rgb = centers_rgb
        self.palette_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers_rgb]

        # Identifica qual índice da paleta representa o fundo (analisando as bordas do primeiro frame)
        if self.config.remove_background and frames_rgb:
            first_quant = self.quantize_frame(frames_rgb[0])
            border_pixels = np.concatenate([
                first_quant[0, :], first_quant[-1, :],
                first_quant[:, 0], first_quant[:, -1]
            ])
            counts = np.bincount(border_pixels, minlength=self.config.num_colors)
            self.bg_color_index = int(np.argmax(counts))

    def quantize_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Mapeia cada pixel do frame para o índice da cor mais próxima na paleta global CIELAB
        e aplica limpeza morfológica (despeckling).
        """
        h, w = frame_rgb.shape[:2]
        frame_lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

        dists = np.linalg.norm(frame_lab[:, None, :] - self.palette_lab[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1).reshape(h, w).astype(np.uint8)

        labels_clean = cv2.medianBlur(labels, 3)
        return labels_clean

    # -----------------------------------------------------------------
    # ETAPA 3: Vetorização por Camada, Hierarquia de Furos e Path Budget
    # -----------------------------------------------------------------
    def trace_frame(self, label_map: np.ndarray) -> List[VectorShape]:
        """
        Extrai contornos externos e furos internos (RETR_CCOMP) para cada cor,
        respeita o Path Budget e converte os polígonos em curvas de Bézier cúbicas.
        """
        candidate_regions = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for color_idx in range(self.config.num_colors):
            if self.config.remove_background and color_idx == self.bg_color_index:
                continue

            mask = np.where(label_map == color_idx, 255, 0).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # RETR_CCOMP organiza contornos em 2 níveis: [0] Contorno Externo, [1] Furos Internos
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if hierarchy is None:
                continue

            hierarchy = hierarchy[0]
            for idx, (Next, Prev, First_Child, Parent) in enumerate(hierarchy):
                if Parent != -1:
                    continue

                outer_cnt = contours[idx]
                area = cv2.contourArea(outer_cnt)
                if area < self.config.min_region_area:
                    continue

                holes = []
                child_idx = First_Child
                while child_idx != -1:
                    hole_cnt = contours[child_idx]
                    if cv2.contourArea(hole_cnt) >= self.config.min_region_area * 0.5:
                        holes.append(hole_cnt)
                    child_idx = hierarchy[child_idx][0]

                candidate_regions.append((area, color_idx, outer_cnt, holes))

        # Aplica o "Orçamento do caminho" (Path Budget): mantém os N maiores shapes do frame
        candidate_regions.sort(key=lambda x: x[0], reverse=True)
        selected_regions = candidate_regions[:self.config.path_budget]

        shapes: List[VectorShape] = []
        for area, color_idx, outer_cnt, holes in selected_regions:
            subpath_strings = []
            bezier_subpaths = []

            outer_pts = outer_cnt.reshape(-1, 2)
            outer_beziers = CubicBezierFitter.fit_closed_contour(
                outer_pts,
                max_error=self.config.bezier_error_threshold,
                rdp_eps=self.config.rdp_epsilon
            )
            if not outer_beziers:
                continue

            subpath_strings.append(CubicBezierFitter.curves_to_svg_path(outer_beziers))
            bezier_subpaths.append(outer_beziers)

            for hole_cnt in holes:
                hole_pts = hole_cnt.reshape(-1, 2)
                hole_beziers = CubicBezierFitter.fit_closed_contour(
                    hole_pts,
                    max_error=self.config.bezier_error_threshold,
                    rdp_eps=self.config.rdp_epsilon
                )
                if hole_beziers:
                    subpath_strings.append(CubicBezierFitter.curves_to_svg_path(hole_beziers))
                    bezier_subpaths.append(hole_beziers)

            x, y, w, h = cv2.boundingRect(outer_cnt)
            M = cv2.moments(outer_cnt)
            cx = M["m10"] / M["m00"] if M["m00"] > 0 else x + w / 2.0
            cy = M["m01"] / M["m00"] if M["m00"] > 0 else y + h / 2.0

            shapes.append(VectorShape(
                color_index=color_idx,
                hex_color=self.palette_hex[color_idx],
                svg_path_d=" ".join(subpath_strings),
                area=area,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                bezier_subpaths=bezier_subpaths
            ))

        return shapes

    # -----------------------------------------------------------------
    # ETAPA 4: Rastreamento Temporal de Shapes (IoU + Centroid Matching)
    # -----------------------------------------------------------------
    def track_shapes_across_frames(self, all_frames_shapes: List[List[VectorShape]]) -> None:
        next_track_id = 1
        if not all_frames_shapes:
            return

        for shape in all_frames_shapes[0]:
            shape.track_id = next_track_id
            next_track_id += 1

        for t in range(1, len(all_frames_shapes)):
            prev_shapes = all_frames_shapes[t - 1]
            curr_shapes = all_frames_shapes[t]
            used_prev = set()

            for curr in curr_shapes:
                best_iou = 0.0
                best_prev_idx = -1
                for p_idx, prev in enumerate(prev_shapes):
                    if p_idx in used_prev or prev.color_index != curr.color_index:
                        continue
                    iou = self._bbox_iou(prev.bbox, curr.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_prev_idx = p_idx

                if best_iou >= 0.35 and best_prev_idx != -1:
                    curr.track_id = prev_shapes[best_prev_idx].track_id
                    used_prev.add(best_prev_idx)
                else:
                    curr.track_id = next_track_id
                    next_track_id += 1

    @staticmethod
    def _bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)

    # -----------------------------------------------------------------
    # ETAPA 5: Empacotamento em SVG Animado Independente e Lottie JSON
    # -----------------------------------------------------------------
    def export_animated_svg(self, all_frames_shapes: List[List[VectorShape]], canvas_size: Tuple[int, int]) -> Tuple[str, int]:
        width, height = canvas_size
        num_frames = len(all_frames_shapes)
        duration = num_frames / float(self.config.target_fps)
        total_shapes = sum(len(f) for f in all_frames_shapes)

        css_vars = "\n    ".join([f"--color-{i}: {hex_c};" for i, hex_c in enumerate(self.palette_hex)])

        frame_css = []
        pct_per_frame = 100.0 / max(1, num_frames)
        for i in range(num_frames):
            start_pct = i * pct_per_frame
            end_pct = (i + 1) * pct_per_frame
            frame_css.append(
                f"  #frame-{i} {{ animation: show-frame-{i} {duration:.4f}s steps(1) infinite; }}\n"
                f"  @keyframes show-frame-{i} {{\n"
                f"    0%, 100% {{ display: none; }}\n"
                f"    {start_pct:.2f}%, {end_pct - 0.01:.2f}% {{ display: inline; }}\n"
                f"  }}"
            )

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'data-total-shapes="{total_shapes}" data-fps="{self.config.target_fps}">',
            '  <style>',
            f'    :root {{\n    {css_vars}\n    }}',
            '    .anim8-frame { display: none; }',
            "\n".join(frame_css),
            '  </style>'
        ]

        for f_idx, frame_shapes in enumerate(all_frames_shapes):
            svg_lines.append(f'  <g id="frame-{f_idx}" class="anim8-frame" data-frame="{f_idx}">')
            for s_idx, shape in enumerate(frame_shapes):
                svg_lines.append(
                    f'    <path id="f{f_idx}-s{s_idx}" data-track-id="{shape.track_id}" '
                    f'data-color-idx="{shape.color_index}" fill="var(--color-{shape.color_index})" '
                    f'fill-rule="evenodd" d="{shape.svg_path_d}" />'
                )
            svg_lines.append('  </g>')

        svg_lines.append('</svg>')
        return "\n".join(svg_lines), total_shapes

    def export_lottie_json(self, all_frames_shapes: List[List[VectorShape]], canvas_size: Tuple[int, int]) -> str:
        width, height = canvas_size
        num_frames = len(all_frames_shapes)
        layers = []

        for f_idx, frame_shapes in enumerate(all_frames_shapes):
            shape_items = []
            for s_idx, shape in enumerate(frame_shapes):
                r, g, b = self.palette_rgb[shape.color_index] / 255.0
                group_items = []

                for sub_beziers in shape.bezier_subpaths:
                    vertices = []
                    in_tangents = []
                    out_tangents = []
                    n_segs = len(sub_beziers)

                    for seg_idx in range(n_segs):
                        p0, p1, p2, p3 = sub_beziers[seg_idx]
                        prev_p2 = sub_beziers[(seg_idx - 1) % n_segs][2]
                        vertices.append([round(float(p0[0]), 2), round(float(p0[1]), 2)])
                        in_tangents.append([round(float(prev_p2[0] - p0[0]), 2), round(float(prev_p2[1] - p0[1]), 2)])
                        out_tangents.append([round(float(p1[0] - p0[0]), 2), round(float(p1[1] - p0[1]), 2)])

                    group_items.append({
                        "ty": "sh",
                        "ks": {"a": 0, "k": {"i": in_tangents, "o": out_tangents, "v": vertices, "c": True}}
                    })

                group_items.append({
                    "ty": "fl",
                    "c": {"a": 0, "k": [round(float(r), 3), round(float(g), 3), round(float(b), 3), 1]},
                    "o": {"a": 0, "k": 100},
                    "r": 2  # Even-Odd Fill Rule
                })
                group_items.append({"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}})

                shape_items.append({"ty": "gr", "it": group_items, "nm": f"track_{shape.track_id}"})

            layers.append({
                "ddd": 0,
                "ind": f_idx + 1,
                "ty": 4,
                "nm": f"frame_{f_idx}",
                "ip": f_idx,
                "op": f_idx + 1,
                "st": f_idx,
                "ks": {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0}, "p": {"a": 0, "k": [0, 0, 0]}, "a": {"a": 0, "k": [0, 0, 0]}, "s": {"a": 0, "k": [100, 100, 100]}},
                "shapes": shape_items
            })

        lottie_dict = {
            "v": "5.7.4",
            "fr": self.config.target_fps,
            "ip": 0,
            "op": num_frames,
            "w": width,
            "h": height,
            "nm": "Anim8 Vector Export",
            "ddd": 0,
            "assets": [],
            "layers": layers
        }
        return json.dumps(lottie_dict)

    # -----------------------------------------------------------------
    # PIPELINE COMPLETO
    # -----------------------------------------------------------------
    def process_video(self, video_path: str, output_svg_path: str, output_lottie_path: Optional[str] = None) -> Dict:
        frames_rgb, canvas_size = self.extract_video_frames(video_path)
        self.compute_global_palette(frames_rgb)

        all_frames_shapes = []
        for idx, frame in enumerate(frames_rgb):
            label_map = self.quantize_frame(frame)
            frame_shapes = self.trace_frame(label_map)
            all_frames_shapes.append(frame_shapes)

        self.track_shapes_across_frames(all_frames_shapes)
        svg_content, total_shapes = self.export_animated_svg(all_frames_shapes, canvas_size)

        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        if output_lottie_path:
            lottie_content = self.export_lottie_json(all_frames_shapes, canvas_size)
            with open(output_lottie_path, "w", encoding="utf-8") as f:
                f.write(lottie_content)

        return {
            "frames_processed": len(frames_rgb),
            "total_shapes": total_shapes,
            "palette_hex": self.palette_hex,
            "canvas_size": canvas_size,
            "svg_path": output_svg_path,
            "lottie_path": output_lottie_path
        }


# =====================================================================
# TESTE AUTOMÁTICO E DEMONSTRAÇÃO
# =====================================================================
def generate_demo_video(demo_video: str = "sample_ai_mascot.mp4"):
    w, h, fps, n_frames = 256, 256, 15, 12
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(demo_video, fourcc, fps, (w, h))

    rng = np.random.default_rng(42)
    for t in range(n_frames):
        canvas = np.full((h, w, 3), (235, 235, 235), dtype=np.uint8)
        dy = int(round(8 * math.sin(2 * math.pi * t / n_frames)))

        # Corpo branco do mascote
        cv2.circle(canvas, (135, 135 + dy), 55, (250, 250, 250), -1)
        # Óculos cinzas com furos brancos (testando hierarquia RETR_CCOMP)
        cv2.circle(canvas, (115, 120 + dy), 28, (195, 195, 195), -1)
        cv2.circle(canvas, (115, 120 + dy), 20, (252, 252, 252), -1)
        cv2.circle(canvas, (162, 120 + dy), 28, (195, 195, 195), -1)
        cv2.circle(canvas, (162, 120 + dy), 20, (252, 252, 252), -1)
        # Pupilas pretas e crista laranja
        cv2.circle(canvas, (120, 120 + dy), 6, (20, 20, 20), -1)
        cv2.circle(canvas, (158, 120 + dy), 6, (20, 20, 20), -1)
        cv2.ellipse(canvas, (138, 92 + dy), (8, 16), 0, 0, 360, (30, 90, 225), -1)
        # Ovo dourado animado
        cv2.ellipse(canvas, (85, 145 - dy), (32, 44), -10, 0, 360, (35, 165, 235), -1)

        # Adiciona ruído granulado típico de geradores de vídeo por IA
        noise = rng.integers(-8, 9, size=canvas.shape, dtype=np.int16)
        noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(noisy_canvas)

    writer.release()


if __name__ == "__main__":
    demo_video = "sample_ai_mascot.mp4"
    generate_demo_video(demo_video)

    cfg = VectorizerConfig(
        num_colors=8,
        detail_level="balanced",
        smoothing=40.0,
        path_budget=128,
        target_fps=15,
        remove_background=False
    )
    engine = Anim8Engine(cfg)
    stats = engine.process_video(demo_video, "output_animated.svg", "output_lottie.json")
    print("Processamento concluído com sucesso:", json.dumps(stats, indent=2))
