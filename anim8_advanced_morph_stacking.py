"""
Anim8 Advanced Engine: Topological Layer Stacking & Bezier Morphing
===================================================================
Resolve os dois maiores desafios de vetorização de vídeo:
1. Gap-Free Layer Stacking (elimina frestas entre caminhos adjacentes)
2. Topologia Fixa e Alinhamento de Fase para Morphing Nativo em Lottie/SVG
"""

import cv2
import json
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from anim8_backend_engine import (
    VectorizerConfig,
    VectorShape,
    CubicBezierFitter,
    Anim8Engine,
)

# Tipo para um segmento de Bézier Cúbico: (P0, P1, P2, P3) onde cada P é np.ndarray([x, y])
BezierSegment = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# =====================================================================
# MÓDULO 1: MATEMÁTICA DE SUBDIVISÃO DE DE CASTELJAU E ALINHAMENTO
# =====================================================================
class BezierTopologySynchronizer:
    """
    Iguala o número de nós de Bézier entre diferentes quadros de um mesmo `track_id`
    e alinha o índice inicial (rotação circular de fase) para permitir interpolação
    contínua (Morphing) sem torcer ou cruzar as curvas.
    """

    @staticmethod
    def split_cubic_bezier(seg: BezierSegment, t: float = 0.5) -> Tuple[BezierSegment, BezierSegment]:
        """
        Divide uma curva de Bézier cúbica em duas no parâmetro `t` usando o
        Algoritmo de De Casteljau, mantendo a geometria 100% idêntica.
        """
        p0, p1, p2, p3 = seg
        p01 = (1.0 - t) * p0 + t * p1
        p12 = (1.0 - t) * p1 + t * p2
        p23 = (1.0 - t) * p2 + t * p3

        p012 = (1.0 - t) * p01 + t * p12
        p123 = (1.0 - t) * p12 + t * p23

        p0123 = (1.0 - t) * p012 + t * p123

        left: BezierSegment = (p0, p01, p012, p0123)
        right: BezierSegment = (p0123, p123, p23, p3)
        return left, right

    @staticmethod
    def segment_arc_length(seg: BezierSegment) -> float:
        """Aproximação rápida de comprimento de arco por corda + polígono de controle."""
        p0, p1, p2, p3 = seg
        chord = np.linalg.norm(p3 - p0)
        ctrl_net = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
        return 0.5 * (chord + ctrl_net)

    @classmethod
    def upsample_to_target_count(cls, curves: List[BezierSegment], target_n: int) -> List[BezierSegment]:
        """
        Subdivide iterativamente o segmento de Bézier mais longo até que o contorno
        fechado tenha exatamente `target_n` segmentos.
        """
        if not curves or len(curves) >= target_n:
            return list(curves)

        work_curves = list(curves)
        while len(work_curves) < target_n:
            # Encontra o índice do segmento de maior comprimento de arco
            lengths = [cls.segment_arc_length(c) for c in work_curves]
            longest_idx = int(np.argmax(lengths))
            left, right = cls.split_cubic_bezier(work_curves[longest_idx], t=0.5)
            work_curves[longest_idx:longest_idx + 1] = [left, right]

        return work_curves

    @staticmethod
    def align_starting_phase(ref_curves: List[BezierSegment], target_curves: List[BezierSegment]) -> List[BezierSegment]:
        """
        Encontra o deslocamento circular k que minimiza a soma das distâncias quadráticas
        entre as âncoras P0 da curva de referência (Frame t) e da curva alvo (Frame t+1).
        Evita que a forma "gire sobre si mesma" durante o morphing.
        """
        n = len(ref_curves)
        if n == 0 or len(target_curves) != n:
            return target_curves

        ref_anchors = np.array([c[0] for c in ref_curves])      # (N, 2)
        tgt_anchors = np.array([c[0] for c in target_curves])   # (N, 2)

        # Normaliza pelo centroide para que a translação pura não afete o alinhamento angular
        ref_centered = ref_anchors - np.mean(ref_anchors, axis=0)
        tgt_centered = tgt_anchors - np.mean(tgt_anchors, axis=0)

        best_shift = 0
        min_cost = float("inf")

        for shift in range(n):
            shifted_tgt = np.roll(tgt_centered, -shift, axis=0)
            cost = float(np.sum((ref_centered - shifted_tgt) ** 2))
            if cost < min_cost:
                min_cost = cost
                best_shift = shift

        return target_curves[best_shift:] + target_curves[:best_shift]


# =====================================================================
# MÓDULO 2: ENGINE AVANÇADA COM EMPILHAMENTO SEM FRESTAS E MORPHING
# =====================================================================
class Anim8ProEngine(Anim8Engine):
    """
    Estende a Anim8Engine original adicionando:
    - Empilhamento cumulativo de máscaras (Stacked Layer Tracing) para zero frestas.
    - Exportação de Keyframes Interpolados (Morphing real de nós) para Lottie JSON e SVG.
    """

    def sort_palette_by_luminance(self) -> List[int]:
        """
        Ordena os índices da paleta global da cor mais clara (fundo/base) para a
        mais escura (detalhes/contornos superiores), usando o canal L do CIELAB.
        """
        l_values = self.palette_lab[:, 0]
        order = list(np.argsort(-l_values))  # Decrescente em Luminância (claro -> escuro)

        # Se houver cor de fundo marcada para remoção, coloca no início absoluto
        if self.config.remove_background and self.bg_color_index is not None:
            if self.bg_color_index in order:
                order.remove(self.bg_color_index)
            order.insert(0, self.bg_color_index)

        return [int(i) for i in order]

    def trace_frame_stacked(self, label_map: np.ndarray) -> List[VectorShape]:
        """
        Substitui o `trace_frame` simples pelo Empilhamento Topológico (Layer Stacking):
        Em vez de `mask = (label_map == color_idx)`, cada cor forma uma máscara cumulativa
        com todas as cores que serão desenhadas POR CIMA dela.
        Assim, a forma de baixo se estende por baixo da forma de cima, eliminando qualquer
        fresta branca causada pela suavização das curvas de Bézier!
        """
        layer_order = self.sort_palette_by_luminance()
        candidate_regions = []
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for rank_idx, color_idx in enumerate(layer_order):
            if self.config.remove_background and color_idx == self.bg_color_index:
                continue

            # Máscara própria desta cor
            exact_mask = (label_map == color_idx)
            if not np.any(exact_mask):
                continue

            # União cumulativa: esta cor + todas as cores que serão pintadas depois (por cima)
            upper_colors = layer_order[rank_idx:]
            cumulative_mask = np.isin(label_map, upper_colors).astype(np.uint8) * 255
            cumulative_mask = cv2.morphologyEx(cumulative_mask, cv2.MORPH_CLOSE, kernel_close)

            # Mantém apenas os componentes conectados da máscara cumulativa que realmente
            # contêm pixels da cor atual (`exact_mask`), evitando clonar ilhas isoladas de outras cores
            num_labels, cc_labels = cv2.connectedComponents(cumulative_mask)
            valid_cc = np.unique(cc_labels[exact_mask])
            valid_cc = valid_cc[valid_cc != 0]
            mask = np.isin(cc_labels, valid_cc).astype(np.uint8) * 255

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

                # Guarda `rank_idx` para preservar a ordem de pintura das camadas empilhadas
                candidate_regions.append((area, rank_idx, color_idx, outer_cnt, holes))

        # Filtra pelo Path Budget mantendo as N regiões de maior relevância visual
        candidate_regions.sort(key=lambda x: x[0], reverse=True)
        selected = candidate_regions[:self.config.path_budget]

        # Reordena pelo `rank_idx` crescente (camadas base primeiro, detalhes por cima)
        selected.sort(key=lambda x: (x[1], -x[0]))

        shapes: List[VectorShape] = []
        for area, _, color_idx, outer_cnt, holes in selected:
            subpath_strings = []
            bezier_subpaths = []

            outer_beziers = CubicBezierFitter.fit_closed_contour(
                outer_cnt.reshape(-1, 2),
                max_error=self.config.bezier_error_threshold,
                rdp_eps=self.config.rdp_epsilon
            )
            if not outer_beziers:
                continue

            subpath_strings.append(CubicBezierFitter.curves_to_svg_path(outer_beziers))
            bezier_subpaths.append(outer_beziers)

            for hole_cnt in holes:
                hole_beziers = CubicBezierFitter.fit_closed_contour(
                    hole_cnt.reshape(-1, 2),
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

    def synchronize_tracked_topology(self, all_frames_shapes: List[List[VectorShape]]) -> Dict[int, List[Tuple[int, VectorShape]]]:
        """
        Agrupa todos os shapes pelo `track_id` ao longo do tempo e iguala o número
        de nós de Bézier do contorno principal usando subdivisão de De Casteljau
        + alinhamento circular de fase.
        """
        tracks: Dict[int, List[Tuple[int, VectorShape]]] = {}
        for f_idx, frame_shapes in enumerate(all_frames_shapes):
            for shape in frame_shapes:
                tracks.setdefault(shape.track_id, []).append((f_idx, shape))

        for track_id, timeline in tracks.items():
            if len(timeline) < 2:
                continue

            # Filtra shapes que possuem apenas 1 contorno simples fechado (sem mudança de topologia de furos)
            if not all(len(s.bezier_subpaths) == 1 for _, s in timeline):
                continue

            max_segments = max(len(s.bezier_subpaths[0]) for _, s in timeline)
            prev_aligned: List[BezierSegment] = []

            for idx, (f_idx, shape) in enumerate(timeline):
                upsampled = BezierTopologySynchronizer.upsample_to_target_count(
                    shape.bezier_subpaths[0], max_segments
                )
                if idx > 0 and prev_aligned:
                    upsampled = BezierTopologySynchronizer.align_starting_phase(prev_aligned, upsampled)

                shape.bezier_subpaths[0] = upsampled
                shape.svg_path_d = CubicBezierFitter.curves_to_svg_path(upsampled)
                prev_aligned = upsampled

        return tracks

    def export_morphed_lottie_json(self, all_frames_shapes: List[List[VectorShape]], canvas_size: Tuple[int, int]) -> str:
        """
        Exporta um Lottie JSON onde objetos rastreados continuamente (`track_id` em múltiplos
        quadros consecutivos) são convertidos em UMA ÚNICA LAYER com Keyframes animados (`"a": 1`),
        permitindo editar a curva na timeline e ter interpolação fluida a 60fps!
        """
        width, height = canvas_size
        num_frames = len(all_frames_shapes)
        tracks = self.synchronize_tracked_topology(all_frames_shapes)

        layers = []
        layer_ind = 1

        for track_id, timeline in tracks.items():
            first_frame = timeline[0][0]
            last_frame = timeline[-1][0]
            first_shape = timeline[0][1]
            r, g, b = self.palette_rgb[first_shape.color_index] / 255.0

            # Verifica se o track é contínuo e tem topologia simples sincronizada
            is_continuous = (
                len(timeline) > 1
                and (last_frame - first_frame + 1) == len(timeline)
                and all(len(s.bezier_subpaths) == 1 for _, s in timeline)
            )

            if is_continuous:
                keyframes = []
                for f_idx, shape in timeline:
                    sub = shape.bezier_subpaths[0]
                    n_segs = len(sub)
                    vertices, in_tan, out_tan = [], [], []
                    for seg_idx in range(n_segs):
                        p0, p1, p2, p3 = sub[seg_idx]
                        prev_p2 = sub[(seg_idx - 1) % n_segs][2]
                        vertices.append([round(float(p0[0]), 2), round(float(p0[1]), 2)])
                        in_tan.append([round(float(prev_p2[0] - p0[0]), 2), round(float(prev_p2[1] - p0[1]), 2)])
                        out_tan.append([round(float(p1[0] - p0[0]), 2), round(float(p1[1] - p0[1]), 2)])

                    keyframes.append({
                        "t": f_idx,
                        "s": [{"i": in_tan, "o": out_tan, "v": vertices, "c": True}],
                        "i": {"x": [0.667], "y": [1.0]},
                        "o": {"x": [0.333], "y": [0.0]}
                    })

                shape_group = {
                    "ty": "gr",
                    "nm": f"morphed_track_{track_id}",
                    "it": [
                        {"ty": "sh", "ks": {"a": 1, "k": keyframes}},
                        {"ty": "fl", "c": {"a": 0, "k": [round(float(r), 3), round(float(g), 3), round(float(b), 3), 1]}, "o": {"a": 0, "k": 100}, "r": 2},
                        {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
                    ]
                }

                layers.append({
                    "ddd": 0,
                    "ind": layer_ind,
                    "ty": 4,
                    "nm": f"track_{track_id}_morphed",
                    "ip": first_frame,
                    "op": last_frame + 1,
                    "st": first_frame,
                    "ks": {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0}, "p": {"a": 0, "k": [0, 0, 0]}, "a": {"a": 0, "k": [0, 0, 0]}, "s": {"a": 0, "k": [100, 100, 100]}},
                    "shapes": [shape_group]
                })
                layer_ind += 1
            else:
                # Fallback frame-a-frame para shapes com múltiplos furos que mudam de topologia
                for f_idx, shape in timeline:
                    group_items = []
                    for sub_beziers in shape.bezier_subpaths:
                        n_segs = len(sub_beziers)
                        vertices, in_tan, out_tan = [], [], []
                        for seg_idx in range(n_segs):
                            p0, p1, p2, p3 = sub_beziers[seg_idx]
                            prev_p2 = sub_beziers[(seg_idx - 1) % n_segs][2]
                            vertices.append([round(float(p0[0]), 2), round(float(p0[1]), 2)])
                            in_tan.append([round(float(prev_p2[0] - p0[0]), 2), round(float(prev_p2[1] - p0[1]), 2)])
                            out_tan.append([round(float(p1[0] - p0[0]), 2), round(float(p1[1] - p0[1]), 2)])
                        group_items.append({"ty": "sh", "ks": {"a": 0, "k": {"i": in_tan, "o": out_tan, "v": vertices, "c": True}}})

                    group_items.append({"ty": "fl", "c": {"a": 0, "k": [round(float(r), 3), round(float(g), 3), round(float(b), 3), 1]}, "o": {"a": 0, "k": 100}, "r": 2})
                    group_items.append({"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}})

                    layers.append({
                        "ddd": 0,
                        "ind": layer_ind,
                        "ty": 4,
                        "nm": f"track_{track_id}_f{f_idx}",
                        "ip": f_idx,
                        "op": f_idx + 1,
                        "st": f_idx,
                        "ks": {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0}, "p": {"a": 0, "k": [0, 0, 0]}, "a": {"a": 0, "k": [0, 0, 0]}, "s": {"a": 0, "k": [100, 100, 100]}},
                        "shapes": [{"ty": "gr", "it": group_items, "nm": f"track_{track_id}"}]
                    })
                    layer_ind += 1

        return json.dumps({
            "v": "5.7.4",
            "fr": self.config.target_fps,
            "ip": 0,
            "op": num_frames,
            "w": width,
            "h": height,
            "nm": "Anim8 Pro Morphed Export",
            "ddd": 0,
            "assets": [],
            "layers": layers
        })

    def process_video_advanced(
        self,
        video_path: str,
        output_svg_path: Optional[str] = None,
        output_lottie_path: Optional[str] = None,
        use_stacking: bool = True,
        use_morphing: bool = True,
        progress_callback: Optional[Callable[[int, str, Optional[Dict]], None]] = None,
    ) -> Dict:
        """
        Executa o pipeline completo avançado da Anim8 Engine:
        1. Extração e reamostragem temporal dos quadros com filtro bilateral
        2. Treinamento da paleta global no espaço perceptivo CIELAB (anti-flickering)
        3. Vetorização quadro a quadro (com opção de Empilhamento Topológico Anti-Frestas)
        4. Rastreamento temporal de formas contínuas (IoU + Centroide)
        5. Sincronização de topologia via Subdivisão de De Casteljau (Morphing)
        6. Empacotamento em SVG com CSS Variables recoloríveis e Lottie JSON profissional
        """
        def notify(pct: int, msg: str, extra: Optional[Dict] = None):
            if progress_callback:
                try:
                    progress_callback(pct, msg, extra)
                except Exception:
                    pass

        notify(5, "Descodificando e extraindo quadros do vídeo...")
        frames_rgb, canvas_size = self.extract_video_frames(video_path)
        num_frames = len(frames_rgb)

        if num_frames == 0:
            raise ValueError("Nenhum quadro pôde ser extraído do arquivo de vídeo.")

        notify(20, f"Quantizando paleta global no espaço CIELAB ({self.config.num_colors} cores)...")
        self.compute_global_palette(frames_rgb)

        all_frames_shapes = []
        for idx, frame in enumerate(frames_rgb):
            progress_pct = 25 + int((idx / num_frames) * 45)
            notify(
                progress_pct,
                f"Vetorizando quadro {idx + 1}/{num_frames}...",
                {"current_frame": idx + 1, "total_frames": num_frames}
            )
            label_map = self.quantize_frame(frame)
            if use_stacking:
                frame_shapes = self.trace_frame_stacked(label_map)
            else:
                frame_shapes = self.trace_frame(label_map)
            all_frames_shapes.append(frame_shapes)

        notify(72, "Rastreando correspondência temporal de formas (IoU + Centroide)...")
        self.track_shapes_across_frames(all_frames_shapes)

        if use_morphing:
            notify(82, "Sincronizando topologia de Bézier (Subdivisão De Casteljau)...")
            self.synchronize_tracked_topology(all_frames_shapes)

        notify(88, "Gerando SVG animado e variáveis CSS de cores...")
        svg_content, total_shapes = self.export_animated_svg(all_frames_shapes, canvas_size)

        notify(94, "Compilando especificação vetorial Lottie JSON...")
        if use_morphing:
            lottie_content = self.export_morphed_lottie_json(all_frames_shapes, canvas_size)
        else:
            lottie_content = self.export_lottie_json(all_frames_shapes, canvas_size)

        if output_svg_path:
            with open(output_svg_path, "w", encoding="utf-8") as f:
                f.write(svg_content)

        if output_lottie_path:
            with open(output_lottie_path, "w", encoding="utf-8") as f:
                f.write(lottie_content)

        # Preparar dados de nós e âncoras para o editor visual (inspeção de Béziers)
        shapes_by_frame = []
        for f_shapes in all_frames_shapes:
            f_list = []
            for s in f_shapes:
                anchors = []
                for sub in s.bezier_subpaths:
                    sub_anchors = []
                    for seg in sub:
                        p0, p1, p2, p3 = seg
                        sub_anchors.append([
                            [round(float(p0[0]), 2), round(float(p0[1]), 2)],
                            [round(float(p1[0]), 2), round(float(p1[1]), 2)],
                            [round(float(p2[0]), 2), round(float(p2[1]), 2)],
                            [round(float(p3[0]), 2), round(float(p3[1]), 2)],
                        ])
                    anchors.append(sub_anchors)

                f_list.append({
                    "track_id": int(s.track_id),
                    "color_index": int(s.color_index),
                    "hex_color": str(s.hex_color),
                    "area": round(float(s.area), 1),
                    "bbox": [int(x) for x in s.bbox],
                    "centroid": [round(float(s.centroid[0]), 1), round(float(s.centroid[1]), 1)],
                    "d": s.svg_path_d,
                    "anchors": anchors
                })
            shapes_by_frame.append(f_list)

        notify(100, "Vetorização concluída com sucesso!")

        return {
            "frames_processed": num_frames,
            "total_shapes": total_shapes,
            "palette_hex": self.palette_hex,
            "canvas_size": list(canvas_size),
            "target_fps": self.config.target_fps,
            "duration": round(num_frames / float(self.config.target_fps), 3),
            "svg_content": svg_content,
            "lottie_content": lottie_content,
            "svg_path": output_svg_path,
            "lottie_path": output_lottie_path,
            "use_stacking": use_stacking,
            "use_morphing": use_morphing,
            "frame_stats": [{"frame": i, "shapes_count": len(f)} for i, f in enumerate(all_frames_shapes)],
            "shapes_by_frame": shapes_by_frame,
        }