from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


COLORS = {
    "blue": "#3566a8",
    "green": "#2d8c61",
    "red": "#b94d48",
    "orange": "#c8872d",
    "purple": "#7257a8",
    "gray": "#7a7f87",
    "light_gray": "#edf0f4",
    "dark": "#20242a",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    opener = (ROOT / path).open() if not Path(path).is_absolute() else Path(path).open()
    with opener as f:
        return json.load(f)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Canvas:
    def __init__(self, width: int = 720, height: int = 480) -> None:
        self.width = width
        self.height = height
        self.elements: list[tuple[str, tuple[Any, ...]]] = []

    def rect(self, x: float, y: float, w: float, h: float, *, fill: str, stroke: str | None = None) -> None:
        self.elements.append(("rect", (x, y, w, h, fill, stroke)))

    def line(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str = "#20242a", width: float = 1.0) -> None:
        self.elements.append(("line", (x1, y1, x2, y2, stroke, width)))

    def circle(self, x: float, y: float, r: float, *, fill: str, stroke: str | None = None) -> None:
        self.elements.append(("circle", (x, y, r, fill, stroke)))

    def text(self, x: float, y: float, text: str, *, size: int = 12, fill: str = "#20242a", anchor: str = "start") -> None:
        self.elements.append(("text", (x, y, text, size, fill, anchor)))

    def polyline(self, points: list[tuple[float, float]], *, stroke: str, width: float = 2.0) -> None:
        self.elements.append(("polyline", (points, stroke, width)))

    def save_svg(self, path: Path) -> None:
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        for kind, args in self.elements:
            if kind == "rect":
                x, y, w, h, fill, stroke = args
                stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
                lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"{stroke_attr}/>')
            elif kind == "line":
                x1, y1, x2, y2, stroke, width = args
                lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}"/>')
            elif kind == "circle":
                x, y, r, fill, stroke = args
                stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}"{stroke_attr}/>')
            elif kind == "text":
                x, y, text, size, fill, anchor = args
                lines.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{_escape(str(text))}</text>')
            elif kind == "polyline":
                points, stroke, width = args
                point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                lines.append(f'<polyline points="{point_text}" fill="none" stroke="{stroke}" stroke-width="{width:.2f}"/>')
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n")

    def save_pdf(self, path: Path) -> None:
        commands: list[str] = []

        def y_pdf(y: float) -> float:
            return self.height - y

        for kind, args in self.elements:
            if kind == "rect":
                x, y, w, h, fill, stroke = args
                r, g, b = _hex_to_rgb(fill)
                commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg {x:.2f} {y_pdf(y + h):.2f} {w:.2f} {h:.2f} re f")
                if stroke:
                    r, g, b = _hex_to_rgb(stroke)
                    commands.append(f"{r:.4f} {g:.4f} {b:.4f} RG {x:.2f} {y_pdf(y + h):.2f} {w:.2f} {h:.2f} re S")
            elif kind == "line":
                x1, y1, x2, y2, stroke, width = args
                r, g, b = _hex_to_rgb(stroke)
                commands.append(f"{width:.2f} w {r:.4f} {g:.4f} {b:.4f} RG {x1:.2f} {y_pdf(y1):.2f} m {x2:.2f} {y_pdf(y2):.2f} l S")
            elif kind == "circle":
                x, y, radius, fill, stroke = args
                r, g, b = _hex_to_rgb(fill)
                commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg {x - radius:.2f} {y_pdf(y + radius):.2f} {2 * radius:.2f} {2 * radius:.2f} re f")
                if stroke:
                    r, g, b = _hex_to_rgb(stroke)
                    commands.append(f"{r:.4f} {g:.4f} {b:.4f} RG {x - radius:.2f} {y_pdf(y + radius):.2f} {2 * radius:.2f} {2 * radius:.2f} re S")
            elif kind == "text":
                x, y, text, size, fill, _anchor = args
                r, g, b = _hex_to_rgb(fill)
                commands.append(f"BT /F1 {size} Tf {r:.4f} {g:.4f} {b:.4f} rg {x:.2f} {y_pdf(y):.2f} Td ({_pdf_escape(str(text))}) Tj ET")
            elif kind == "polyline":
                points, stroke, width = args
                if not points:
                    continue
                r, g, b = _hex_to_rgb(stroke)
                start = points[0]
                rest = " ".join(f"{x:.2f} {y_pdf(y):.2f} l" for x, y in points[1:])
                commands.append(f"{width:.2f} w {r:.4f} {g:.4f} {b:.4f} RG {start[0]:.2f} {y_pdf(start[1]):.2f} m {rest} S")

        stream = "\n".join(commands).encode()
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{idx} 0 obj\n".encode() + obj + b"\nendobj\n")
        xref = len(pdf)
        pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode())
        pdf.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        path.write_bytes(pdf)


def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _heat_color(value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    t = 0.0 if vmax <= vmin else (value - vmin) / (vmax - vmin)
    t = float(np.clip(t, 0.0, 1.0))
    r = _lerp(237, 53, t)
    g = _lerp(240, 102, t)
    b = _lerp(244, 168, t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _chart_frame(title: str) -> Canvas:
    canvas = Canvas()
    canvas.text(36, 34, title, size=18)
    return canvas


def _axis(canvas: Canvas, x0: float, y0: float, w: float, h: float, *, xlabel: str = "", ylabel: str = "") -> None:
    canvas.line(x0, y0 + h, x0 + w, y0 + h, stroke="#20242a", width=1)
    canvas.line(x0, y0, x0, y0 + h, stroke="#20242a", width=1)
    if xlabel:
        canvas.text(x0 + w / 2 - 40, y0 + h + 38, xlabel, size=11)
    if ylabel:
        canvas.text(16, y0 + h / 2, ylabel, size=11)


def _scale(values: list[float], low: float, high: float, *, log: bool = False) -> tuple[Callable[[float], float], float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if log:
        arr = np.log10(np.maximum(arr, 1e-12))
    vmin = float(np.min(arr)) if arr.size else 0.0
    vmax = float(np.max(arr)) if arr.size else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0

    def convert(value: float) -> float:
        value = math.log10(max(value, 1e-12)) if log else value
        return low + (float(value) - vmin) / (vmax - vmin) * (high - low)

    return convert, vmin, vmax


def _draw_heatmap(rows: list[dict[str, Any]], title: str, labels: list[str] | None = None) -> Canvas:
    labels = labels or sorted({str(r["x"]) for r in rows} | {str(r["y"]) for r in rows})
    values = {(str(r["x"]), str(r["y"])): float(r["value"]) for r in rows}
    canvas = _chart_frame(title)
    x0, y0, cell = 150, 80, min(58, 360 / max(len(labels), 1))
    for i, ylab in enumerate(labels):
        canvas.text(x0 - 8, y0 + i * cell + cell * 0.62, ylab, size=10, anchor="end")
        canvas.text(x0 + i * cell + cell * 0.5, y0 - 12, labels[i], size=10, anchor="middle")
        for j, xlab in enumerate(labels):
            value = values.get((xlab, ylab), values.get((ylab, xlab), 1.0 if xlab == ylab else 0.0))
            canvas.rect(x0 + j * cell, y0 + i * cell, cell, cell, fill=_heat_color(value), stroke="#ffffff")
            canvas.text(x0 + j * cell + cell * 0.5 - 12, y0 + i * cell + cell * 0.58, f"{value:.2f}", size=8)
    canvas.text(x0, y0 + len(labels) * cell + 38, "Cell value: subspace overlap / agreement", size=11, fill=COLORS["gray"])
    return canvas


def _draw_lines(rows: list[dict[str, Any]], title: str, *, xkey: str, ykey: str, groupkey: str, logx: bool = False) -> Canvas:
    canvas = _chart_frame(title)
    x0, y0, w, h = 72, 70, 560, 330
    xs = [float(r[xkey]) for r in rows]
    ys = [float(r[ykey]) for r in rows]
    sx, _, _ = _scale(xs, x0, x0 + w, log=logx)
    sy, _, _ = _scale(ys, y0 + h, y0)
    _axis(canvas, x0, y0, w, h, xlabel=xkey, ylabel=ykey)
    groups = sorted({str(r[groupkey]) for r in rows})
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]]
    for idx, group in enumerate(groups):
        group_rows = sorted([r for r in rows if str(r[groupkey]) == group], key=lambda r: float(r[xkey]))
        points = [(sx(float(r[xkey])), sy(float(r[ykey]))) for r in group_rows]
        canvas.polyline(points, stroke=palette[idx % len(palette)], width=2)
        for x, y in points:
            canvas.circle(x, y, 3, fill=palette[idx % len(palette)])
        canvas.text(650, 80 + 18 * idx, group, size=10, fill=palette[idx % len(palette)])
    return canvas


def _draw_bars(rows: list[dict[str, Any]], title: str, *, labelkey: str, valuekey: str, groupkey: str | None = None) -> Canvas:
    canvas = _chart_frame(title)
    x0, y0, w, h = 64, 72, 590, 315
    values = [float(r[valuekey]) for r in rows]
    ymax = max(max(values, default=1.0), 1e-6) * 1.15
    _axis(canvas, x0, y0, w, h, ylabel=valuekey)
    bar_w = w / max(len(rows), 1) * 0.72
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"], COLORS["gray"]]
    for idx, row in enumerate(rows):
        value = float(row[valuekey])
        x = x0 + idx * (w / max(len(rows), 1)) + 4
        bh = value / ymax * h
        group = str(row[groupkey]) if groupkey else ""
        color = palette[(hash(group) if group else idx) % len(palette)]
        canvas.rect(x, y0 + h - bh, bar_w, bh, fill=color)
        label = str(row[labelkey])[:14]
        canvas.text(x, y0 + h + 18, label, size=8)
    return canvas


def _draw_bars_with_ci(
    rows: list[dict[str, Any]],
    title: str,
    *,
    labelkey: str,
    valuekey: str,
    lowkey: str,
    highkey: str,
    groupkey: str | None = None,
) -> Canvas:
    canvas = _chart_frame(title)
    x0, y0, w, h = 64, 72, 590, 315
    evaluated = [row for row in rows if row.get("status", "evaluated") == "evaluated"]
    values = [float(row[valuekey]) for row in evaluated]
    lows = [float(row.get(lowkey, row[valuekey])) for row in evaluated]
    highs = [float(row.get(highkey, row[valuekey])) for row in evaluated]
    ymin = min([0.0, *values, *lows], default=-0.05)
    ymax = max([0.05, *values, *highs], default=0.05)
    pad = max((ymax - ymin) * 0.08, 0.02)
    ymin -= pad
    ymax += pad

    def sy(value: float) -> float:
        return y0 + h - (value - ymin) / (ymax - ymin) * h

    _axis(canvas, x0, y0, w, h, ylabel=valuekey)
    zero = sy(0.0)
    canvas.line(x0, zero, x0 + w, zero, stroke=COLORS["gray"], width=1)
    if not evaluated:
        canvas.text(x0 + 20, y0 + 150, "No evaluated rows", size=12, fill=COLORS["gray"])
        return canvas

    palette = {
        "parametric_b": COLORS["blue"],
        "average_controllability_a": COLORS["green"],
        "koopman_dmd_c": COLORS["orange"],
        "original_graph": COLORS["gray"],
        "pca_proxy": COLORS["gray"],
        "random": COLORS["purple"],
    }
    slot = w / max(len(evaluated), 1)
    bar_w = min(slot * 0.68, 28)
    for idx, row in enumerate(evaluated):
        value = float(row[valuekey])
        low = float(row.get(lowkey, value))
        high = float(row.get(highkey, value))
        x = x0 + idx * slot + (slot - bar_w) / 2
        y_value = sy(value)
        y_base = sy(0.0)
        top = min(y_value, y_base)
        height = max(abs(y_base - y_value), 1.0)
        group = str(row[groupkey]) if groupkey else str(row.get("method", ""))
        color = palette.get(group, COLORS["blue"])
        canvas.rect(x, top, bar_w, height, fill=color)
        cx = x + bar_w / 2
        canvas.line(cx, sy(low), cx, sy(high), stroke=COLORS["dark"], width=1)
        canvas.line(cx - 4, sy(low), cx + 4, sy(low), stroke=COLORS["dark"], width=1)
        canvas.line(cx - 4, sy(high), cx + 4, sy(high), stroke=COLORS["dark"], width=1)
        canvas.text(x, y0 + h + 17, str(row[labelkey])[:16], size=7)

    skipped = sorted({str(row["method"]) for row in rows if row.get("status") != "evaluated"})
    if skipped:
        canvas.text(x0, y0 + h + 42, f"Not evaluated: {', '.join(skipped)}", size=10, fill=COLORS["gray"])
    canvas.text(610, y0 + 12, "0 = chance", size=10, fill=COLORS["gray"])
    return canvas


def _draw_stage5_confusion(rows: list[dict[str, Any]], title: str) -> Canvas:
    canvas = _chart_frame(title)
    envs = sorted({str(row["env"]) for row in rows})
    for env_idx, env in enumerate(envs[:2]):
        env_rows = [row for row in rows if str(row["env"]) == env]
        labels = sorted({int(row["truth"]) for row in env_rows} | {int(row["predicted"]) for row in env_rows})
        x0 = 70 + env_idx * 330
        y0 = 86
        cell = min(34, 240 / max(len(labels), 1))
        canvas.text(x0, y0 - 24, env, size=13)
        for y_idx, truth in enumerate(labels):
            canvas.text(x0 - 8, y0 + y_idx * cell + cell * 0.62, str(truth), size=8, anchor="end")
            canvas.text(x0 + y_idx * cell + cell * 0.5, y0 - 8, str(truth), size=8, anchor="middle")
            for x_idx, pred in enumerate(labels):
                value = next(
                    (float(row["value"]) for row in env_rows if int(row["truth"]) == truth and int(row["predicted"]) == pred),
                    0.0,
                )
                canvas.rect(x0 + x_idx * cell, y0 + y_idx * cell, cell, cell, fill=_heat_color(value), stroke="#ffffff")
                if value >= 0.1:
                    canvas.text(x0 + x_idx * cell + cell * 0.5 - 8, y0 + y_idx * cell + cell * 0.58, f"{value:.2f}", size=7)
    canvas.text(70, 420, "Rows: true coordinate. Columns: predicted coordinate. Values are row-normalized.", size=11, fill=COLORS["gray"])
    return canvas


def _draw_scatter(rows: list[dict[str, Any]], title: str, *, xkey: str, ykey: str, groupkey: str) -> Canvas:
    canvas = _chart_frame(title)
    x0, y0, w, h = 72, 72, 560, 315
    sx, _, _ = _scale([float(r[xkey]) for r in rows], x0, x0 + w)
    sy, _, _ = _scale([float(r[ykey]) for r in rows], y0 + h, y0, log=True)
    _axis(canvas, x0, y0, w, h, xlabel=xkey, ylabel=f"log {ykey}")
    groups = sorted({str(r[groupkey]) for r in rows})
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]]
    for idx, group in enumerate(groups):
        color = palette[idx % len(palette)]
        for row in rows:
            if str(row[groupkey]) == group:
                canvas.circle(sx(float(row[xkey])), sy(float(row[ykey])), 4, fill=color, stroke="#ffffff")
        canvas.text(650, 80 + 18 * idx, group, size=10, fill=color)
    return canvas


def _draw_scatter_linear(rows: list[dict[str, Any]], title: str, *, xkey: str, ykey: str, groupkey: str) -> Canvas:
    canvas = _chart_frame(title)
    x0, y0, w, h = 72, 72, 560, 315
    xvals = [float(r[xkey]) for r in rows]
    yvals = [float(r[ykey]) for r in rows]
    sx, _, _ = _scale(xvals + yvals, x0, x0 + w)
    sy, _, _ = _scale(xvals + yvals, y0 + h, y0)
    _axis(canvas, x0, y0, w, h, xlabel=xkey, ylabel=ykey)
    canvas.line(x0, y0 + h, x0 + w, y0, stroke=COLORS["light_gray"], width=2)
    groups = sorted({str(r[groupkey]) for r in rows})
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]]
    for idx, group in enumerate(groups):
        color = palette[idx % len(palette)]
        for row in rows:
            if str(row[groupkey]) == group:
                canvas.circle(sx(float(row[xkey])), sy(float(row[ykey])), 4, fill=color, stroke="#ffffff")
        canvas.text(650, 80 + 18 * idx, group, size=10, fill=color)
    return canvas


def _stage1_paths(env: str) -> dict[str, str]:
    if env == "swe":
        return {
            "original": "runs/stage1_v2_original_graph_Qwen_Qwen3_8B_prime_swe_grep",
            "v2": "runs/stage1_v2_graph_free_Qwen_Qwen3_8B_prime_swe_grep",
            "spectra": "runs/stage1_v2_graph_free_spectra_Qwen_Qwen3_8B_prime_swe_grep",
            "stage3": "runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep",
        }
    return {
        "original": "runs/stage1_v2_original_graph_Qwen_Qwen3_8B_primeintellect_math500",
        "v2": "runs/stage1_v2_graph_free_Qwen_Qwen3_8B_primeintellect_math500",
        "spectra": "runs/stage1_v2_graph_free_spectra_Qwen_Qwen3_8B_primeintellect_math500",
        "stage3": "runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500",
    }


def _stage4_paths(env: str) -> str:
    if env == "swe":
        return "runs/stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b_pilot_rerun3"
    return "runs/stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b_pilot_rerun3"


def _stage5_path() -> str:
    return "runs/stage5_v2_Qwen_Qwen3_8B_tier_a_stage5"


def _stage5_short_method(method: str) -> str:
    return {
        "parametric_b": "B",
        "average_controllability_a": "A",
        "koopman_dmd_c": "C",
        "original_graph": "graph",
        "pca_proxy": "PCA",
        "random": "rand",
        "temperature": "temp",
        "sae": "SAE",
    }.get(method, method)


def _stage5_bar_row(row: dict[str, Any], *, conditioning: str) -> dict[str, Any]:
    method = str(row["method"])
    env = str(row["env"])
    return {
        "env": env,
        "method": method,
        "conditioning": conditioning,
        "status": str(row.get("status", "evaluated")),
        "label": f"{env[:4]}-{conditioning[:3]}-{_stage5_short_method(method)}",
        "identifiability": float(row.get("identifiability", 0.0)),
        "identifiability_ci_low": float(row.get("identifiability_ci_low", 0.0)),
        "identifiability_ci_high": float(row.get("identifiability_ci_high", 0.0)),
    }


def _pairwise_rows(pairwise: dict[str, float], labels: list[str], env: str) -> list[dict[str, Any]]:
    rows = [{"env": env, "x": label, "y": label, "value": 1.0} for label in labels]
    for key, value in pairwise.items():
        a, b = key.split("__")
        rows.append({"env": env, "x": a, "y": b, "value": float(value)})
        rows.append({"env": env, "x": b, "y": a, "value": float(value)})
    return rows


def build_data(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    figure_type = manifest["figure_type"]
    envs = manifest.get("envs", ["swe", "math500"])
    if figure_type == "chart_invariance":
        labels = ["PCA", "transition_PCA", "contrastive", "flow"]
        rows = []
        for env in envs:
            path = _stage1_paths(env)["original"]
            rows.extend(_pairwise_rows(_read_json(f"{path}/chart_invariance.json")["pairwise"], labels, env))
        return rows
    if figure_type == "discretization_invariance":
        labels = ["graph_laplacian", "landmark_nystrom_64", "landmark_nystrom_256", "parametric_neural"]
        rows = []
        for env in envs:
            path = _stage1_paths(env)["original"]
            rows.extend(_pairwise_rows(_read_json(f"{path}/discretization_invariance.json")["pairwise"], labels, env))
        return rows
    if figure_type == "sample_size_convergence":
        rows = []
        for env in envs:
            data = _read_json(f"{_stage1_paths(env)['original']}/sample_size_convergence.json")
            for size, overlap in data["overlaps"].items():
                rows.append({"env": env, "sample_size": int(size), "overlap": float(overlap)})
        return rows
    if figure_type == "layer_pair_heatmap":
        rows = []
        for env in envs:
            data = _read_json(f"{_stage1_paths(env)['original']}/layer_pair_heatmap.json")
            for row in data["rank_sweep_rows"]:
                rows.append({"env": env, "patch_layer": row["patch_layer"], "readout_layer": row["readout_layer"], "effective_rank": row["effective_rank_median"]})
        return rows
    if figure_type == "effective_rank_distribution":
        rows = []
        for env in envs:
            data = _read_json(f"{_stage1_paths(env)['original']}/effective_rank_distribution.json")
            for value in data["values"]:
                rows.append({"env": env, "value": float(value), "label": env})
        return rows
    if figure_type == "cumulative_gain_curve":
        rows = []
        for env in envs:
            values = _read_json(f"{_stage1_paths(env)['original']}/cumulative_gain_curve.json")["cumulative"]
            for idx, value in enumerate(values, start=1):
                rows.append({"env": env, "coordinate": idx, "cumulative_gain": float(value)})
        return rows
    if figure_type == "cross_formulation_comparison":
        rows = []
        for env in envs:
            data = _read_json(f"{_stage1_paths(env)['v2']}/method_comparison.json")
            rows.extend(
                [
                    {"env": env, "method": "B", "score": data["parametric_b"]["inverse_energy_effective_rank"], "status": "pass"},
                    {"env": env, "method": "A", "score": data["average_controllability_a"]["generalized_effective_rank"], "status": "pass"},
                    {"env": env, "method": "C", "score": data["koopman_dmd_c"]["test_r2"] * 30.0, "status": "pass"},
                    {"env": env, "method": "graph", "score": data["original_graph"]["metrics"]["sample_convergence_10000_vs_50000"] * 10.0, "status": "fail"},
                ]
            )
        return rows
    if figure_type.startswith("stage2_"):
        rows = []
        runs = [
            ("swe", "B", "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b"),
            ("swe", "A", "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_average_controllability_a"),
            ("swe", "C", "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_koopman_dmd_c"),
            ("swe", "graph", "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_original_graph"),
            ("math500", "B", "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b"),
            ("math500", "A", "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_average_controllability_a"),
            ("math500", "C", "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_koopman_dmd_c"),
            ("math500", "graph", "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_original_graph"),
        ]
        for env, method, run in runs:
            summary = _read_json(f"runs/{run}/summary.json")
            metrics = summary["metrics"]
            rows.append(
                {
                    "env": env,
                    "method": method,
                    "label": f"{env}-{method}",
                    "null_wasserstein": metrics["null_wasserstein"],
                    "random_accuracy": metrics["random_mean_accuracy"],
                    "chance": metrics["random_chance_accuracy"],
                    "chance_corrected": metrics["random_mean_accuracy"] - metrics["random_chance_accuracy"],
                }
            )
        return rows
    if figure_type.startswith("stage3_"):
        rows = []
        for env in envs:
            stage3 = _stage1_paths(env)["stage3"]
            if figure_type == "stage3_spectrum":
                for spectrum in _read_json(f"{stage3}/controllability_spectrum.json")["spectra"]:
                    for idx, value in enumerate(spectrum["values"], start=1):
                        rows.append({"env": env, "formulation": spectrum["formulation"], "coordinate": idx, "gamma": float(value)})
            else:
                for row in _read_json(f"{stage3}/aggregated_per_coordinate.json")["rows"]:
                    row = dict(row)
                    row["env"] = env
                    rows.append(row)
        return rows
    if figure_type == "stage4_trust_region":
        rows = []
        for env in envs:
            for row in _read_json(f"{_stage4_paths(env)}/trust_region_violation_rate_by_edit_norm.json")["rows"]:
                rows.append({"env": env, **row})
        return rows
    if figure_type == "stage4_displacement":
        rows = []
        for env in envs:
            for row in _load_parquet(ROOT / _stage4_paths(env) / "realized_vs_predicted_displacement.parquet"):
                rows.append({"env": env, **row})
        return rows
    if figure_type == "stage4_length":
        rows = []
        for env in envs:
            for row in _read_json(f"{_stage4_paths(env)}/trajectory_length_distribution_by_controller.json")["rows"]:
                rows.append({"env": env, **row})
        return rows
    if figure_type == "stage4_coherence":
        rows = []
        for env in envs:
            for row in _read_json(f"{_stage4_paths(env)}/coherence_rate_by_edit_norm.json")["rows"]:
                rows.append({"env": env, **row})
        return rows
    if figure_type == "stage5_identifiability_bars":
        run = _stage5_path()
        methods = {"parametric_b", "random", "pca_proxy", "temperature", "sae"}
        unconditional = _read_json(f"{run}/identifiability_unconditional.json")["rows"]
        coherent = _read_json(f"{run}/identifiability_coherence_conditioned.json")["rows"]
        rows = [_stage5_bar_row(row, conditioning="unconditional") for row in unconditional if row["method"] in methods]
        for row in coherent:
            if row["method"] == "parametric_b":
                rows.append(_stage5_bar_row(row, conditioning="coherent"))
            if row["method"] == "original_graph":
                proxy = dict(row)
                proxy["method"] = "pca_proxy"
                rows.append(_stage5_bar_row(proxy, conditioning="coherent"))
        for row in unconditional:
            if row["method"] in {"random", "temperature", "sae"}:
                rows.append(_stage5_bar_row(row, conditioning="coherent"))
        return rows
    if figure_type == "stage5_per_coordinate_identifiability":
        rows = []
        for row in _load_parquet(ROOT / _stage5_path() / "per_coordinate_identifiability.parquet"):
            row = dict(row)
            row["label"] = f"{row['env'][:4]}-k{int(row['coordinate'])}"
            rows.append(row)
        return rows
    if figure_type == "stage5_cross_formulation_identifiability":
        methods = {"parametric_b", "average_controllability_a", "koopman_dmd_c", "original_graph"}
        rows = []
        for row in _read_json(f"{_stage5_path()}/identifiability_coherence_conditioned.json")["rows"]:
            if row["method"] in methods:
                out = _stage5_bar_row(row, conditioning="coherent")
                out["label"] = f"{out['env'][:4]}-{_stage5_short_method(out['method'])}"
                rows.append(out)
        return rows
    if figure_type == "stage5_confusion_matrix_headline":
        rows = []
        for env in ["swe_grep", "math500"]:
            data = _read_json(f"{_stage5_path()}/confusion_matrices/{env}_parametric_b.json")
            for truth, values in enumerate(data["matrix"]):
                for predicted, value in enumerate(values):
                    rows.append({"env": env, "truth": truth, "predicted": predicted, "value": float(value)})
        return rows
    if figure_type == "stage5_quality_conditioning_sensitivity":
        rows = []
        for row in _read_json(f"{_stage5_path()}/quality_conditioning_sensitivity.json")["rows"]:
            if row["method"] == "parametric_b":
                rows.append(
                    {
                        "env": row["env"],
                        "quality_floor": float(row["quality_floor"]),
                        "identifiability": float(row.get("identifiability", 0.0)),
                        "status": row.get("status", "evaluated"),
                    }
                )
        return rows
    if figure_type == "phase1_per_controller_identifiability":
        rows = []
        for summary_path in sorted((ROOT / "runs").glob("phase1_per_controller_*/summary.json")):
            summary = _read_json(summary_path)
            env = summary["metrics"]["env"]
            formulation = summary["metrics"]["formulation"]
            for controller, result in summary["metrics"]["controllers"].items():
                rows.append(
                    {
                        "env": env,
                        "formulation": formulation,
                        "controller": controller,
                        "method": controller,
                        "status": result.get("status", "not_evaluable"),
                        "label": f"{env[:4]}-{_stage5_short_method(formulation)}-{controller[:5]}",
                        "identifiability": float(result.get("identifiability", 0.0)),
                        "identifiability_ci_low": float(result.get("identifiability_ci_low", 0.0)),
                        "identifiability_ci_high": float(result.get("identifiability_ci_high", 0.0)),
                    }
                )
        return rows
    if figure_type == "phase1_sign_asymmetry":
        rows = []
        for summary_path in sorted((ROOT / "runs").glob("phase1_sign_as_class_*/summary.json")):
            summary = _read_json(summary_path)
            env = summary["metrics"]["env"]
            formulation = summary["metrics"]["formulation"]
            for class_scheme, result in [
                ("K", summary["metrics"]["k_class"]),
                ("2K", summary["metrics"]["sign_as_class"]),
            ]:
                rows.append(
                    {
                        "env": env,
                        "formulation": formulation,
                        "class_scheme": class_scheme,
                        "method": class_scheme,
                        "status": result.get("status", "not_evaluable"),
                        "label": f"{env[:4]}-{_stage5_short_method(formulation)}-{class_scheme}",
                        "identifiability": float(result.get("identifiability", 0.0)),
                        "identifiability_ci_low": float(result.get("identifiability_ci_low", 0.0)),
                        "identifiability_ci_high": float(result.get("identifiability_ci_high", 0.0)),
                    }
                )
        return rows
    if figure_type == "phase1_quality_conditioning":
        rows = []
        path = ROOT / "runs/phase1_quality_conditioned/quality_conditioned_rows.json"
        if path.exists():
            rows = _read_json(path)["rows"]
            for row in rows:
                row["label"] = f"{row['env'][:4]}-{_stage5_short_method(row['formulation'])}-{row['quality_floor']}"
                row["method"] = row["formulation"]
        return rows
    if figure_type == "phase1_encoder_feature_variance":
        path = ROOT / "runs/phase1_encoder_audit/encoder_feature_variance.parquet"
        rows = _load_parquet(path) if path.exists() else []
        for row in rows:
            row["label"] = f"{row['env'][:4]}-{_stage5_short_method(row['formulation'])}"
            row["method"] = row["formulation"]
        return rows
    raise ValueError(f"unknown figure_type: {figure_type}")


def render_figure(fig_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> Canvas:
    figure_type = manifest["figure_type"]
    title = manifest.get("title", figure_type.replace("_", " ").title())
    if figure_type in {"chart_invariance", "discretization_invariance"}:
        averaged = []
        keys = sorted({(row["x"], row["y"]) for row in rows})
        for x, y in keys:
            vals = [float(row["value"]) for row in rows if row["x"] == x and row["y"] == y]
            averaged.append({"x": x, "y": y, "value": float(np.mean(vals))})
        return _draw_heatmap(averaged, title)
    if figure_type == "sample_size_convergence":
        return _draw_lines(rows, title, xkey="sample_size", ykey="overlap", groupkey="env", logx=True)
    if figure_type == "layer_pair_heatmap":
        canvas = _chart_frame(title)
        x0, y0, w, h = 80, 72, 560, 320
        vals = [float(r["effective_rank"]) for r in rows]
        patches = sorted({int(r["patch_layer"]) for r in rows})
        readouts = sorted({int(r["readout_layer"]) for r in rows})
        vmin, vmax = min(vals), max(vals)
        for row in rows:
            px = patches.index(int(row["patch_layer"]))
            ry = readouts.index(int(row["readout_layer"]))
            x = x0 + px * (w / max(len(patches), 1))
            y = y0 + ry * (h / max(len(readouts), 1))
            canvas.rect(x, y, 36, 26, fill=_heat_color(float(row["effective_rank"]), vmin, vmax), stroke="#ffffff")
        canvas.text(x0, y0 + h + 34, "x: patch layer, y: readout layer, color: effective rank", size=11, fill=COLORS["gray"])
        return canvas
    if figure_type == "effective_rank_distribution":
        return _draw_bars(rows, title, labelkey="label", valuekey="value", groupkey="env")
    if figure_type == "cumulative_gain_curve":
        return _draw_lines(rows, title, xkey="coordinate", ykey="cumulative_gain", groupkey="env")
    if figure_type == "cross_formulation_comparison":
        return _draw_bars(rows, title, labelkey="method", valuekey="score", groupkey="method")
    if figure_type == "stage2_null":
        return _draw_bars(rows, title, labelkey="label", valuekey="null_wasserstein", groupkey="method")
    if figure_type == "stage2_random":
        return _draw_bars(rows, title, labelkey="label", valuekey="random_accuracy", groupkey="method")
    if figure_type == "stage2_baseline_profile":
        return _draw_bars(rows, title, labelkey="label", valuekey="chance_corrected", groupkey="method")
    if figure_type == "stage3_scatter":
        return _draw_scatter(rows, title, xkey="rho_mean", ykey="gamma_mean", groupkey="formulation")
    if figure_type == "stage3_eta":
        return _draw_bars(rows, title, labelkey="formulation", valuekey="eta_star_mean", groupkey="formulation")
    if figure_type == "stage3_reachability":
        return _draw_bars(rows, title, labelkey="formulation", valuekey="rho_mean", groupkey="formulation")
    if figure_type == "stage3_spectrum":
        return _draw_lines(rows, title, xkey="coordinate", ykey="gamma", groupkey="formulation")
    if figure_type == "stage3_cross_formulation":
        return _draw_bars(rows, title, labelkey="formulation", valuekey="gamma_mean", groupkey="formulation")
    if figure_type == "stage4_trust_region":
        return _draw_lines(rows, title, xkey="edit_norm", ykey="violation_rate", groupkey="env")
    if figure_type == "stage4_displacement":
        return _draw_scatter_linear(rows, title, xkey="predicted_displacement", ykey="measured_displacement", groupkey="env")
    if figure_type == "stage4_length":
        grouped = []
        for env in sorted({str(row["env"]) for row in rows}):
            for controller in sorted({str(row["controller"]) for row in rows if str(row["env"]) == env}):
                values = [float(row["generated_tokens"]) for row in rows if str(row["env"]) == env and str(row["controller"]) == controller]
                grouped.append({"label": f"{env}-{controller}", "controller": controller, "mean_generated_tokens": float(np.mean(values))})
        return _draw_bars(grouped, title, labelkey="label", valuekey="mean_generated_tokens", groupkey="controller")
    if figure_type == "stage4_coherence":
        return _draw_lines(rows, title, xkey="edit_norm", ykey="coherence_rate", groupkey="env")
    if figure_type == "stage5_identifiability_bars":
        return _draw_bars_with_ci(
            rows,
            title,
            labelkey="label",
            valuekey="identifiability",
            lowkey="identifiability_ci_low",
            highkey="identifiability_ci_high",
            groupkey="method",
        )
    if figure_type == "stage5_per_coordinate_identifiability":
        return _draw_bars_with_ci(
            rows,
            title,
            labelkey="label",
            valuekey="identifiability",
            lowkey="ci_low",
            highkey="ci_high",
            groupkey="env",
        )
    if figure_type == "stage5_cross_formulation_identifiability":
        return _draw_bars_with_ci(
            rows,
            title,
            labelkey="label",
            valuekey="identifiability",
            lowkey="identifiability_ci_low",
            highkey="identifiability_ci_high",
            groupkey="method",
        )
    if figure_type == "stage5_confusion_matrix_headline":
        return _draw_stage5_confusion(rows, title)
    if figure_type == "stage5_quality_conditioning_sensitivity":
        canvas = _draw_lines(rows, title, xkey="quality_floor", ykey="identifiability", groupkey="env")
        if any(row.get("status") != "evaluated" for row in rows):
            canvas.text(72, 430, "Floors above zero have insufficient data in reduced Stage 4 runs because quality is unset.", size=10, fill=COLORS["gray"])
        return canvas
    if figure_type == "phase1_per_controller_identifiability":
        return _draw_bars_with_ci(
            rows,
            title,
            labelkey="label",
            valuekey="identifiability",
            lowkey="identifiability_ci_low",
            highkey="identifiability_ci_high",
            groupkey="controller",
        )
    if figure_type == "phase1_sign_asymmetry":
        return _draw_bars_with_ci(
            rows,
            title,
            labelkey="label",
            valuekey="identifiability",
            lowkey="identifiability_ci_low",
            highkey="identifiability_ci_high",
            groupkey="class_scheme",
        )
    if figure_type == "phase1_quality_conditioning":
        canvas = _draw_lines(rows, title, xkey="quality_floor", ykey="identifiability", groupkey="formulation")
        if any(row.get("status") != "evaluated" for row in rows):
            canvas.text(72, 430, "Quality is unset in reduced Stage 4 rows; strict Q > q* filters are not evaluable.", size=10, fill=COLORS["gray"])
        return canvas
    if figure_type == "phase1_encoder_feature_variance":
        return _draw_bars(rows, title, labelkey="label", valuekey="active_feature_fraction", groupkey="formulation")
    raise ValueError(f"unknown figure_type: {figure_type}")


def build_figure(fig_dir: Path) -> None:
    manifest = json.loads((fig_dir / "manifest.json").read_text())
    rows = build_data(manifest)
    _write_parquet(fig_dir / "data.parquet", rows)
    canvas = render_figure(fig_dir, manifest, rows)
    canvas.save_svg(fig_dir / "figure.svg")
    canvas.save_pdf(fig_dir / "figure.pdf")
