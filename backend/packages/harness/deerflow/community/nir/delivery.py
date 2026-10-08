"""Offline reports and allowlisted, integrity-indexed NIR delivery archives."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path


def _regular(path: Path) -> bool:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Delivery source must not be a symlink")
    return path.is_file()


def _inline(value: str) -> str:
    escaped = html.escape(value)
    # Only explicit HTTPS links become active; all markup is escaped first.
    escaped = re.sub(r"\[([^\]]+)\]\((https://[^\s)]+)\)", r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', escaped)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def _render(markdown: str, figures: dict[str, Path]) -> str:
    """Render the report's limited Markdown dialect; escape all raw HTML."""
    result, table = [], False
    for line in markdown.splitlines():
        if line.startswith("|"):
            if re.fullmatch(r"[| :\-]+", line):
                continue
            if not table:
                result.append("<table>")
                table = True
            cells = re.split(r"(?<!\\)\|", line.strip("|"))
            result.append("<tr>" + "".join(f"<td>{_inline(cell.strip().replace(chr(92) + '|', '|'))}</td>" for cell in cells) + "</tr>")
            continue
        if table:
            result.append("</table>")
            table = False
        picture = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        heading = re.match(r"^(#{1,4}) (.*)$", line)
        if picture and picture[2] in figures:
            encoded = base64.b64encode(figures[picture[2]].read_bytes()).decode("ascii")
            result.append(f'<figure><img alt="{html.escape(picture[1], quote=True)}" src="data:image/png;base64,{encoded}"><figcaption>{_inline(picture[1])}</figcaption></figure>')
        elif heading:
            level = len(heading[1])
            result.append(f"<h{level}>{_inline(heading[2])}</h{level}>")
        elif line:
            result.append(f"<p>{_inline(line)}</p>")
    if table:
        result.append("</table>")
    return "\n".join(result)


def build_delivery(report: Path, *, model: Path | None = None, metrics: Path | None = None, figures: list[Path] | None = None, additional: list[tuple[str, Path]] | None = None) -> dict[str, Path]:
    """Package explicitly supplied outputs; never walk a directory recursively."""
    report = Path(report)
    if not _regular(report):
        raise ValueError("Report does not exist")
    root = report.parent
    entries = [(report.name, report)]
    if model:
        model = Path(model)
        entries += [(f"model/{model.name}", model), (f"model/{model.name}.manifest.json", Path(str(model) + ".manifest.json"))]
    if metrics:
        entries.append((f"evidence/{Path(metrics).name}", Path(metrics)))
    plot_map = {}
    for plot in figures or []:
        plot = Path(plot)
        if _regular(plot) and plot.suffix == ".png" and plot.resolve().is_relative_to(root.resolve()):
            relative = plot.relative_to(root).as_posix()
            plot_map[relative] = plot
            entries.append((relative, plot))
    entries += additional or []
    entries = [(name, Path(path)) for name, path in entries if _regular(Path(path))]
    names = set()
    for name, _ in entries:
        if name in names or name.startswith("/") or ".." in Path(name).parts or "\\" in name:
            raise ValueError("Unsafe or duplicate delivery archive name")
        names.add(name)
    body = _render(report.read_text(encoding="utf-8"), plot_map)
    html_path = root / (report.stem + ".html")
    html_text = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NIR 建模报告</title><style>'
        "body{font:16px/1.8 system-ui,sans-serif;color:#1e293b;background:#fff;margin:40px auto;padding:0 32px;max-width:1120px}"
        "h1{border-bottom:3px solid #0f766e;padding-bottom:16px}h2{margin-top:42px;color:#0f766e}"
        "table{width:100%;border-collapse:collapse;margin:20px 0;font-variant-numeric:tabular-nums}"
        "td{border:1px solid #cbd5e1;padding:10px}tr:first-child{background:#f1f5f9;font-weight:600}"
        "img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{color:#64748b}p,td{overflow-wrap:anywhere}code{background:#f1f5f9;padding:2px 4px}"
        "@media print{body{margin:0;padding:0;font-size:11pt}h2{break-after:avoid}figure,tr{break-inside:avoid}}</style><body>" + body + "</body></html>"
    )
    fd, temporary = tempfile.mkstemp(dir=root, suffix=".html.part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(html_text)
        os.replace(temporary, html_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    entries.append((html_path.name, html_path))
    bundle = root / "delivery.zip"
    fd, temporary = tempfile.mkstemp(dir=root, suffix=".zip.part")
    os.close(fd)
    try:
        records = []
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in entries:
                payload = path.read_bytes()
                archive.writestr(name, payload)
                records.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            archive.writestr("delivery_manifest.json", json.dumps({"schema_version": 1, "files": records}, ensure_ascii=False, indent=2))
            archive.writestr("README.txt", "打开根目录HTML查看完整报告。model/含模型与原安全清单，evidence/含指标。校验信息见delivery_manifest.json。质量达标不等同于生产批准。\n")
        os.replace(temporary, bundle)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"html": html_path, "bundle": bundle}


def delivery_payload(report: str, model: str, metrics: str, *, figures: list[str] | None = None, virtual_dir: str) -> dict:
    """Compact pointers; report bodies and binaries never enter tool results."""
    paths = build_delivery(Path(report), model=Path(model), metrics=Path(metrics), figures=[Path(value) for value in figures or []])
    html_virtual = virtual_dir.rstrip("/") + "/" + paths["html"].name
    bundle_virtual = virtual_dir.rstrip("/") + "/" + paths["bundle"].name
    return {"report_html": html_virtual, "delivery_bundle": bundle_virtual, "deliverables": [html_virtual, bundle_virtual]}
