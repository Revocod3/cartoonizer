from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from cartoonizer.core import BatchWorker, Database, SimulatedPipeline, export_version, render_svg
from cartoonizer.providers import OpenAIPipeline


ROOT = Path(os.getenv("CARTOONIZER_DATA_DIR", Path(__file__).parent / "data"))
DB = Database(ROOT / "cartoonizer.sqlite")

st.set_page_config(page_title="Cartoonizer", page_icon="✦", layout="wide")
st.markdown("""
<style>
:root { --ink:#17202a; --muted:#52606d; --line:#cbd2d9; --surface:#ffffff; --accent:#275dad; }
.stApp { background:#f7f8fa; color:var(--ink); }
.block-container { max-width:1280px; padding-top:1.6rem; }
h1,h2,h3 { letter-spacing:-.025em; }
.hero { border-bottom:2px solid var(--accent); padding-bottom:1.2rem; margin-bottom:1.4rem; }
.hero h1 { color:var(--ink); font-size:3rem; margin:.15rem 0; }
.hero > div:last-child { color:var(--muted); font-size:1.05rem; }
.eyebrow { font-size:.78rem; font-weight:650; color:var(--muted); }
.metric { background:var(--surface); border:1px solid var(--line); border-radius:.5rem; padding:.85rem 1rem; min-height:92px; }
.metric strong { color:var(--ink); display:block; font-size:1.8rem; line-height:1.25; }
button:focus-visible, input:focus-visible, textarea:focus-visible { outline:3px solid #8bb7f0 !important; outline-offset:2px; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">Mesa local de selección</div><h1>Cartoonizer</h1><div>Primera pasada completa; corrige solo los candidatos prometedores.</div></div>', unsafe_allow_html=True)


def normalize_upload(upload, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(upload) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        image.save(destination.with_suffix(".png"), "PNG")


def preview_image(path: str | Path, background: str):
    from PIL import ImageDraw

    with Image.open(path) as source:
        image = source.convert("RGBA")
    if background == "Transparente":
        tile = Image.new("RGBA", image.size, "white")
        draw = ImageDraw.Draw(tile)
        for y in range(0, image.height, 16):
            for x in range(0, image.width, 16):
                draw.rectangle((x, y, x + 15, y + 15), fill="#e0e0e0" if (x // 16 + y // 16) % 2 else "#fafafa")
        tile.alpha_composite(image)
        return tile
    base = Image.new("RGBA", image.size, "white" if background == "Blanco" else "#202624")
    base.alpha_composite(image)
    return base


def pipeline_for(mode: str):
    if mode == "OpenAI real":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Define OPENAI_API_KEY antes de usar el flujo real.")
        return OpenAIPipeline()
    return SimulatedPipeline()


with st.sidebar:
    st.markdown("### Lote activo")
    batches = DB.batches()
    labels = {row["id"]: f"{row['name']} · {row['created_at'][:10]}" for row in batches}
    existing = st.selectbox("Recuperar lote", [""] + list(labels), format_func=lambda value: labels.get(value, "Nuevo lote"))
    if existing:
        st.session_state["batch_id"] = existing
    st.divider()
    name = st.text_input("Nombre", "Piloto de hoy")
    uploads = st.file_uploader("Capturas", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    background = st.selectbox("Fondo", ["transparent", "white", "dark"])
    cleanup = st.select_slider("Limpieza", options=["soft", "medium", "strong"], value="soft")
    colour = st.selectbox("Color", ["faithful", "slightly simplified", "flat"])
    detail = st.select_slider("Detalle", options=["low", "balanced", "high"], value="high")
    image_quality = st.radio("Calidad GPT Image 2", ["low", "medium"], index=1, horizontal=True)
    vector_mode = st.selectbox("Vectorización", ["detail", "curves"], format_func=lambda value: "Conservar detalle" if value == "detail" else "Simplificar curvas")
    notes = st.text_area("Instrucciones comunes", placeholder="Conserva expresión, colores y elementos visibles…")
    if st.button("Crear lote", type="primary", width="stretch"):
        if not uploads:
            st.error("Añade al menos una imagen.")
        else:
            settings = {"preset": "Anime fiel", "background": background, "cleanup": cleanup, "colour": colour, "detail": detail, "image_quality": image_quality, "vector_mode": vector_mode, "notes": notes}
            batch_id = DB.create_batch(name, settings)
            for index, upload in enumerate(uploads):
                target = ROOT / batch_id / "originals" / f"{index + 1:03d}-{Path(upload.name).stem}.png"
                normalize_upload(upload, target)
                DB.add_job(batch_id, str(target))
            st.session_state["batch_id"] = batch_id
            st.rerun()

batch_id = st.session_state.get("batch_id")
if not batch_id:
    st.info("Crea un lote o recupera uno anterior para empezar.")
    st.stop()

batch = DB.batch(batch_id)
report = DB.batch_report(batch_id)
jobs = report["jobs"]
pending = sum(job["stage"] not in ("ready", "failed", "generation_uncertain") for job in jobs)
reviewed = sum(job["decision"] is not None for job in jobs)

cols = st.columns(4)
for col, label, value in zip(cols, ["Pendientes", "Revisadas", "Seleccionadas", "Listas"], [pending, reviewed, report["counts"]["selected"], report["counts"]["ready"]]):
    col.markdown(f'<div class="metric"><span class="eyebrow">{label}</span><strong>{value}</strong></div>', unsafe_allow_html=True)

control, status = st.columns([1, 2])
with control:
    mode = st.selectbox("Motor", ["Simulado", "OpenAI real"], help="El modo simulado no consume API y permite comprobar la interfaz.")
with status:
    st.caption("Configuración efectiva: " + json.dumps(batch["settings"], ensure_ascii=False))
    if report["counts"]["failed"]:
        st.warning(f"{report['counts']['failed']} imágenes requieren atención por un fallo.")
if st.button("Procesar o reanudar primera pasada", type="primary", disabled=pending == 0):
    try:
        completed = BatchWorker(DB, pipeline_for(mode), ROOT / batch_id / "artifacts").run_batch(batch_id)
        st.success(f"Se completaron {completed} imágenes.")
        st.rerun()
    except Exception as exc:
        st.error(str(exc))
if report["counts"]["failed"] and st.button("Reintentar trabajos fallidos"):
    retried = DB.retry_failed(batch_id)
    st.success(f"{retried} trabajos preparados para continuar desde la última etapa guardada.")
    st.rerun()

review_tab, export_tab, usage_tab = st.tabs(["Revisión", "Exportar", "Consumo"])
with review_tab:
    category_filter = st.multiselect("Categorías", ["no_defects", "possible_correction", "requires_attention", "sin evaluación"], default=["no_defects", "possible_correction", "requires_attention", "sin evaluación"])
    preview_background = st.radio("Fondo de previsualización", ["Transparente", "Blanco", "Oscuro"], horizontal=True)
    ready_jobs = [job for job in DB.jobs(batch_id) if DB.versions(job["id"])]
    bulk_ids = st.multiselect("Selección múltiple", [job["id"] for job in ready_jobs], format_func=lambda job_id: Path(next(job["original_path"] for job in ready_jobs if job["id"] == job_id)).name)
    bulk_accept, bulk_discard = st.columns(2)
    if bulk_accept.button("Aceptar últimas versiones", disabled=not bulk_ids):
        for job_id in bulk_ids:
            DB.accept_version(job_id, DB.versions(job_id)[0]["id"])
        st.rerun()
    if bulk_discard.button("Descartar selección", disabled=not bulk_ids):
        for job_id in bulk_ids:
            DB.decide(job_id, "discarded")
        st.rerun()
    for job in DB.jobs(batch_id):
        evaluation = json.loads(job["evaluation"]) if job["evaluation"] else {}
        category = evaluation.get("category", "sin evaluación")
        if category not in category_filter:
            continue
        with st.container(border=True):
            st.markdown(f"#### {Path(job['original_path']).name} · {category}")
            if job["error"]:
                st.error(job["error"])
            visual = st.columns(3)
            visual[0].image(preview_image(job["original_path"], preview_background), caption="Original", width="stretch")
            if job["candidate_path"] and Path(job["candidate_path"]).exists():
                visual[1].image(preview_image(job["candidate_path"], preview_background), caption="Raster", width="stretch")
            if job["svg_path"] and Path(job["svg_path"]).exists():
                rendered = Path(job["svg_path"]).with_suffix(".preview.png")
                if not rendered.exists():
                    render_svg(Path(job["svg_path"]), rendered, 800)
                visual[2].image(preview_image(rendered, preview_background), caption="SVG renderizado", width="stretch")
            with st.expander("Instrucciones, evaluación y versiones"):
                st.json({"analysis": json.loads(job["analysis"]) if job["analysis"] else None, "evaluation": evaluation, "versions": DB.versions(job["id"])})
                overrides = json.loads(job["individual_settings"])
                override_background = st.selectbox("Fondo individual", ["usar ajuste común", "transparent", "white", "dark"], index=(["usar ajuste común", "transparent", "white", "dark"].index(overrides.get("background", "usar ajuste común"))), key=f"background-{job['id']}")
                override_notes = st.text_area("Instrucciones individuales", value=overrides.get("notes", ""), key=f"instructions-{job['id']}")
                if st.button("Guardar ajustes individuales", key=f"settings-{job['id']}", disabled=job["stage"] != "pending"):
                    DB.update_job_settings(job["id"], {**({"background": override_background} if override_background != "usar ajuste común" else {}), **({"notes": override_notes} if override_notes else {})})
                    st.rerun()
            versions = DB.versions(job["id"])
            selected_version = st.selectbox("Versión", versions, format_func=lambda value: f"v{value['number']}" + (" · aceptada" if value["accepted"] else ""), key=f"version-{job['id']}") if versions else None
            action_cols = st.columns(3)
            if action_cols[0].button("Aceptar versión", key=f"accept-{job['id']}", disabled=selected_version is None):
                DB.accept_version(job["id"], selected_version["id"])
                st.rerun()
            if action_cols[1].button("Descartar", key=f"discard-{job['id']}"):
                DB.decide(job["id"], "discarded")
                st.rerun()
            reasons = st.multiselect("Motivos de corrección", ["expresión", "detalles", "colores", "líneas", "fondo", "texto"], key=f"reasons-{job['id']}")
            note = st.text_input("Nota libre", key=f"note-{job['id']}")
            if action_cols[2].button("Crear corrección", key=f"correct-{job['id']}", disabled=not versions):
                correction = ", ".join(reasons + ([note] if note else []))
                try:
                    BatchWorker(DB, pipeline_for(mode), ROOT / batch_id / "artifacts").correct(batch_id, job["id"], correction)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

with export_tab:
    long_side = st.number_input("Lado mayor del PNG", min_value=512, max_value=12000, value=6000, step=100)
    accepted = [(job, DB.accepted_version(job["id"])) for job in jobs]
    accepted = [(job, version) for job, version in accepted if version]
    st.write(f"{len(accepted)} versiones aceptadas")
    if st.button("Preparar paquete", disabled=not accepted):
        export_root = ROOT / batch_id / "export"
        for index, (job, version) in enumerate(accepted, 1):
            with Image.open(version["candidate_path"]) as image:
                width, height = image.size
            if width >= height:
                export_width, export_height = int(long_side), round(int(long_side) * height / width)
            else:
                export_height, export_width = int(long_side), round(int(long_side) * width / height)
            export_version(Path(version["candidate_path"]), Path(version["svg_path"]), export_root, f"{index:03d}-{Path(job['original_path']).stem}", export_width, export_height, {"batch_id": batch_id, "job_id": job["id"], "version": version["number"], "evaluation": json.loads(version["evaluation"]), "usage": report["usage"]})
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
            for path in export_root.iterdir():
                package.write(path, path.name)
        st.download_button("Descargar ZIP", archive.getvalue(), file_name=f"{batch['name']}.zip", mime="application/zip")

with usage_tab:
    st.json(report["usage"] or {"mensaje": "Todavía no hay consumo registrado"})
    token_total = sum(row["input_tokens"] + row["output_tokens"] for row in report["usage"].values())
    attempts = sum(row["image_attempts"] for row in report["usage"].values())
    if report["counts"]["total"] and (token_total or attempts):
        per_image_tokens = token_total / report["counts"]["total"]
        per_image_attempts = attempts / report["counts"]["total"]
        st.table({"Lote": [40, 50], "Tokens estimados": [round(per_image_tokens * 40), round(per_image_tokens * 50)], "Generaciones estimadas": [round(per_image_attempts * 40, 1), round(per_image_attempts * 50, 1)]})
    st.download_button("Descargar informe JSON", json.dumps(DB.batch_report(batch_id), indent=2, ensure_ascii=False), file_name=f"{batch['name']}-report.json", mime="application/json")
