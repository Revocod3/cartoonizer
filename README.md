# Cartoonizer

Aplicación local para procesar 40–50 capturas, revisar versiones y seleccionar 20–30 imágenes publicables. Conserva originales, decisiones, consumo y artefactos en SQLite y en el directorio `data/`.

## Instalación

VTracer 0.6.15 falla en Python 3.14 en macOS; usa Python 3.12 o 3.13.

```bash
brew install cairo
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Abre la URL que muestra Streamlit. Puedes consultar lotes guardados, revisar versiones y exportar resultados sin configurar la API.

## OpenAI

Configura la clave solo en tu terminal; no la pegues en la interfaz ni la guardes en el repositorio.

```bash
export OPENAI_API_KEY="tu-clave"
streamlit run app.py
```

También puedes guardarla en un `.env` local (ese archivo está excluido de Git):

```bash
set -a
source .env
set +a
streamlit run app.py
```

La aplicación utiliza OpenAI directamente al procesar una primera pasada o crear una corrección. El pipeline utiliza:

- GPT-5.6 Luna con `reasoning.effort: none` para análisis estructurado.
- GPT Image 2 para edición en PNG. Los lotes nuevos usan calidad `low` por defecto y permiten seleccionar `medium` al crearlos.
- GPT-5.6 Terra con `reasoning.effort: none` para evaluar raster y render SVG.
- VTracer y CairoSVG para vectorizar y verificar el SVG.

Una generación interrumpida queda como `generation_uncertain` y no se repite automáticamente. Los fallos de análisis, evaluación o vectorización se aíslan por imagen, de modo que el resto del lote continúa.

## Pruebas

```bash
.venv/bin/python -m unittest discover -s tests -v
```

La descarga final contiene el SVG maestro, PNG redimensionado con proporción conservada e informe JSON para cada versión aceptada.
