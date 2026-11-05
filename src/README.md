# 🧪 Scripts de Demostración en Vivo para Clases
### UTN Facultad Regional San Rafael · Electiva: Desarrollo de Aplicaciones con LLMs y Modelos Generativos

Esta carpeta contiene scripts prácticos en Python organizados por unidad/clase (`class_2/`, etc.) diseñados para ser ejecutados y modificados **en vivo durante la clase (en 5 a 10 minutos cada uno)**. Conectan la teoría con código ejecutable utilizando **LiteLLM** y la API de **Google Gemini** (tier gratuito).

Todo el entorno está preparado para ser gestionado de forma ultra rápida y simple con **`uv`** (o con `pip` tradicional).

---

## 📁 Estructura del Directorio `src/`

```text
src/
├── pyproject.toml                 # Configuración de proyecto y dependencias para uv
├── requirements.txt               # Dependencias en formato estándar pip
├── .env.example                   # Plantilla de variables de entorno (Gemini API Key)
├── README.md                      # Esta guía de uso y ejecución
│
├── class_2/                       # Scripts para la Unidad 2: Prompt Engineering
│   ├── 01_chain_of_thought.py     # Filminas 12 y 13: Direct vs. Zero-Shot CoT vs. Few-Shot CoT
│   ├── 02_chaining_and_metaprompting.py # Filmina 14: Pipelines de 3 pasos y Meta-Prompting
│   └── 03_structured_outputs.py   # Filmina 15: Nivel 1 (Prompt) vs Nivel 2 (JSON) vs Nivel 3 (Pydantic)
│
└── class_3/                       # Scripts para la Unidad 3: Sistemas RAG y Recuperación
    ├── 01_chunking_and_vector_db.py     # Filminas 6, 7, 9, 10: Chunking, Embeddings y Vector DB
    ├── 02_retrieval_and_hybrid_search.py# Filminas 11-14: Densa vs. Léxica (BM25) vs. Híbrida (RRF)
    └── 03_rag_pipeline_and_grounding.py # Filminas 5, 8, 15-17: Grounding, Citas y Corrective RAG
```

---

## ⚡ Modelo por Defecto: `gemini/gemini-3.5-flash-lite`

Para garantizar una experiencia de **Live Coding fluida y sin bloqueos de cuota**, todos los scripts utilizan por defecto:
- **Modelo Generador:** `gemini/gemini-3.5-flash-lite` (ofrece los límites de cuota gratuita por minuto y por día más amplios de Google AI Studio, con latencia ultra baja).
- **Modelo de Embeddings:** `gemini/gemini-embedding-001` (para representaciones vectoriales densas).

*(Si disponés de una clave de OpenAI, los scripts seleccionan automáticamente `openai/gpt-4o-mini` y `openai/text-embedding-3-small`).*

---

## ⚡ Instalación y Ejecución con `uv` (Recomendado)

[uv](https://docs.astral.sh/uv/) es el gestor de paquetes y entornos de Python más rápido del ecosistema. Permite ejecutar todo sin lidiar con configuraciones manuales de virtualenv.

### 1. Configurar la clave gratuita de Gemini (Solo una vez)
1. Obtené tu clave gratuita en [Google AI Studio](https://aistudio.google.com/) (botón **"Get API key"**).
2. Copiá la plantilla `.env.example` a `.env` en la raíz del repositorio o dentro de `src/`:
   ```bash
   cp src/.env.example .env
   ```
3. Editá `.env` y pegá tu clave:
   ```bash
   GEMINI_API_KEY=AIzaSy...
   ```

### 2. Ejecutar directamente con `uv run` (Cero configuración)
Desde la raíz del proyecto o dentro de `src/`, `uv` resolverá las dependencias automáticamente y ejecutará el script en un entorno aislado:

```bash
# Desde la carpeta src/:
cd src

# Unidad 2: Prompt Engineering
uv run python class_2/01_chain_of_thought.py
uv run python class_2/02_chaining_and_metaprompting.py
uv run python class_2/03_structured_outputs.py

# Unidad 3: Sistemas RAG y Recuperación
uv run python class_3/01_chunking_and_vector_db.py
uv run python class_3/02_retrieval_and_hybrid_search.py
uv run python class_3/03_rag_pipeline_and_grounding.py
```

*(O desde la raíz del repositorio: `uv run --project src python src/class_2/01_chain_of_thought.py`)*

### 3. Alternativa con `uv venv` y `uv pip` (Si preferís activar el entorno)
```bash
uv venv
source .venv/bin/activate
uv pip install -r src/requirements.txt

python src/class_2/01_chain_of_thought.py
```

---

## 🐢 Alternativa Tradicional: `pip` y `venv` estándar

Si no tenés `uv` instalado, podés usar el flujo tradicional de Python:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt

python src/class_2/01_chain_of_thought.py
```

---

## 🗺️ Mapa de Scripts de la Clase 2 y Filminas

| Script | Filminas | Conceptos Clave | Caso de Negocio (*FinTech Horizon*) |
| :--- | :--- | :--- | :--- |
| **[`class_2/01_chain_of_thought.py`](class_2/01_chain_of_thought.py)** | **12 y 13** | Inferencia Directa vs. Zero-Shot CoT (*"pensemos paso a paso"*) vs. Few-Shot CoT | Liquidación y comisiones compuestas con deducciones escalonadas |
| **[`class_2/02_chaining_and_metaprompting.py`](class_2/02_chaining_and_metaprompting.py)** | **14** | Descomposición de tareas (Chaining de 3 pasos) + Meta-Prompting con los 5 principios | Pipeline de auditoría y reclamo por cobros duplicados |
| **[`class_2/03_structured_outputs.py`](class_2/03_structured_outputs.py)** | **15** | Tres niveles de control: Prompting ingenuo vs. Modo JSON vs. Validación con Pydantic | Ingesta de reclamo a objeto fuertemente tipado (`TicketIncidenteFintech`) |

---

## 🗺️ Mapa de Scripts de la Clase 3 y Filminas

| Script | Filminas | Conceptos Clave | Caso de Negocio (*Vertex Horizon Seguros*) |
| :--- | :--- | :--- | :--- |
| **[`class_3/01_chunking_and_vector_db.py`](class_3/01_chunking_and_vector_db.py)** | **6, 7, 9 y 10** | Chunking Fijo Naive vs. Recursivo con Solapamiento (Overlap) + Embeddings densos + Base vectorial en memoria y telemetría de delay | Pérdida de excepciones legales (`CLUST-90`) ante chunking ciego en póliza Hogar Total |
| **[`class_3/02_retrieval_and_hybrid_search.py`](class_3/02_retrieval_and_hybrid_search.py)** | **11, 12, 13 y 14** | Puntos ciegos de Búsqueda Densa vs. Léxica (BM25) + Búsqueda Híbrida con Fusión Recíproca de Rangos (RRF) | Códigos técnicos alfanuméricos (`RCT-19`) vs. Paráfrasis coloquial de auxilio en ruta |
| **[`class_3/03_rag_pipeline_and_grounding.py`](class_3/03_rag_pipeline_and_grounding.py)** | **5, 8, 15, 16 y 17** | Examen a Libro Cerrado (Alucinación) vs. RAG Fundamentado con Citas obligatorias y abstención + Corrective RAG (CRAG) | El incidente histórico de la circular falsa de inundaciones rurales y preguntas capciosas |

---

## 🛠️ Ideas para interactuar en vivo en clase

### Unidad 2 (Prompt Engineering):
1. **En `01_chain_of_thought.py`**: Cambiá las reglas de comisión agregando un impuesto extra y observá cómo la inferencia directa se equivoca mientras CoT desglosa la cuenta correctamente.
2. **En `02_chaining_and_metaprompting.py`**: Modificá el requerimiento informal en la Parte 2 para que el modelo redacte un prompt de detección de fraude en tiempo real.
3. **En `03_structured_outputs.py`**: Agregá un nuevo atributo al modelo Pydantic (ej. `score_riesgo: int = Field(ge=1, le=10)`) y ejecutá para ver cómo LiteLLM y Gemini respetan el nuevo contrato de datos.

### Unidad 3 (Sistemas RAG):
1. **En `01_chunking_and_vector_db.py`**: Reducí el solapamiento (`overlap=0`) en la estrategia B y observá cómo el chunking mutila la exclusión `CLUST-90`, demostrando en vivo la causa del fallo de Vertex Horizon.
2. **En `02_retrieval_and_hybrid_search.py`**: Probá una nueva consulta inventada por un alumno que contenga una sigla inventada o un modismo regional y analizá qué motor la rescata primero y cómo pondera el RRF.
3. **En `03_rag_pipeline_and_grounding.py`**: Subí el umbral de CRAG de `0.55` a `0.80` y observá cómo el sistema se vuelve hiper-cauteloso; luego probá preguntar algo no presente en el manual para ver el mensaje estricto de abstención.

---

## 🔑 Cómo Obtener y Gestionar tu API Key en Google AI Studio

Para utilizar los modelos de Gemini no necesitás ingresar ninguna tarjeta de crédito. El nivel gratuito te permite experimentar para las clases y laboratorios con límites de solicitudes suficientes.

### 📝 Pasos para generar tu clave:
1. **Iniciar sesión:** Ingresá a [Google AI Studio](https://aistudio.google.com/) con tu cuenta de Google.
2. **Acceder a las claves:** En el menú lateral izquierdo o en el panel principal, hacé clic en **"Get API key"** (Obtener clave de API).
3. **Crear la clave:** Hacé clic en el botón azul **"Create API key"** (en la esquina superior derecha).
4. **Asignar proyecto:** Elegí *"Create API key in a new project"* (o seleccioná un proyecto de Google Cloud existente).
5. **Copiar y guardar:** Hacé clic en **"Copy key"** y pegala en tu archivo `.env`:
   ```bash
   GEMINI_API_KEY=AIzaSy...
   ```

### ⚠️ Consideraciones clave del nivel gratuito:
- **Límites de cuota (Rate Limits):** Cuenta con restricciones por minuto (RPM) y por día (RPD) según el modelo (`gemini-3.8-flash`, `gemini-1.5-flash`). Si se supera el límite en ráfaga, la API responderá con un error `429 (Resource Exhausted)` hasta que se reinicie la ventana temporal.
- **Privacidad y uso de datos:** En la capa gratuita, Google puede utilizar los datos de las solicitudes para evaluar y mejorar sus modelos. Para datos corporativos o información sensible en producción, se recomienda utilizar el nivel de pago donde los datos son estrictamente privados.
