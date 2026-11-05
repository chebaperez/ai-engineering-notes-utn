# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere al versionado semántico [Semantic Versioning](https://semver.org/lang/es/).

---

## [1.5.0] - 2026-11-05

### Añadido
- **Scripts de Demostración en Vivo para Clase 3 (`src/class_3/`):** Suite didáctica modular de 3 scripts punta a punta alineados con las filminas y el caso de estudio de Vertex Horizon Seguros:
  - `src/class_3/01_chunking_and_vector_db.py`: Segmentación fija vs. solapada (overlap), embeddings multidimensionales, base vectorial en memoria y telemetría de delay (Filminas 6, 7, 9 y 10).
  - `src/class_3/02_retrieval_and_hybrid_search.py`: Puntos ciegos de Búsqueda Densa vs. Léxica (BM25) y Búsqueda Híbrida con Fusión Recíproca de Rangos (RRF) (Filminas 11, 12, 13 y 14).
  - `src/class_3/03_rag_pipeline_and_grounding.py`: Examen a libro cerrado vs. RAG fundamentado con citas auditables y Corrective RAG (CRAG) con filtro evaluador de similitud previa (Filminas 5, 8, 15, 16 y 17).
- **Ajuste del Modelo por Defecto:** Configuración y documentación de `gemini/gemini-3.5-flash-lite` como modelo generador por defecto en todos los scripts y plantillas de entorno (`.env.example`), por ofrecer la cuota gratuita más generosa y amplia para live coding.
- **Refinamiento en Diapositivas:** Actualización en la filmina 16 de `presentaciones/utn_class_3_presentation_system.html` sobre el límite de Corrective RAG (ruido temático vs. huecos documentales).

---

## [1.4.0] - 2026-10-27

### Añadido
- **Scripts de Demostración en Vivo (`src/`):** Suite de scripts en Python gestionada con `uv` y `LiteLLM` para demostraciones en vivo durante las clases teóricas:
  - `src/class_2/01_chain_of_thought.py`: Inferencia directa vs. Zero-Shot CoT vs. Few-Shot CoT (Filminas 12 y 13).
  - `src/class_2/02_chaining_and_metaprompting.py`: Pipeline encadenado de 3 pasos y Meta-Prompting (Filmina 14).
  - `src/class_2/03_structured_outputs.py`: Los 3 niveles de estructuración con validación tipada vía Pydantic (Filmina 15).
- **Gestión Moderna de Dependencias:** Soporte nativo para `uv` (`pyproject.toml`, `requirements.txt`) y reintentos automáticos con `tenacity`.
- **Documentación y Guías:** `src/README.md` con instrucciones paso a paso para obtener y gestionar claves gratuitas de Gemini en Google AI Studio.

---

## [1.3.0] - 2026-10-20

### Añadido
- **Laboratorios Prácticos:** Incorporación de los 4 cuadernos interactivos en Google Colab con soporte GPU T4 en `laboratorios/`:
  - `Laboratorio_1_Ecosistema_LLMs.ipynb` (Inferencia local, cuantización GGUF y TTFT)
  - `Laboratorio_2_Prompt_Engineering.ipynb` (In-context learning, Pydantic y gramáticas GBNF)
  - `Laboratorio_3_Sistemas_RAG.ipynb` (Pipeline RAG de punta a punta, ChromaDB y validación)
  - `Laboratorio_4_Herramientas_Validacion_Agentes.ipynb` (Tool Calling, ciclo ReAct y presupuestos)
- **Puntero Láser Interactivo:** Implementación de puntero láser sobre canvas con estela dinámica y toast de activación en las 4 presentaciones web (tecla <kbd>L</kbd>).
- **Recursos Visuales:** Diagrama tridimensional del espacio semántico de embeddings en `presentaciones/assets/espacio_semantico_embeddings.png`, integrado en la Unidad 1.

---

## [1.2.0] - 2026-09-17

### Añadido
- En `presentaciones/utn_class_1_presentation_system.html`, desdoblamiento de la introducción institucional en dos diapositivas consecutivas:
  - **Slide 02 (Estructura Temática: 4 Unidades y Laboratorios en GPU):** visualización espaciosa de unidades y pie de laboratorios en dos filas.
  - **Slide 03 (Proyecto Integrador: Challenge Competitivo & Leaderboard):** presentación del benchmark, tablero de mérito y filosofía de iteración técnica continua.

### Modificado
- Reindexación secuencial completa de slides en Unidad 1 (20 diapositivas totales).
- Readecuación del marco pedagógico y criterios de evaluación para la carrera Ingeniería en Sistemas de Información.

---

## [1.1.0] - 2026-09-17

### Añadido
- Generación de los 4 sistemas de presentación interactiva para la **UTN Facultad Regional San Rafael**:
  - `utn_class_1_presentation_system.html` (Unidad 1)
  - `utn_class_2_presentation_system.html` (Unidad 2)
  - `utn_class_3_presentation_system.html` (Unidad 3)
  - `utn_class_4_presentation_system.html` (Unidad 4)
- Identidad visual y diseño institucional UTN:
  - Paleta cromática oficial: Azul UTN (`#003882`), Celeste Institucional (`#0284C7`), fondo nocturno Midnight (`#0B132B`).
  - Tipografía técnica: `Montserrat` combinada con `Roboto` y `JetBrains Mono`.
- Canales independientes de sincronización `BroadcastChannel` para notas docentes de teleprompter en tiempo real.

### Modificado
- **Calibración de Escala Base:** Tamaño equivalente al 80% configurado como el **100% por defecto** (`--base-font-scale: 0.8`), eliminando necesidad de scroll vertical en proyectores.
- **Nomenclatura Curricular:** Denominación modular por **Unidades 1 a 4** adaptada al programa de Materia Electiva.

---

## [1.0.0] - 2026-09-17

### Añadido
- Programa analítico oficial de la materia electiva para la carrera Ingeniería en Sistemas de Información de la UTN FRSR (`programa/UTN-Programa_Desarrollo_Aplicaciones_LLMs.docx`).
