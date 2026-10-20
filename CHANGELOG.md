# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere al versionado semántico [Semantic Versioning](https://semver.org/lang/es/).

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
