# Desarrollo de Aplicaciones con LLMs y Modelos Generativos · UTN FRSR

Repositorio de contenidos académicos, sistemas de presentación interactiva, laboratorios prácticos en Google Colab y programa analítico para la materia electiva **Desarrollo de Aplicaciones con LLMs y Modelos Generativos**, dictada en la carrera **Ingeniería en Sistemas de Información** de la **Universidad Tecnológica Nacional (Facultad Regional San Rafael - UTN FRSR)**.

- **Docente:** Sebastián Pérez ([@chebaperez](https://github.com/chebaperez))
- **Institución:** Universidad Tecnológica Nacional · Facultad Regional San Rafael (UTN FRSR)
- **Carrera:** Ingeniería en Sistemas de Información (Materia Electiva)

---

## 📁 Estructura del Repositorio

```text
.
├── laboratorios/                     # Notebooks interactivos (.ipynb para Colab / Jupyter)
│   ├── Laboratorio_1_Ecosistema_LLMs.ipynb
│   ├── Laboratorio_2_Prompt_Engineering.ipynb
│   ├── Laboratorio_3_Sistemas_RAG.ipynb
│   └── Laboratorio_4_Herramientas_Validacion_Agentes.ipynb
│
├── presentaciones/                   # Sistemas de presentación interactiva HTML (UTN FRSR)
│   ├── assets/                       # Diagramas e ilustraciones esquemáticas
│   │   └── espacio_semantico_embeddings.png
│   ├── utn_class_1_presentation_system.html   # Unidad 1
│   ├── utn_class_2_presentation_system.html   # Unidad 2
│   ├── utn_class_3_presentation_system.html   # Unidad 3
│   └── utn_class_4_presentation_system.html   # Unidad 4
│
├── programa/                         # Programa analítico de la materia electiva
│   └── UTN-Programa_Desarrollo_Aplicaciones_LLMs.docx
│
├── src/                              # Scripts de demostración en vivo (Python / LiteLLM / uv)
│   ├── class_2/                      # Unidad 2: Prompt Engineering
│   │   ├── 01_chain_of_thought.py    # Filminas 12 y 13: Direct vs. Zero-Shot CoT vs. Few-Shot CoT
│   │   ├── 02_chaining_and_metaprompting.py # Filmina 14: Pipelines de 3 pasos y Meta-Prompting
│   │   └── 03_structured_outputs.py  # Filmina 15: Nivel 1 (Prompt) vs Nivel 2 (JSON) vs Nivel 3 (Pydantic)
│   ├── pyproject.toml                # Gestión moderna y rápida de dependencias con uv
│   ├── requirements.txt              # Dependencias en formato estándar pip
│   ├── .env.example                  # Plantilla de credenciales de Google Gemini
│   └── README.md                     # Guía de ejecución rápida en vivo con uv y pip
│
├── CHANGELOG.md                      # Historial de cambios y versiones del repositorio
├── README.md                         # Documentación general y guía de uso
├── .gitattributes                    # Configuración de git (merge/diff celdas Jupyter vía nbdime)
└── .gitignore                        # Exclusiones de Git (modelos, checkpoints, entornos)
```

---

## 📚 Unidades Temáticas y Prácticas

| Unidad | Tema Principal | Enfoque Teórico y de Arquitectura | Laboratorio Práctico (GPU T4) |
| :---: | :--- | :--- | :--- |
| **1** | **El Ecosistema de LLMs: Cloud vs. Pesos Abiertos** | Inferencia propietaria vs. autoalojada. VRAM, cuantización (GGUF), tokens/seg, cuellos de botella de hardware y balance de costos OpEx vs. CapEx. | Inferencia local con `llama-cpp-python`, cuantización de 4 a 8 bits y medición de tiempos al primer token (TTFT). |
| **2** | **Prompt Engineering y Estructuración de Datos** | In-context learning, fronteras de decisión, Chain-of-Thought (CoT) y cumplimiento determinista de esquema JSON vía gramáticas GBNF y muestreo restringido. | Clasificación estructurada multi-clase, parseo con Pydantic, mitigación de sesgo en few-shot y auditoría de alucinaciones. |
| **3** | **Sistemas RAG · Consulta Documental Fundamentada** | Segmentación (chunking), espacios vectoriales semánticos, bases vectoriales (ChromaDB), generación fundamentada con citas verificables y métricas de la tríada RAG. | Pipeline RAG completo de punta a punta, ingesta de manual de seguros, recuperación vectorial, evaluación de ranking y auditoría de citas falsas. |
| **4** | **Uso de Herramientas, Validación y Conceptos de Agentes** | Tool Calling estructurado, esquemas JSON de APIs, ciclo ReAct (Reason + Act), trazabilidad de ejecución y barreras de seguridad (Guardrails). | Implementación de agente acotado con herramientas de cálculo y base documental, con validación estricta de salida y control presupuestario. |

---

## 🏆 Proyecto Integrador: Challenge Competitivo & Leaderboard

El trayecto formativo de la materia electiva culmina con el desarrollo de un **Challenge Competitivo** integrador:
- **El Benchmark (Banco de Pruebas):** Evaluación cuantitativa sobre un dataset de validación ciego con métricas de precisión, latencia y presupuesto de cómputo.
- **El Leaderboard (Tablero de Mérito):** Tabla de clasificación viva donde los alumnos iteran sus soluciones durante las semanas de desarrollo intensivo.
- **Criterio Pedagógico:** La competencia es formativa y no punitiva; el ranking no define la aprobación, siendo requisito excluyente que la arquitectura sea funcional de punta a punta, respete el contrato de interfaz y satisfaga los controles de evaluación continua.

---

## 🖥️ Sistema de Presentaciones Interactivas

Las presentaciones son aplicaciones web autónomas e interactivas desarrolladas en HTML5 puro con Tailwind CSS y Mermaid.js, con identidad visual adaptada a la UTN Facultad Regional San Rafael.

### Características Principales
- **Identidad Institucional UTN:** Paleta oficial UTN (Azul UTN `#003882`, Celeste `#0284C7`, fondo nocturno Midnight `#0B132B`), tipografía técnica `Montserrat` y `Roboto`.
- **Calibración de Escala para Proyectores:** Escala base precalibrada al 80% como 100% por defecto (`--base-font-scale: 0.8`), garantizando visualización integral sin scroll en proyectores de aula y pantallas compactas.
- **Puntero Láser Integrado (<kbd>L</kbd>):** Renderizado en canvas con estela dinámica y alta visibilidad para exposiciones.
- **Teleprompter Sincronizado:** Ventana de notas docentes sincronizada vía `BroadcastChannel` para operación multi-pantalla sin servidor backend.

### Cómo abrirlas
- **Uso directo:** Abrir cualquier archivo `.html` de la carpeta `presentaciones/` en un navegador web moderno.
- **Servidor local (Recomendado):** Para habilitar la sincronización de notas docentes entre pestañas:
  ```bash
  python3 -m http.server 8080
  ```
  Luego acceder a:
  - Unidad 1: `http://localhost:8080/presentaciones/utn_class_1_presentation_system.html`
  - Unidad 2: `http://localhost:8080/presentaciones/utn_class_2_presentation_system.html`
  - Unidad 3: `http://localhost:8080/presentaciones/utn_class_3_presentation_system.html`
  - Unidad 4: `http://localhost:8080/presentaciones/utn_class_4_presentation_system.html`

### Atajos de Teclado y Controles
| Tecla | Acción | Descripción |
| :---: | :--- | :--- |
| <kbd>→</kbd> / <kbd>Espacio</kbd> | **Siguiente diapositiva** | Avanza al siguiente slide de la secuencia. |
| <kbd>←</kbd> | **Diapositiva anterior** | Retrocede al slide previo. |
| <kbd>L</kbd> | **Puntero Láser** | Activa / desactiva el puntero láser sobre el lienzo de proyección. |
| <kbd>P</kbd> | **Cronómetro Global** | Inicia / pausa el cronómetro de la sesión (2 horas). |
| <kbd>D</kbd> | **Modo Oscuro / Claro** | Alterna entre tema oscuro nocturno y tema claro. |
| <kbd>F</kbd> | **Pantalla Completa** | Activa o desactiva el modo de proyección a pantalla completa. |
| <kbd>N</kbd> | **Notas Docente** | Abre el drawer o ventana independiente de teleprompter sincronizado vía `BroadcastChannel`. |
| <kbd>+</kbd> / <kbd>-</kbd> | **Escala de Zoom** | Ajusta el tamaño de renderizado en tramos de 5% sobre la escala base. |

---

## 🧪 Ejecución de Laboratorios Prácticos (Jupyter Notebooks)

Los laboratorios prácticos están preparados para ejecutarse con aceleración por GPU (NVIDIA T4 o superior):

### Opción A: Google Colab (Recomendada)
1. Subir el notebook deseado a Google Drive o abrirlo directamente desde GitHub.
2. Ir a **Entorno de ejecución** > **Cambiar tipo de entorno de ejecución** y seleccionar **GPU T4**.
3. Ejecutar las celdas secuencialmente; las dependencias (`llama-cpp-python`, `sentence-transformers`, `chromadb`, etc.) se instalan automáticamente.

### Opción B: Entorno Local
Para ejecutar los laboratorios localmente con GPU NVIDIA y soporte CUDA:
```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar Jupyter y soporte
pip install jupyterlab nbdime

# Iniciar Jupyter Lab
jupyter lab laboratorios/
```

---

## 🛠️ Herramientas y Buenas Prácticas de Versionado

- **Merge y Diff de Notebooks:** El repositorio incluye `.gitattributes` configurado para utilizar `nbdime`. Se recomienda activarlo localmente con:
  ```bash
  pip install nbdime && nbdime config-git --enable
  ```
- **Modelos y Checkpoints:** Los archivos de pesos pesados (`.gguf`, `.safetensors`, `.bin`) y los reportes temporales de diagnóstico (`diagnostico_lab*.json`) están excluidos del control de versiones mediante `.gitignore`.
