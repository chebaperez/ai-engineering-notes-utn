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
└── class_2/                       # Scripts para la Unidad 2: Prompt Engineering
    ├── 01_chain_of_thought.py     # Filminas 12 y 13: Direct vs. Zero-Shot CoT vs. Few-Shot CoT
    ├── 02_chaining_and_metaprompting.py # Filmina 14: Pipelines de 3 pasos y Meta-Prompting
    └── 03_structured_outputs.py   # Filmina 15: Nivel 1 (Prompt) vs Nivel 2 (JSON) vs Nivel 3 (Pydantic)
```

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
uv run python class_2/01_chain_of_thought.py
uv run python class_2/02_chaining_and_metaprompting.py
uv run python class_2/03_structured_outputs.py
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

## 🛠️ Ideas para interactuar en vivo en clase
1. **En `01_chain_of_thought.py`**: Cambiá las reglas de comisión agregando un impuesto extra y observá cómo la inferencia directa se equivoca mientras CoT desglosa la cuenta correctamente.
2. **En `02_chaining_and_metaprompting.py`**: Modificá el requerimiento informal en la Parte 2 para que el modelo redacte un prompt de detección de fraude en tiempo real.
3. **En `03_structured_outputs.py`**: Agregá un nuevo atributo al modelo Pydantic (ej. `score_riesgo: int = Field(ge=1, le=10)`) y ejecutá para ver cómo LiteLLM y Gemini respetan el nuevo contrato de datos.

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
