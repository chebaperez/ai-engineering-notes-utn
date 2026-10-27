#!/usr/bin/env python3
"""
==============================================================================
Clase 2: Prompt Engineering y Estructuración de Datos (UTN FRSR)
Script 2: Descomposición de Tareas (Chaining) y Meta-Prompting
Alineación: Filmina 14 (Principio 5: Dividir el Trabajo)
==============================================================================

Objetivos pedagógicos:
  1. Demostrar el encadenamiento de llamadas (Pipelines): en lugar de pedirle
     todo a un único prompt gigante, se dividen las responsabilidades.
  2. Implementar Meta-Prompting: usar el LLM para diseñar y perfeccionar
     prompts de producción aplicando los principios de la materia.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
load_dotenv("../.env")
load_dotenv("../../.env")

# Verificación pedagógica de credenciales (soporta Google Gemini y OpenAI)
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

tiene_gemini = bool(GEMINI_KEY and "tu_api_key" not in GEMINI_KEY)
tiene_openai = bool(OPENAI_KEY and "tu_api_key" not in OPENAI_KEY)

if not (tiene_gemini or tiene_openai):
    print("\n" + "=" * 70)
    print("⚠️  AVISO: No se detectó GEMINI_API_KEY ni OPENAI_API_KEY configurada.")
    print("Para ejecutar llamadas reales contra un proveedor de LLM:")
    print("  1. Obtené una clave en https://aistudio.google.com/ (Gemini)")
    print("     o en https://platform.openai.com/api-keys (OpenAI).")
    print("  2. Copiá src/.env.example a .env y pegá tu clave:")
    print("     cp src/.env.example .env")
    print("  3. Volvé a ejecutar este script.")
    print("=" * 70 + "\n")
    sys.exit(1)

if tiene_gemini:
    os.environ["GEMINI_API_KEY"] = GEMINI_KEY
if tiene_openai:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

import logging
import warnings
import litellm
from litellm import completion

# Silenciar advertencias y logs de depuración de LiteLLM para mantener la consola limpia
warnings.filterwarnings("ignore")
litellm.suppress_debug_info = True
litellm.verbose_logger.setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# Configuración del modelo (selecciona automáticamente según la API Key disponible)
default_provider_model = "openai/gpt-4o-mini" if (tiene_openai and not tiene_gemini) else "gemini/gemini-3.8-flash"
MODEL = os.getenv("DEFAULT_MODEL", default_provider_model)
if MODEL.startswith("gemini/") and not tiene_gemini and tiene_openai:
    MODEL = "openai/gpt-4o-mini"
elif (MODEL.startswith("openai/") or MODEL.startswith("gpt-")) and not tiene_openai and tiene_gemini:
    MODEL = "gemini/gemini-3.8-flash"

# ==============================================================================
# PARTE 1: CADENA DE TAREAS (TASK CHAINING) - PIPELINE DE 3 PASOS
# ==============================================================================
# Caso: Incidente reportado por cliente en FinTech Horizon (mezcla de quejas,
# ruido emocional y datos operativos críticos).
INCIDENTE_BRUTO = """
Hola, soy Andrea Beltrán, titular de la cuenta 8841-209. Escribo porque estoy furiosa.
Ayer 26 de Octubre a las 19:42 intenté hacer un pago con mi tarjeta virtual por $45.000 ARS
en el comercio 'Electro Hogar'. La app me tiró 'Error de Timeout 504' pero en mi extracto
aparecen DOS débitos idénticos de $45.000 ARS con un minuto de diferencia (19:42 y 19:43)
y el comercio me dice que a ellos no les impactó ninguna compra y no me entregaron el producto.
Exijo que me devuelvan los $90.000 ARS de inmediato o hago la denuncia en Defensa al Consumidor.
Mi DNI es 33.109.845.
"""

def demo_pipeline_encadenado(texto_incidente: str):
    """
    En lugar de pedirle a un solo prompt: 'Extrae, evalúa la política y redacta el email',
    dividimos la carga cognitiva en 3 llamadas especializadas.
    """
    print("\n" + "█" * 70)
    print("🔗 PARTE 1: PIPELINE ENCADENADO DE 3 PASOS (Filmina 14)")
    print("█" * 70)

    # --------------------------------------------------------------------------
    # PASO 1: Extracción Analítica (Filtra ruido, aisla variables duras)
    # --------------------------------------------------------------------------
    print("\n[Paso 1/3] Extracción de Parámetros Críticos (Aislando ruido emocional)...")
    prompt_paso_1 = f"""
Analiza la siguiente comunicación de un cliente y extrae ÚNICAMENTE los datos fácticos.
Formato de salida requerido (clave: valor):
- CLIENTE:
- DNI:
- CUENTA:
- MONTO_TOTAL_EN_DISPUTA:
- NUMERO_DE_CARGOS:
- COMERCIO:
- PROBLEMA_TECNICO:

TEXTO DEL INCIDENTE:
\"\"\"
{texto_incidente.strip()}
\"\"\"
"""
    t0 = time.time()
    resp_paso_1 = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_paso_1}],
        num_retries=3
    )
    datos_extraidos = resp_paso_1.choices[0].message.content.strip()
    print(f"⏱️ Tiempo: {time.time()-t0:.2f}s")
    print(f"📋 Ficha de Datos Extraídos:\n{datos_extraidos}")

    # --------------------------------------------------------------------------
    # PASO 2: Evaluación Normativa con CoT (Sobre el extracto limpio)
    # --------------------------------------------------------------------------
    print("\n[Paso 2/3] Evaluación de Políticas con Chain-of-Thought (Sobre el extracto)...")
    politicas_horizon = """
POLÍTICA DE REEMBOLSOS FINTECH HORIZON (Manual de Operaciones Art. 4):
- Regla 4.1: Si existen débitos duplicados con idéntico monto en < 5 minutos por error de timeout del switch,
  corresponde REEMBOLSO_INMEDIATO del segundo cargo.
- Regla 4.2: El primer cargo permanece en estado PENDIENTE_CONCILIACION con el adquirente por 48 horas.
- Regla 4.3: Si el usuario amenaza con acciones legales, se debe clasificar prioridad ALTA y notificar a Legales.
"""
    prompt_paso_2 = f"""
{politicas_horizon}

A partir de los siguientes datos extraídos, razona paso a paso aplicando las políticas de FinTech Horizon:
{datos_extraidos}

Estructura de tu razonamiento:
1. Análisis de duplicación (Regla 4.1).
2. Estado del cobro inicial (Regla 4.2).
3. Nivel de severidad y escalamiento (Regla 4.3).
4. RESOLUCIÓN FINAL (Monto a devolver inmediatamente y acción).
"""
    t0 = time.time()
    resp_paso_2 = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_paso_2}],
        num_retries=3
    )
    dictamen_evaluacion = resp_paso_2.choices[0].message.content.strip()
    print(f"⏱️ Tiempo: {time.time()-t0:.2f}s")
    print(f"⚖️ Dictamen Normativo Interno:\n{dictamen_evaluacion}")

    # --------------------------------------------------------------------------
    # PASO 3: Redacción Ejecutiva para la Persona Usuaria
    # --------------------------------------------------------------------------
    print("\n[Paso 3/3] Redacción de Respuesta Oficial al Cliente...")
    prompt_paso_3 = f"""
Actúa como Oficial de Atención al Cliente de FinTech Horizon.
A partir del siguiente dictamen técnico, redacta un email empático, claro y profesional
para el cliente Andrea Beltrán comunicándole la resolución oficial.
Evita jerga técnica innecesaria, confirma el monto acreditado hoy y los próximos pasos.

DICTAMEN TÉCNICO:
{dictamen_evaluacion}
"""
    t0 = time.time()
    resp_paso_3 = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_paso_3}],
        num_retries=3
    )
    email_cliente = resp_paso_3.choices[0].message.content.strip()
    print(f"⏱️ Tiempo: {time.time()-t0:.2f}s")
    print(f"✉️ Comunicación Final Emitida:\n{email_cliente}")


# ==============================================================================
# PARTE 2: META-PROMPTING (EL MODELO COMO INGENIERO DE PROMPTS)
# ==============================================================================
def demo_meta_prompting():
    """
    Usa el modelo para generar sistemáticamente un prompt robusto
    a partir de un requerimiento informal del usuario o equipo de producto.
    """
    print("\n" + "█" * 70)
    print("🔁 PARTE 2: META-PROMPTING (El LLM como Diseñador de Prompts)")
    print("█" * 70)

    requerimiento_informal = (
        "Necesito un bot interno para Slack que reciba capturas o descripciones de errores "
        "de bases de datos de los desarrolladores juniors y les devuelva la causa probable y "
        "el comando SQL para solucionarlo, pero advirtiéndoles si el comando es destructivo."
    )

    print(f"📝 Requerimiento en lenguaje informal:\n\"{requerimiento_informal}\"\n")
    print("⏳ Generando System Prompt optimizado con los 5 principios de la materia...")

    meta_prompt = f"""
Eres un Ingeniero Principal de Prompts de IA.
Tu tarea es convertir un requerimiento informal en un SYSTEM PROMPT de grado de producción,
utilizando las mejores prácticas de la ingeniería de prompts:
1. Rol claro y especializado (Persona).
2. Contexto operativo y objetivo del sistema.
3. Delimitadores explícitos de entrada.
4. Restricciones operativas (Guardrails) y manejo de situaciones críticas.
5. Formato de salida estandarizado y predecible.

REQUERIMIENTO:
\"\"\"{requerimiento_informal}\"\"\"

Genera el System Prompt completo dentro de un bloque markdown listo para copiar y usar:
"""

    t0 = time.time()
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": meta_prompt}],
        num_retries=3
    )
    prompt_generado = response.choices[0].message.content.strip()
    print(f"⏱️ Tiempo: {time.time()-t0:.2f}s")
    print(f"\n🚀 SYSTEM PROMPT GENERADO:\n{prompt_generado}")


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  UTN FRSR - ELECTIVA LLMs Y MODELOS GENERATIVOS")
    print(f"  DEMOSTRACIÓN: CHAINING Y META-PROMPTING (Modelo: {MODEL})")
    print("█" * 70)

    try:
        demo_pipeline_encadenado(INCIDENTE_BRUTO)
        demo_meta_prompting()
        print("\n" + "=" * 70)
        print("💡 CONCLUSIÓN PEDAGÓGICA (Filmina 14):")
        print("  - Chaining: Menor 'cognitive load' por llamada, trazabilidad por etapa.")
        print("  - Meta-Prompting: El LLM es el mejor redactor de sus propias directivas.")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Error durante la invocación: {e}")
        sys.exit(1)
