#!/usr/bin/env python3
"""
==============================================================================
Clase 2: Prompt Engineering y Estructuración de Datos (UTN FRSR)
Script 1: Chain-of-Thought (CoT) - El Pizarrón como Memoria de Trabajo
Alineación: Filminas 12 y 13 (Caso Transversal: FinTech Horizon)
==============================================================================

Objetivos pedagógicos:
  1. Contrastar Inferencia Directa vs. Zero-Shot CoT vs. Few-Shot CoT.
  2. Demostrar empíricamente cómo los tokens intermedios permiten al modelo
     computar deducciones complejas antes de emitir la conclusión.
  3. Evaluar el trade-off en latencia y tokens de salida.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Carga de variables de entorno desde .env (en ./ o en ../)
load_dotenv()
load_dotenv("../.env")
load_dotenv("../../.env")

# Verificación pedagógica de credenciales
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY or "tu_api_key" in API_KEY:
    print("\n" + "=" * 70)
    print("⚠️  AVISO: No se detectó GEMINI_API_KEY configurada en el entorno.")
    print("Para ejecutar llamadas reales contra Google Gemini:")
    print("  1. Creá una clave gratuita en: https://aistudio.google.com/")
    print("  2. Copiá src/.env.example a .env y pegá tu clave:")
    print("     cp src/.env.example .env")
    print("  3. Volvé a ejecutar este script.")
    print("=" * 70 + "\n")
    sys.exit(1)

# Asegurar que LiteLLM reconozca la clave en la variable estándar
os.environ["GEMINI_API_KEY"] = API_KEY

import logging
import warnings
import litellm
from litellm import completion

# Silenciar advertencias y logs de depuración de LiteLLM para mantener la consola limpia
warnings.filterwarnings("ignore")
litellm.suppress_debug_info = True
litellm.verbose_logger.setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

litellm.suppress_debug_info = True
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# Configuración del modelo (configurable por entorno)
MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-3.8-flash")

# ==============================================================================
# CASO DE PRUEBA: Liquidación Financiera con Reglas Encadenadas (FinTech Horizon)
# ==============================================================================
PROBLEMA_FINANCIERO = """
Un cliente corporativo de FinTech Horizon realizó una transferencia internacional
por un monto de $12.500 USD con las siguientes condiciones operativas:
1. Comisión base de la plataforma: 2.0% sobre el monto total transferido.
2. Bonificación por segmento: Por pertenecer al nivel "Cuenta Corporativa Gold",
   recibe un descuento del 25% APLICADO EXCLUSIVAMENTE sobre la comisión base.
3. Tasa de procesamiento bancario SWIFT: Cargo fijo de $45 USD.
4. Retención impositiva fiscal: Se retiene un 10% sobre la sumatoria total
   de las comisiones y cargos generados (es decir, sobre la comisión neta + cargo SWIFT).

Pregunta: ¿Cuál es el costo total EXACTO que pagará el cliente por la operación
(comisión neta + cargo SWIFT + retención impositiva) y cuál es el desglose numérico?
"""

# ==============================================================================
# MODO 1: Inferencia Directa (Naive / Sin CoT) - Filmina 12
# ==============================================================================
def demo_inferencia_directa(problema: str):
    """
    Obliga al modelo a responder sin generar tokens de razonamiento intermedio.
    Demuestra la vulnerabilidad a fallas de cálculo aritmético y deducción.
    """
    print("\n" + "─" * 70)
    print("🔹 MODO 1: INFERENCIA DIRECTA (Sin CoT - Intento en un solo paso)")
    print("─" * 70)
    
    prompt = f"""{problema}

INSTRUCCIÓN ESTRICTA:
Responde de inmediato con el valor numérico final del costo total.
NO pienses en voz alta, NO muestres pasos ni fórmulas, responde únicamente
con la cifra final calculada."""

    inicio = time.time()
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        num_retries=3,
    )
    duracion = time.time() - inicio

    resultado = response.choices[0].message.content.strip()
    tokens = response.usage.completion_tokens if hasattr(response, "usage") else "N/D"

    print(f"⏱️  Latencia: {duracion:.2f}s | 🪙 Tokens de salida: {tokens}")
    print(f"📤 Respuesta del modelo:\n{resultado}")


# ==============================================================================
# MODO 2: Zero-Shot Chain-of-Thought (Kojima et al., 2022) - Filmina 13
# ==============================================================================
def demo_zero_shot_cot(problema: str):
    """
    Añade una directiva de activación cognitiva ("Pensemos paso a paso").
    El modelo usa sus propios tokens de salida como 'pizarrón' de cálculo.
    """
    print("\n" + "─" * 70)
    print("🔹 MODO 2: ZERO-SHOT CoT ('Pensemos paso a paso' - Kojima et al.)")
    print("─" * 70)

    prompt = f"""{problema}

INSTRUCCIÓN:
Pensemos paso a paso con máxima rigurosidad técnica.
Desglosa cada una de las 4 operaciones en orden secuencial explícito.
Muestra el cálculo intermedio de cada paso y finaliza indicando:
'COSTO_TOTAL_FINAL: $X USD'."""

    inicio = time.time()
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        num_retries=3,
    )
    duracion = time.time() - inicio

    resultado = response.choices[0].message.content.strip()
    tokens = response.usage.completion_tokens if hasattr(response, "usage") else "N/D"

    print(f"⏱️  Latencia: {duracion:.2f}s | 🪙 Tokens de salida: {tokens}")
    print(f"📤 Respuesta del modelo:\n{resultado}")


# ==============================================================================
# MODO 3: Few-Shot Chain-of-Thought (Wei et al., 2022) - Filmina 13
# ==============================================================================
def demo_few_shot_cot(problema: str):
    """
    Provee al modelo un ejemplo resuelto que no solo contiene la respuesta final,
    sino la estructura deductiva paso a paso deseada.
    """
    print("\n" + "─" * 70)
    print("🔹 MODO 3: FEW-SHOT CoT (Razonamiento Guiado con Ejemplos - Wei et al.)")
    print("─" * 70)

    # Ejemplo previo que entrena en contexto el estilo de razonamiento
    ejemplo_previo = """
Ejemplo de referencia resuelta:
Problema: Transferencia de $5.000 USD. Comisión base 3%. Descuento corporativo 20% sobre comisión base. SWIFT $30. Impuesto 10% sobre comisiones y SWIFT.
Cadena de Razonamiento Guiado:
- Paso 1 (Comisión base): $5.000 * 0.03 = $150.00 USD.
- Paso 2 (Descuento corporativo): 20% de $150 = $30.00 USD. Comisión neta = $150 - $30 = $120.00 USD.
- Paso 3 (Cargo fijo SWIFT): $30.00 USD. Subtotal = $120 + $30 = $150.00 USD.
- Paso 4 (Retención fiscal): 10% sobre $150.00 = $15.00 USD.
- Veredicto: COSTO_TOTAL_FINAL: $165.00 USD.
"""

    prompt = f"""Actúa como el Liquidador Contable Automatizado de FinTech Horizon.
Aplica con exactitud la misma metodología y estructura deductiva del siguiente ejemplo:

{ejemplo_previo}

Ahora resuelve este caso:
{problema}

Cadena de Razonamiento Guiado:"""

    inicio = time.time()
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        num_retries=3,
    )
    duracion = time.time() - inicio

    resultado = response.choices[0].message.content.strip()
    tokens = response.usage.completion_tokens if hasattr(response, "usage") else "N/D"

    print(f"⏱️  Latencia: {duracion:.2f}s | 🪙 Tokens de salida: {tokens}")
    print(f"📤 Respuesta del modelo:\n{resultado}")


# ==============================================================================
# EJECUCIÓN PRINCIPAL DE LA DEMOSTRACIÓN EN CLASE
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  UTN FRSR - ELECTIVA LLMs Y MODELOS GENERATIVOS")
    print(f"  DEMOSTRACIÓN EN VIVO: CHAIN-OF-THOUGHT (Modelo: {MODEL})")
    print("█" * 70)
    print("\nProblema planteado:")
    print(PROBLEMA_FINANCIERO.strip())

    try:
        # Ejecución comparativa de las 3 estrategias
        demo_inferencia_directa(PROBLEMA_FINANCIERO)
        demo_zero_shot_cot(PROBLEMA_FINANCIERO)
        demo_few_shot_cot(PROBLEMA_FINANCIERO)

        print("\n" + "=" * 70)
        print("💡 CONCLUSIÓN PEDAGÓGICA (Filminas 12 y 13):")
        print("  - Inferencia directa: Menor latencia/costo, pero alto riesgo de alucinación.")
        print("  - Zero-Shot CoT: Desbloquea la computación secuencial ('pizarrón').")
        print("  - Few-Shot CoT: Otorga control absoluto sobre el estándar de reporte.")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Error durante la invocación: {e}")
        sys.exit(1)
