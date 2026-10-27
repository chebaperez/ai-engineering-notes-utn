#!/usr/bin/env python3
"""
==============================================================================
Clase 2: Prompt Engineering y Estructuración de Datos (UTN FRSR)
Script 3: Salidas Estructuradas - Los Tres Niveles de Control
Alineación: Filmina 15 (Caso Transversal: FinTech Horizon)
==============================================================================

Objetivos pedagógicos:
  1. Experimentar los 3 niveles de control presentados en clase:
     - Nivel 1 (Frágil): Instrucción por Prompting ("devolveme solo un JSON").
     - Nivel 2 (Robusto): Modo JSON Nativo (response_format={'type': 'json_object'}).
     - Nivel 3 (Máximo): Validación y Esquemas Fuertemente Tipados con Pydantic.
  2. Comprender por qué el software empresarial exige contratos rígidos
     frente al lenguaje natural libre del LLM.
"""

import os
import sys
import json
import time
from enum import Enum
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

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
# CASO: Transcripción de Soporte Técnico en FinTech Horizon (Filmina 15)
# ==============================================================================
TRANSCRIPCION_SOPORTE = """
Operador: FinTech Horizon, le atiende Martín, ¿en qué puedo ayudarle?
Cliente: Hola Martín, mirá, estoy desesperado. Soy Carlos Gómez, DNI 28.441.902,
titular de la cuenta 4002-9918. Recién a las 14:15 quise sacar plata del cajero de la
sucursal Centro y la máquina se trabó, no me tiró la plata pero me descontó $30.000 ARS.
Necesito la plata urgente para comprar medicamentos de mi hijo.
Operador: Entiendo Carlos, voy a abrir el ticket con urgencia para Tesorería.
"""

# ==============================================================================
# NIVEL 1: Instrucción por Prompting (Frágil) - Filmina 15
# ==============================================================================
def demo_nivel_1_prompting(texto: str):
    """
    Se confía ciegamente en que el modelo respete la directiva de texto.
    Riesgo: Antepone saludos, introduce backticks markdown (```json),
    o agrega comentarios finales, rompiendo `json.loads()`.
    """
    print("\n" + "─" * 70)
    print("🔹 NIVEL 1: INSTRUCCIÓN POR PROMPTING (Frágil - Sin garantías del motor)")
    print("─" * 70)

    prompt = f"""
A partir de la siguiente conversación de soporte, extrae los datos del incidente.
INSTRUCCIÓN: Devuelve ÚNICAMENTE un objeto JSON con las claves:
cliente, dni, cuenta, monto_debitado, motivo, prioridad.
No incluyas explicaciones, ni texto introductorio, solo el JSON.

CONVERSACIÓN:
{texto}
"""
    t0 = time.time()
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        num_retries=3
    )
    salida_cruda = response.choices[0].message.content
    print(f"⏱️ Tiempo: {time.time()-t0:.2f}s")
    print(f"📤 Salida cruda emitida por el modelo:\n{salida_cruda}")

    # Intentamos parsear como lo haría un backend tradicional en producción
    print("\n🔎 Intentando parsear con json.loads() estándar en Python...")
    try:
        data = json.loads(salida_cruda)
        print("✅ Parseo exitoso (esta vez el modelo cooperó estadísticamente).")
        print(f"   Datos parseados: {data}")
    except json.JSONDecodeError as err:
        print(f"💥 ERROR DE PARSEO CRÍTICO: {err}")
        print("   👉 El parser del backend explotó porque el modelo agregó texto o markdown.")


# ==============================================================================
# NIVEL 2: Modo JSON Nativo (response_format={'type': 'json_object'})
# ==============================================================================
def demo_nivel_2_modo_json(texto: str):
    """
    El motor del proveedor garantiza que los tokens emitidos cierren sintácticamente
    un objeto JSON válido.
    Limitación: No garantiza los nombres de claves ni los tipos de datos internos.
    """
    print("\n" + "─" * 70)
    print("🔹 NIVEL 2: MODO JSON NATIVO (Sintaxis válida garantizada por el API)")
    print("─" * 70)

    prompt = f"""
Analiza la siguiente conversación y extrae un JSON con:
- cliente (string)
- dni (string)
- cuenta (string)
- monto_debitado (numero)
- prioridad (ALTA, MEDIA o BAJA)

CONVERSACIÓN:
{texto}
"""
    t0 = time.time()
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        num_retries=3
    )
    salida_json = response.choices[0].message.content
    print(f"⏱️ Tiempo: {time.time()-t0:.2f}s")
    print(f"📤 JSON válido garantizado emitido:\n{salida_json}")

    data = json.loads(salida_json)
    print("✅ json.loads() funciona siempre sin romperse.")
    print("⚠️  Advertencia pedagógica: ¿Garantiza que 'monto_debitado' sea float y no string?")
    print(f"   Tipo detectado para monto: {type(data.get('monto_debitado')).__name__}")


# ==============================================================================
# NIVEL 3: Esquema Fuertemente Tipado con Pydantic (Contrato de Ingeniería)
# ==============================================================================
class PrioridadTicket(str, Enum):
    BAJA = "BAJA"
    MEDIA = "MEDIA"
    ALTA = "ALTA"
    CRITICA = "CRITICA"

class TicketIncidenteFintech(BaseModel):
    """Contrato formal de datos para el sistema de tickets de FinTech Horizon."""
    cliente_nombre: str = Field(description="Nombre y apellido del titular")
    dni: str = Field(description="Documento nacional de identidad del cliente")
    numero_cuenta: str = Field(description="Número de cuenta o tarjeta involucrada")
    monto_reclamado: float = Field(description="Monto numérico exacto en disputa")
    motivo_falla: str = Field(description="Descripción concisa de la falla operativa")
    prioridad: PrioridadTicket = Field(description="Nivel de severidad asignado")
    requiere_bloqueo_preventivo: bool = Field(description="True si hay sospecha de fraude o clonación")
    etiquetas: List[str] = Field(default_factory=list, description="Tags clasificatorios para triaje")

def demo_nivel_3_pydantic(texto: str):
    """
    Nivel Máximo: Obliga al modelo a conformar un esquema Pydantic estricto.
    Se obtiene un objeto Python fuertemente tipado con auto-completado,
    validación de tipos y listo para persistir en SQL/NoSQL.
    """
    print("\n" + "─" * 70)
    print("🔹 NIVEL 3: ESQUEMA ESTRICTO CON PYDANTIC (Contrato de Datos Garantizado)")
    print("─" * 70)

    prompt = f"""
Eres el motor de ingesta y triaje de incidentes de FinTech Horizon.
Procesa la siguiente conversación y extrae el ticket respetando estrictamente el esquema:

{texto}
"""
    t0 = time.time()
    try:
        # LiteLLM soporta pasar la clase Pydantic directamente en response_format
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=TicketIncidenteFintech,
            num_retries=3
        )
        contenido = response.choices[0].message.content
        ticket = TicketIncidenteFintech.model_validate_json(contenido)
    except Exception:
        # Fallback de compatibilidad: modo json_object + validación Pydantic
        response = completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"Devuelve un JSON que respete este esquema JSON Schema:\n{json.dumps(TicketIncidenteFintech.model_json_schema())}"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            num_retries=3
        )
        ticket = TicketIncidenteFintech.model_validate_json(response.choices[0].message.content)

    print(f"⏱️ Tiempo: {time.time()-t0:.2f}s")
    print("\n📦 OBJETO PYDANTIC OBTENIDO (Tipado Estricto):")
    print(f"   • Cliente:           {ticket.cliente_nombre}")
    print(f"   • DNI:               {ticket.dni}")
    print(f"   • Cuenta:            {ticket.numero_cuenta}")
    print(f"   • Monto reclamado:   ${ticket.monto_reclamado:,.2f} ARS (tipo: {type(ticket.monto_reclamado).__name__})")
    print(f"   • Prioridad:         {ticket.prioridad.value} (Enum: {ticket.prioridad})")
    print(f"   • Bloqueo requerido: {ticket.requiere_bloqueo_preventivo}")
    print(f"   • Etiquetas:         {ticket.etiquetas}")
    print(f"   • Falla:             {ticket.motivo_falla}")

    print("\n💾 Serialización limpia a JSON para base de datos/API externa:")
    print(ticket.model_dump_json(indent=2))


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  UTN FRSR - ELECTIVA LLMs Y MODELOS GENERATIVOS")
    print(f"  DEMOSTRACIÓN: ESTRUCTURACIÓN GARANTIZADA (Modelo: {MODEL})")
    print("█" * 70)
    print("\nTranscripción recibida del centro de atención:")
    print(TRANSCRIPCION_SOPORTE.strip())

    try:
        demo_nivel_1_prompting(TRANSCRIPCION_SOPORTE)
        demo_nivel_2_modo_json(TRANSCRIPCION_SOPORTE)
        demo_nivel_3_pydantic(TRANSCRIPCION_SOPORTE)

        print("\n" + "=" * 70)
        print("💡 CONCLUSIÓN PEDAGÓGICA (Filmina 15):")
        print("  - Nivel 1: Cero garantías. No usar en pipelines de misión crítica.")
        print("  - Nivel 2: Sintaxis válida asegurada por el motor de inferencia.")
        print("  - Nivel 3: Contrato tipado garantizado por Pydantic.")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Error durante la invocación: {e}")
        sys.exit(1)
