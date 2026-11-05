#!/usr/bin/env python3
"""
==============================================================================
Clase 3: Sistemas RAG y Recuperación de Conocimiento (UTN FRSR)
Script 3: Pipeline RAG Punta a Punta: Fundamentación, Citas y Corrective RAG
Alineación: Filminas 5, 8, 15, 16 y 17 (Circuito 2 Completo, Grounding y CRAG)
Caso Transversal: Vertex Horizon Seguros (El Incidente de la Circular Inexistente)
==============================================================================

Objetivos pedagógicos:
  1. Contrastar el régimen de examen: Libro Cerrado (Sin RAG / Memoria Paramétrica)
     frente a Libro Abierto (RAG con evidencia documental privada).
  2. Implementar RAG Fundamentado con directivas estrictas de Abstención y Citas:
     - Trazabilidad auditable mediante citas obligatorias [ARTICULO X] [CLAUSULA Y].
     - Obligación de abstenerse ("NO CONSTA EN LA DOCUMENTACIÓN") ante huecos informativos.
  3. Implementar Corrective RAG (CRAG) con filtro evaluador de similitud pre-generación:
     - Bloquear consultas fuera de dominio sin invocar al LLM generador (ahorro 100% tokens).
  4. Monitorear telemetría completa de producción:
     - Tokens de entrada (prompt), salida (completion) y embeddings.
     - Delay de recuperación vectorial (ms) vs. Latencia de inferencia del LLM (s).
"""

import os
import sys
import time
import math
import logging
import warnings
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# ==============================================================================
# ESTILOS Y COLORES ANSI PARA TERMINAL
# ==============================================================================
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

def pausa_clase(mensaje: str = "Presiona Enter para continuar..."):
    """Pausa interactiva visible para permitir la explicación docente."""
    print(f"\n{C.BOLD}{C.YELLOW}⏸️  [Pausa de clase] {mensaje}{C.RESET}", end="")
    input()
    print()

def imprimir_tarjeta_respuesta(
    titulo: str, 
    contenido: str, 
    color_borde: str = C.GREEN, 
    color_texto: str = C.WHITE
):
    """Muestra la respuesta del modelo en un recuadro limpio con márgenes."""
    ancho = 68
    encabezado = f"┌── 📤 [{titulo}] "
    linea_top = encabezado + "─" * max(2, ancho - len(encabezado))
    print(f"{color_borde}{linea_top}{C.RESET}")
    for linea in contenido.split("\n"):
        print(f"{color_borde}│ {color_texto}{linea}{C.RESET}")
    print(f"{color_borde}└──{'─' * (ancho - 3)}{C.RESET}\n")


# ==============================================================================
# CARGA DE VARIABLES DE ENTORNO Y CREDENCIALES
# ==============================================================================
load_dotenv()
load_dotenv("../.env")
load_dotenv("../../.env")

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

tiene_gemini = bool(GEMINI_KEY and "tu_api_key" not in GEMINI_KEY)
tiene_openai = bool(OPENAI_KEY and "tu_api_key" not in OPENAI_KEY)

if not (tiene_gemini or tiene_openai):
    print("\n" + "=" * 70)
    print("⚠️  AVISO: No se detectó GEMINI_API_KEY ni OPENAI_API_KEY configurada.")
    print("Copiá src/.env.example a src/.env y pegá tu clave para continuar.")
    print("=" * 70 + "\n")
    sys.exit(1)

if tiene_gemini:
    os.environ["GEMINI_API_KEY"] = GEMINI_KEY
if tiene_openai:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

import litellm
from litellm import completion, embedding

warnings.filterwarnings("ignore")
litellm.suppress_debug_info = True
litellm.verbose_logger.setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# ==============================================================================
# CONFIGURACIÓN DE MODELOS
# Por defecto se utiliza gemini/gemini-3.5-flash-lite por ofrecer la cuota más amplia
# y gemini/gemini-embedding-001 para la generación de vectores de texto.
# ==============================================================================
default_gen_model = "openai/gpt-4o-mini" if (tiene_openai and not tiene_gemini) else "gemini/gemini-3.5-flash-lite"
MODELO_GENERADOR = os.getenv("DEFAULT_MODEL", default_gen_model)

default_emb_model = "openai/text-embedding-3-small" if (tiene_openai and not tiene_gemini) else "gemini/gemini-embedding-001"
MODELO_EMBEDDING = os.getenv("DEFAULT_EMBEDDING_MODEL", default_emb_model)


# ==============================================================================
# BASE DOCUMENTAL OFICIAL: VERTEX HORIZON SEGUROS S.A. (MANUAL 2024)
# ==============================================================================
DOCUMENTOS_BASE = [
    {
        "id": "ART-1",
        "codigo": "CLUST-90",
        "texto": (
            "ARTICULO 1: COBERTURA DE DANOS CLIMATICOS EN VIVIENDA URBANA.\n"
            "La poliza Hogar Total 2024 comprende la cobertura integral frente a contingencias "
            "meteorologicas como granizo, tempestades severas y anegamiento ordinario en cascos urbanos. "
            "La denuncia se canaliza mediante el formulario digital F-22. Verificada la cobertura, "
            "la compania liquidara el 100% del peritaje con franquicia deducible de 150 USD por evento.\n"
            "EXCLUSION EXPRESA CLUST-90: Quedan excluidos de toda indemnizacion los siniestros derivados "
            "de inundaciones extraordinarias o desbordes de cauces hidricos en predios y zonas rurales "
            "que no cuenten con obras de mitigacion y diques homologados ante la Superintendencia de Seguros."
        )
    },
    {
        "id": "ART-2",
        "codigo": "RCT-19",
        "texto": (
            "ARTICULO 2: PROTOCOLO DE DENUNCIA DE SINIESTROS VEHICULARES Y PLAZOS.\n"
            "Todo siniestro vehicular debe ser denunciado dentro de un plazo perentorio improrrogable "
            "de setenta y dos horas habiles posteriores al hecho. El titular debe acompanar acta policial "
            "certificada y formulario mecanico de declaracion jurada. La omision del plazo faculta "
            "el rechazo administrativo por caducidad del derecho conforme clausula operativa RCT-19."
        )
    },
    {
        "id": "ART-5",
        "codigo": "EXC-114",
        "texto": (
            "ARTICULO 5: COBERTURA DE ROBO TOTAL Y HURTO DE UNIDADES VEHICULARES.\n"
            "En caso de desapoderamiento total de la unidad, la aseguradora indemnizara el valor de reposicion "
            "de mercado con un 25% de depreciacion anual acumulada. No se estipula franquicia deducible para "
            "el evento de robo vehicular total en polizas estandar.\n"
            "EXCLUSION EXC-114: Se deniega el reclamo si el dispositivo de rastreo satelital GPS "
            "se encontraba inactivo o sin bateria al momento del hecho."
        )
    },
    {
        "id": "ART-10",
        "codigo": "REP-04",
        "texto": (
            "ARTICULO 10: COBERTURA ESPECIFICA DE CRISTALES Y GRANIZO VEHICULAR.\n"
            "La rotura de parabrisas, lunetas laterales y abolladuras por impacto de granizo sobre la "
            "carroceria se liquidan con franquicia reducida deducible de 90 USD por evento mediante "
            "la orden de reposicion REP-04, sin afectar la bonificacion por siniestralidad."
        )
    }
]


# ==============================================================================
# UTILIDADES VECTORIALES
# ==============================================================================
def similitud_coseno(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class IndiceVectorial:
    """Índice semántico en memoria para el pipeline de generación."""
    def __init__(self, documentos: List[Dict[str, str]], modelo_emb: str):
        self.documentos = documentos
        self.modelo = modelo_emb
        textos = [d["texto"] for d in documentos]
        
        t0 = time.time()
        res = embedding(model=self.modelo, input=textos)
        self.latencia_ingesta_ms = round((time.time() - t0) * 1000, 2)
        self.tokens_ingesta = res.usage.prompt_tokens if hasattr(res, "usage") and res.usage else 0
        self.vectores = [item["embedding"] for item in res.data]

    def recuperar(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        t0 = time.time()
        res_q = embedding(model=self.modelo, input=[query])
        t_embed_ms = round((time.time() - t0) * 1000, 2)
        tokens_q = res_q.usage.prompt_tokens if hasattr(res_q, "usage") and res_q.usage else 0
        vec_q = res_q.data[0]["embedding"]

        t_scan_0 = time.time()
        scores = []
        for i, doc in enumerate(self.documentos):
            sim = similitud_coseno(vec_q, self.vectores[i])
            scores.append({
                "id": doc["id"],
                "codigo": doc["codigo"],
                "score": round(sim, 4),
                "texto": doc["texto"]
            })
        scores.sort(key=lambda x: x["score"], reverse=True)
        t_scan_ms = round((time.time() - t_scan_0) * 1000, 3)

        return {
            "query": query,
            "tokens_query": tokens_q,
            "delay_recuperacion_vectorial_ms": t_scan_ms,
            "latencia_embedding_ms": t_embed_ms,
            "latencia_total_retrieval_ms": round(t_embed_ms + t_scan_ms, 2),
            "max_similitud": scores[0]["score"] if scores else 0.0,
            "chunks_recuperados": scores[:top_k]
        }


# ==============================================================================
# MODO 1: LIBRO CERRADO (SIN RAG / MEMORIA PARAMÉTRICA) - Filmina 5
# ==============================================================================
def demo_modo_libro_cerrado(pregunta: str) -> None:
    """
    Invoca al LLM sin proporcionarle contexto documental alguno.
    Demuestra la vulnerabilidad de 'Self-Delusion': inventa circulares o afirma coberturas inexistentes.
    """
    print("─" * 70)
    print(f"📕 {C.BOLD}{C.RED}MODO 1: EXAMEN A LIBRO CERRADO (Sin RAG / Memoria Paramétrica){C.RESET}")
    print("─" * 70)
    
    prompt = f"""Actúa como el Liquidador Automatizado de Vertex Horizon Seguros.
Responde a la siguiente consulta del asegurado con tono formal:

Pregunta: {pregunta}"""

    t0 = time.time()
    resp = completion(
        model=MODELO_GENERADOR,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        num_retries=2
    )
    duracion = time.time() - t0
    
    texto = resp.choices[0].message.content.strip()
    u = resp.usage
    tokens_in = u.prompt_tokens if u else 0
    tokens_out = u.completion_tokens if u else 0

    print(f"\n📊 {C.BOLD}Métricas:{C.RESET} ⏱️  Latencia LLM: {duracion:.2f}s │ 🪙 Tokens In: {tokens_in} │ 🪙 Tokens Out: {tokens_out}\n")
    imprimir_tarjeta_respuesta("Respuesta Generada a Libro Cerrado", texto, color_borde=C.RED, color_texto=C.WHITE)
    print(f"{C.BOLD}{C.RED}⚠️  DIAGNÓSTICO PEDAGÓGICO:{C.RESET} Al no poseer el manual interno en sus pesos,")
    print("el modelo es propenso a inventar circulares y condiciones con máxima elocuencia.\n")


# ==============================================================================
# MODO 2: RAG FUNDAMENTADO CON CITAS Y ABSTENCIÓN (GROUNDING) - Filmina 16
# ==============================================================================
def demo_modo_rag_fundamentado(pregunta: str, indice: IndiceVectorial) -> None:
    """
    Recupera la evidencia del manual y ensambla un prompt con contrato estricto de Grounding:
    - Obligación de citar explícitamente [ARTICULO X] o [CLAUSULA Y].
    - Prohibido inferir o inventar hechos.
    - Cláusula de abstención: si no figura taxativamente, responder 'NO CONSTA EN LA DOCUMENTACIÓN'.
    """
    print("─" * 70)
    print(f"📗 {C.BOLD}{C.GREEN}MODO 2: RAG FUNDAMENTADO CON CITAS OBLIGATORIAS Y ABSTENCIÓN (Grounding){C.RESET}")
    print("─" * 70)

    # 1. Recuperación vectorial
    res_rec = indice.recuperar(pregunta, top_k=2)
    chunks = res_rec["chunks_recuperados"]
    contexto_str = "\n\n---\n\n".join([f"[{c['id']}] {c['texto']}" for c in chunks])

    # 2. Ensamblado del prompt con directivas de libro abierto y grounding
    system_prompt = (
        "Eres el Asistente Técnico y Auditor de Siniestros de Vertex Horizon Seguros S.A.\n"
        "Tu función es responder consultas de pólizas basándote EXCLUSIVAMENTE en los fragmentos provistos.\n\n"
        "REGLAS ESTRICTAS DE CONTRATO FACTUAL:\n"
        "1. Cita taxativamente entre corchetes el [ARTICULO X] o [CODIGO] que respalda cada afirmación.\n"
        "2. Si la consulta involucra una excepción, exclusión o plazo, identifícala de forma inequívoca.\n"
        "3. Si un dato no figura explícitamente en el texto provisto (o no existe franquicia), "
        "debes manifestar obligatoriamente: 'NO CONSTA EN EL MANUAL DE PÓLIZAS DISPONIBLE'.\n"
        "4. Queda terminantemente prohibido inventar circulares, números de expediente o normativas no citadas."
    )

    user_prompt = f"""CONTEXTO DOCUMENTAL PROVISTO:
\"\"\"
{contexto_str}
\"\"\"

CONSULTA DEL ASEGURADO:
\"{pregunta}\"

RESPUESTA FUNDAMENTADA (incluye citas exactas):"""

    t0 = time.time()
    resp = completion(
        model=MODELO_GENERADOR,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        num_retries=2
    )
    duracion = time.time() - t0

    texto = resp.choices[0].message.content.strip()
    u = resp.usage
    tokens_in = u.prompt_tokens if u else 0
    tokens_out = u.completion_tokens if u else 0

    print(f"\n📊 {C.BOLD}TELEMETRÍA DE LA LLAMADA RAG:{C.RESET}")
    print(f"  • ⏱️  Delay recuperación vectorial (scan): {C.BOLD}{C.GREEN}{res_rec['delay_recuperacion_vectorial_ms']} ms{C.RESET}")
    print(f"  • ⏱️  Latencia generación LLM: {C.BOLD}{duracion:.2f} s{C.RESET}")
    print(f"  • ⏱️  Latencia total del pipeline: {C.BOLD}{round(res_rec['latencia_total_retrieval_ms']/1000 + duracion, 2)} s{C.RESET}")
    print(f"  • 🪙 Tokens Vector Store: {C.BOLD}{res_rec['tokens_query']} tokens{C.RESET}")
    print(f"  • 🪙 Tokens Prompt LLM (Contexto + Pregunta): {C.BOLD}{tokens_in} tokens{C.RESET}")
    print(f"  • 🪙 Tokens Completion LLM (Respuesta): {C.BOLD}{tokens_out} tokens{C.RESET}")
    print(f"  • 🎯 Similitud Coseno chunk líder: {C.BOLD}{C.CYAN}{res_rec['max_similitud']}{C.RESET}\n")

    imprimir_tarjeta_respuesta("Respuesta Fundamentada con Citas y Trazabilidad", texto, color_borde=C.GREEN, color_texto=C.WHITE)


# ==============================================================================
# MODO 3: CORRECTIVE RAG (CRAG) CON FILTRO EVALUADOR PREVIO - Filmina 15
# ==============================================================================
def demo_modo_corrective_rag(
    pregunta: str, 
    indice: IndiceVectorial, 
    umbral_similitud: float = 0.55
) -> None:
    """
    Implementa el patrón CRAG (Corrective RAG).
    Evalúa la relevancia semántica de los documentos recuperados antes de convocar al LLM.
    Si la similitud no supera el umbral (fuera de dominio), se aborta la generación,
    ahorrando el 100% de los tokens del generador y blindando el sistema.
    """
    print("─" * 70)
    print(f"🛡️  {C.BOLD}{C.CYAN}MODO 3: CORRECTIVE RAG (CRAG) - Filtro de Relevancia (Umbral >= {umbral_similitud}){C.RESET}")
    print("─" * 70)
    print(f"Pregunta analizada: {C.BOLD}{C.CYAN}\"{pregunta}\"{C.RESET}\n")

    # 1. Recuperación y Evaluación de Calidad del Contexto
    res_rec = indice.recuperar(pregunta, top_k=2)
    max_sim = res_rec["max_similitud"]
    chunk_lider = res_rec["chunks_recuperados"][0]

    print(f"🔍 {C.BOLD}Evaluación de Relevancia del Contexto:{C.RESET}")
    print(f"  • Mejor fragmento recuperado: {C.BOLD}{chunk_lider['id']}{C.RESET} (similitud = {C.BOLD}{max_sim}{C.RESET})")
    print(f"  • Umbral mínimo exigido: {C.BOLD}{umbral_similitud}{C.RESET}")

    # 2. Decisión del Filtro Revisor (CRAG Guardrail)
    if max_sim < umbral_similitud:
        print(f"\n{C.BOLD}{C.RED}⛔ FILTRO CRAG ACTIVADO: RELEVANCIA INSUFICIENTE (Consulta Fuera de Dominio).{C.RESET}")
        print(f"   La similitud ({max_sim}) está por debajo del umbral de corte ({umbral_similitud}).")
        print(f"   🛑 {C.BOLD}Acción:{C.RESET} Se interrumpe el pipeline y {C.BOLD}NO se invoca al modelo generador{C.RESET}.")
        print(f"   🪙 Tokens LLM ahorrados: {C.BOLD}{C.GREEN}100% (0 tokens consumidos){C.RESET}")
        print(f"   ⏱️  Delay de decisión: {C.BOLD}{res_rec['latencia_total_retrieval_ms']} ms{C.RESET}\n")
        
        mensaje_rechazo = (
            "Estimado asegurado: Su consulta sobre trámites bancarios o préstamos personales "
            "no corresponde a pólizas o coberturas disponibles en el manual operativo de Vertex Horizon Seguros."
        )
        imprimir_tarjeta_respuesta("Respuesta Segura de Rechazo Temprano (CRAG)", mensaje_rechazo, color_borde=C.RED, color_texto=C.YELLOW)
    else:
        print(f"\n{C.BOLD}{C.GREEN}✅ FILTRO CRAG APROBADO: Similitud suficiente ({max_sim} >= {umbral_similitud}).{C.RESET}")
        print("   ➡️  Derivando al modelo generador con evidencia validada...\n")
        demo_modo_rag_fundamentado(pregunta, indice)


# ==============================================================================
# CASOS DE PRUEBA Y EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  UTN FRSR - ELECTIVA LLMs Y MODELOS GENERATIVOS")
    print("  CLASE 3 · DEMO 3: PIPELINE RAG COMPLETO, GROUNDING Y CORRECTIVE RAG")
    print(f"  Modelo Generador: {MODELO_GENERADOR} (cuota amplia)")
    print(f"  Modelo Embeddings: {MODELO_EMBEDDING}")
    print("█" * 70)

    try:
        print(f"\n⚙️  Indexando manual de Vertex Horizon Seguros...")
        indice = IndiceVectorial(DOCUMENTOS_BASE, MODELO_EMBEDDING)
        print(f"✅ {C.BOLD}{C.GREEN}Índice inicializado: {len(DOCUMENTOS_BASE)} artículos clave cargados.{C.RESET}")

        pausa_clase("Presiona Enter para iniciar el Escenario 1 (Incidente Histórico de Vertex Horizon)...")

        # CASO 1: La consulta que causó el incidente millonario en Vertex Horizon
        PREGUNTA_INCIDENTE = (
            "¿La póliza Hogar Total 2024 de Vertex Horizon cubre daños por inundaciones "
            "extraordinarias en un predio rural según la circular del mes pasado?"
        )
        print("═" * 70)
        print(f"🔥 {C.BOLD}ESCENARIO 1: EL INCIDENTE HISTÓRICO DE VERTEX HORIZON{C.RESET}")
        print(f"Consulta: {C.BOLD}{C.CYAN}\"{PREGUNTA_INCIDENTE}\"{C.RESET}")
        print("═" * 70 + "\n")

        demo_modo_libro_cerrado(PREGUNTA_INCIDENTE)
        pausa_clase("Presiona Enter para ver la resolución del Incidente con RAG Fundamentado...")
        demo_modo_rag_fundamentado(PREGUNTA_INCIDENTE, indice)

        pausa_clase("Presiona Enter para avanzar al Escenario 2 (Pregunta tramposa sin respuesta en manual)...")

        # CASO 2: Pregunta capciosa sobre dato no existente en el manual (Franquicia de robo)
        PREGUNTA_TRAMPOSA = "¿Cuál es el deducible o franquicia fijada ante el robo total de una unidad vehicular?"
        print("═" * 70)
        print(f"🎯 {C.BOLD}ESCENARIO 2: PREGUNTA CAPCIOSA (Dato inexistente en el manual){C.RESET}")
        print(f"Consulta: {C.BOLD}{C.CYAN}\"{PREGUNTA_TRAMPOSA}\"{C.RESET}")
        print("═" * 70)
        print(f"💡 {C.YELLOW}Foco teórico:{C.RESET} En el manual hay franquicia para clima (150 USD) y cristales (90 USD), pero")
        print("para robo el ARTICULO 5 no estipula franquicia. El RAG ingenuo suele alucinar un valor.\n")
        
        demo_modo_rag_fundamentado(PREGUNTA_TRAMPOSA, indice)

        pausa_clase("Presiona Enter para avanzar al Escenario 3 (Corrective RAG - Consulta Fuera de Dominio)...")

        # CASO 3: Consulta fuera de dominio (Filtro CRAG)
        PREGUNTA_FUERA_DOMINIO = "¿Cómo abro una cuenta corriente o solicito un préstamo personal a tasa cero?"
        print("═" * 70)
        print(f"🛡️  {C.BOLD}ESCENARIO 3: FILTRO CRAG CONTRA CONSULTAS FUERA DE DOMINIO{C.RESET}")
        print(f"Consulta: {C.BOLD}{C.CYAN}\"{PREGUNTA_FUERA_DOMINIO}\"{C.RESET}")
        print("═" * 70 + "\n")
        
        demo_modo_corrective_rag(PREGUNTA_FUERA_DOMINIO, indice, umbral_similitud=0.55)

        pausa_clase("Presiona Enter para ver las conclusiones pedagógicas de la unidad...")

        print("=" * 70)
        print("💡 CONCLUSIÓN PEDAGÓGICA (Filminas 5, 8, 15, 16 y 17):")
        print("  1. Libro Abierto (RAG) no enseña nuevos pesos: inyecta evidencia auditable.")
        print("  2. El Grounding exige citas rígidas [ARTICULO X] y abstención explícita.")
        print("  3. Corrective RAG (CRAG) actúa como guardián de costos y seguridad,")
        print("     bloqueando consultas fuera de contexto antes de consumir tokens de LLM.")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Error durante la invocación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
