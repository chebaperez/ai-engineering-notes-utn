#!/usr/bin/env python3
"""
==============================================================================
Clase 3: Sistemas RAG y Recuperación de Conocimiento (UTN FRSR)
Script 2: Recuperación Semántica vs. Léxica y Búsqueda Híbrida (RRF)
Alineación: Filminas 11, 12, 13 y 14 (Densa vs. BM25 vs. Reciprocal Rank Fusion)
Caso Transversal: Vertex Horizon Seguros (Códigos Técnicos vs. Jerga Coloquial)
==============================================================================

Objetivos pedagógicos:
  1. Experimentar los puntos ciegos de los dos paradigmas de recuperación:
     - Búsqueda Densa (Embeddings): capta intenciones pero pierde códigos alfanuméricos exactos.
     - Búsqueda Léxica (BM25): rescata códigos y números exactos pero es sorda a la semántica.
  2. Implementar la Búsqueda Híbrida mediante Fusión Recíproca de Rangos (RRF):
     RRF_score(d) = Σ [ 1 / (k + rank(d)) ] con k=60.
  3. Ejecutar dos duelos empíricos controlados con telemetría en tiempo real:
     - Duelo A (Código técnico exacto): "cláusula RCT-19"
     - Duelo B (Paráfrasis coloquial): "se me quedó la chata tirada en la ruta"
"""

import os
import sys
import time
import math
import re
from collections import Counter
import logging
import warnings
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

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
from litellm import embedding

warnings.filterwarnings("ignore")
litellm.suppress_debug_info = True
litellm.verbose_logger.setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# Configuración de modelos
default_gen_model = "openai/gpt-4o-mini" if (tiene_openai and not tiene_gemini) else "gemini/gemini-3.5-flash-lite"
MODELO_GENERADOR = os.getenv("DEFAULT_MODEL", default_gen_model)

default_emb_model = "openai/text-embedding-3-small" if (tiene_openai and not tiene_gemini) else "gemini/gemini-embedding-001"
MODELO_EMBEDDING = os.getenv("DEFAULT_EMBEDDING_MODEL", default_emb_model)


# ==============================================================================
# CORPUS DE DOCUMENTOS: PÓLIZAS CALIBRADAS DE VERTEX HORIZON SEGUROS
# ==============================================================================
DOCUMENTOS_CORPUS = [
    {
        "id": "DOC-ART-1",
        "titulo": "Artículo 1: Daños climáticos urbanos y exclusión rural",
        "texto": (
            "ARTICULO 1: COBERTURA DE DANOS CLIMATICOS EN VIVIENDA URBANA. "
            "La poliza Hogar Total 2024 cubre tempestades, anegamiento y granizo en cascos urbanos. "
            "Denuncia obligatoria mediante formulario F-22. Franquicia deducible fija de 150 USD. "
            "EXCLUSION EXPRESA CLUST-90: Excluidas indemnizaciones por desbordes en predios rurales."
        )
    },
    {
        "id": "DOC-ART-2",
        "titulo": "Artículo 2: Plazos de denuncia vehicular y caducidad RCT-19",
        "texto": (
            "ARTICULO 2: PROTOCOLO DE DENUNCIA DE SINIESTROS VEHICULARES Y PLAZOS. "
            "Todo choque o destruccion vehicular debe declararse en 72 horas habiles con acta policial. "
            "La omision de este plazo perentorio provoca el rechazo por caducidad legal bajo la clausula RCT-19."
        )
    },
    {
        "id": "DOC-ART-5",
        "titulo": "Artículo 5: Robo vehicular y rastreo satelital",
        "texto": (
            "ARTICULO 5: COBERTURA DE ROBO TOTAL Y HURTO DE UNIDADES VEHICULARES. "
            "Indemnizacion a valor de mercado automotor con 25% de depreciacion anual acumulada. "
            "EXCLUSION EXC-114: Se rechaza la liquidacion de robo si el rastreador satelital GPS estaba inactivo."
        )
    },
    {
        "id": "DOC-ART-9",
        "titulo": "Artículo 9: Remolque y auxilio mecánico en ruta",
        "texto": (
            "ARTICULO 9: SERVICIO DE ASISTENCIA MECANICA Y REMOLQUE EN VIA PUBLICA. "
            "Dispone de hasta seis traslados anuales de grua por averias mecanicas, pinchaduras o fallas de motor, "
            "con un radio de remolque bonificado de hasta 120 kilometros desde el lugar del desperfecto."
        )
    },
    {
        "id": "DOC-ART-10",
        "titulo": "Artículo 10: Cristales y granizo en automóviles",
        "texto": (
            "ARTICULO 10: COBERTURA ESPECIFICA DE CRISTALES Y GRANIZO VEHICULAR. "
            "Rotura de parabrisas, lunetas y abolladuras por impacto de pedrada se liquidan con franquicia "
            "reducida de 90 USD por evento mediante la orden de reposicion REP-04."
        )
    }
]


# ==============================================================================
# MOTOR 1: BÚSQUEDA LÉXICA DISPERSA (BM25 - Best Matching 25)
# ==============================================================================
# Stopwords básicas en español para evitar ruido léxico en preposiciones y artículos
STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", 
    "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero", "sus", "le", 
    "ya", "o", "fue", "este", "ha", "si", "porque", "esta", "son", "entre", "esta", 
    "cuando", "muy", "sin", "sobre", "ser", "tiene", "tambien", "me", "hasta", "hay",
    "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni"
}

def tokenizar_texto(texto: str) -> List[str]:
    """Tokenizador normalizado: minúsculas, remoción de puntuación y filtrado de stopwords."""
    texto_limpio = re.sub(r"[^\w\s-]", " ", texto.lower())
    tokens = [t for t in texto_limpio.split() if len(t) > 1]
    return [t for t in tokens if t not in STOPWORDS_ES]


class MotorBM25:
    """
    Implementación didáctica de BM25 Okapi estándar.
    Utiliza parámetros canónicos k1=1.5 y b=0.75.
    """
    def __init__(self, documentos: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documentos = documentos
        self.N = len(documentos)
        
        # Longitudes y frecuencias
        self.doc_tokens = [tokenizar_texto(d["texto"]) for d in documentos]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / max(1, self.N)
        
        # Frecuencia en documentos (Document Frequency)
        self.df = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1
                
        # Inverse Document Frequency (IDF) probabilístico
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def buscar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Calcula el puntaje BM25 de la consulta contra cada documento."""
        tokens_q = tokenizar_texto(query)
        scores = []
        
        for i, doc in enumerate(self.documentos):
            score = 0.0
            doc_counts = Counter(self.doc_tokens[i])
            dl = self.doc_lens[i]
            
            for term in tokens_q:
                if term in self.idf:
                    tf = doc_counts[term]
                    idf_term = self.idf[term]
                    # Fórmula BM25
                    numerador = tf * (self.k1 + 1.0)
                    denominador = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                    score += idf_term * (numerador / denominador)
                    
            scores.append({
                "id": doc["id"],
                "titulo": doc["titulo"],
                "score_bm25": round(score, 4),
                "texto": doc["texto"]
            })
            
        scores.sort(key=lambda x: x["score_bm25"], reverse=True)
        return scores[:top_k]


# ==============================================================================
# MOTOR 2: BÚSQUEDA DENSA (SEMÁNTICA CON EMBEDDINGS)
# ==============================================================================
def similitud_coseno(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MotorDenso:
    """Indexador y buscador semántico basado en vectores densos."""
    def __init__(self, documentos: List[Dict[str, str]], modelo_embedding: str):
        self.documentos = documentos
        self.modelo = modelo_embedding
        
        textos = [d["texto"] for d in documentos]
        res = embedding(model=self.modelo, input=textos)
        self.vectores = [item["embedding"] for item in res.data]

    def buscar(self, query: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], int]:
        """Calcula la similitud coseno de la consulta contra cada documento."""
        res_q = embedding(model=self.modelo, input=[query])
        vec_q = res_q.data[0]["embedding"]
        tokens_q = res_q.usage.prompt_tokens if hasattr(res_q, "usage") and res_q.usage else 0
        
        scores = []
        for i, doc in enumerate(self.documentos):
            sim = similitud_coseno(vec_q, self.vectores[i])
            scores.append({
                "id": doc["id"],
                "titulo": doc["titulo"],
                "score_denso": round(sim, 4),
                "texto": doc["texto"]
            })
            
        scores.sort(key=lambda x: x["score_denso"], reverse=True)
        return scores[:top_k], tokens_q


# ==============================================================================
# MOTOR 3: BÚSQUEDA HÍBRIDA CON FUSIÓN RECÍPROCA DE RANGOS (RRF)
# ==============================================================================
def reciprocal_rank_fusion(
    ranking_denso: List[Dict[str, Any]], 
    ranking_bm25: List[Dict[str, Any]], 
    k: int = 60,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Aplica RRF (Reciprocal Rank Fusion) para combinar dos listas ordenadas:
    RRF_score(d) = Σ [ 1 / (k + rank(d)) ]
    donde rank(d) es la posición 1-indexada en cada motor.
    """
    rrf_map: Dict[str, Dict[str, Any]] = {}
    
    # 1. Procesar ranking denso
    for rank_idx, doc in enumerate(ranking_denso, 1):
        doc_id = doc["id"]
        rrf_map[doc_id] = {
            "id": doc_id,
            "titulo": doc["titulo"],
            "texto": doc["texto"],
            "rank_denso": rank_idx,
            "score_denso": doc["score_denso"],
            "rank_bm25": None,
            "score_bm25": 0.0,
            "rrf_score": 1.0 / (k + rank_idx)
        }
        
    # 2. Procesar ranking BM25 (solo para documentos con al menos una coincidencia léxica > 0)
    for rank_idx, doc in enumerate(ranking_bm25, 1):
        if doc.get("score_bm25", 0.0) <= 0.0:
            continue
            
        doc_id = doc["id"]
        aporte = 1.0 / (k + rank_idx)
        if doc_id in rrf_map:
            rrf_map[doc_id]["rank_bm25"] = rank_idx
            rrf_map[doc_id]["score_bm25"] = doc["score_bm25"]
            rrf_map[doc_id]["rrf_score"] += aporte
        else:
            rrf_map[doc_id] = {
                "id": doc_id,
                "titulo": doc["titulo"],
                "texto": doc["texto"],
                "rank_denso": None,
                "score_denso": 0.0,
                "rank_bm25": rank_idx,
                "score_bm25": doc["score_bm25"],
                "rrf_score": aporte
            }
            
    # Ordenar por puntaje RRF descendente
    lista_hibrida = list(rrf_map.values())
    lista_hibrida.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    for item in lista_hibrida:
        item["rrf_score"] = round(item["rrf_score"], 6)
        
    return lista_hibrida[:top_k]


# ==============================================================================
# EJECUCIÓN DE DUELOS COMPARATIVOS
# ==============================================================================
def ejecutar_duelo(
    nombre_duelo: str,
    consulta: str, 
    motor_bm25: MotorBM25, 
    motor_denso: MotorDenso, 
    explicacion: str
):
    print("\n" + "█" * 70)
    print(f"⚔️  {nombre_duelo}")
    print(f"   Consulta: \"{consulta}\"")
    print("█" * 70)
    print(f"💡 Foco teórico: {explicacion}\n")

    # 1. Búsqueda Léxica (BM25)
    t0_bm25 = time.time()
    res_bm25 = motor_bm25.buscar(consulta, top_k=3)
    t_bm25_ms = round((time.time() - t0_bm25) * 1000, 2)

    # 2. Búsqueda Densa (Embeddings)
    t0_denso = time.time()
    res_denso, tokens_q = motor_denso.buscar(consulta, top_k=3)
    t_denso_ms = round((time.time() - t0_denso) * 1000, 2)

    # 3. Búsqueda Híbrida (RRF)
    t0_rrf = time.time()
    res_hibrido = reciprocal_rank_fusion(res_denso, res_bm25, k=60, top_k=3)
    t_rrf_ms = round((time.time() - t0_rrf) * 1000, 3)

    # Mostrar Resultados Lado a Lado
    print(f"{'MOTOR':<12} | {'RANK #1':<10} | {'SCORE':<10} | {'LATENCIA':<10} | {'TÍTULO RESULTADO'}")
    print("-" * 75)
    
    top_bm = res_bm25[0]
    bm_desc = f"{top_bm['id']:<10} | {top_bm['score_bm25']:<10}" if top_bm['score_bm25'] > 0 else f"{'(Sin match)':<10} | {'0.0':<10}"
    bm_titulo = top_bm['titulo'] if top_bm['score_bm25'] > 0 else "Ningún documento superó el umbral léxico"
    print(f"{'BM25':<12} | {bm_desc} | {t_bm25_ms:<7} ms | {bm_titulo}")

    top_de = res_denso[0]
    print(f"{'Denso':<12} | {top_de['id']:<10} | {top_de['score_denso']:<10} | {t_denso_ms:<7} ms | {top_de['titulo']}")

    top_hi = res_hibrido[0]
    print(f"{'Híbrido RRF':<12} | {top_hi['id']:<10} | {top_hi['rrf_score']:<10} | {t_rrf_ms:<7} ms | {top_hi['titulo']}")

    print("\n📋 Desglose del Ganador Híbrido (RRF):")
    print(f"  • Documento: {top_hi['id']} - {top_hi['titulo']}")
    print(f"  • Rank Denso: {top_hi.get('rank_denso', 'Fuera de Top-K')} | Rank BM25: {top_hi.get('rank_bm25', 'Fuera de Top-K')}")
    print(f"  • Score RRF consolidado: {top_hi['rrf_score']}")
    print(f"  • Texto rescatado:\n    \"{top_hi['texto'][:180]}...\"")


# ==============================================================================
# EJECUCIÓN PRINCIPAL DE LA DEMOSTRACIÓN EN CLASE
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  UTN FRSR - ELECTIVA LLMs Y MODELOS GENERATIVOS")
    print("  CLASE 3 · DEMO 2: DUELO DE MOTORES (DENSO vs BM25 vs HÍBRIDO)")
    print(f"  Modelo de Embeddings: {MODELO_EMBEDDING}")
    print("█" * 70)

    try:
        print("\n⚙️  Indexando corpus de Vertex Horizon en ambos motores...")
        motor_bm25 = MotorBM25(DOCUMENTOS_CORPUS)
        motor_denso = MotorDenso(DOCUMENTOS_CORPUS, MODELO_EMBEDDING)
        print("✅ Corpus indexado exitosamente (5 documentos con reglas y distractores).")

        # DUELO 1: Código Técnico Exacto
        ejecutar_duelo(
            nombre_duelo="DUELO 1: CÓDIGO TÉCNICO EXACTO (Punto ciego de Embeddings)",
            consulta="¿Qué consecuencias legales contempla la cláusula RCT-19?",
            motor_bm25=motor_bm25,
            motor_denso=motor_denso,
            explicacion=(
                "Los códigos alfanuméricos sufren en el espacio denso porque sus submorfemas "
                "no tienen semántica intrínseca. BM25 encuentra el match léxico exacto al instante."
            )
        )

        input("\n[Pausa de clase] Presiona Enter para continuar con el Duelo 2...")

        # DUELO 2: Paráfrasis Coloquial
        ejecutar_duelo(
            nombre_duelo="DUELO 2: PARÁFRASIS COLOQUIAL (Punto ciego de BM25)",
            consulta="Se me quedó la chata tirada en la ruta de noche, ¿quién me viene a buscar?",
            motor_bm25=motor_bm25,
            motor_denso=motor_denso,
            explicacion=(
                "El asegurado usa lenguaje de la calle ('chata tirada en la ruta', 'quién me viene a buscar'). "
                "BM25 obtiene score 0 porque esas palabras no existen en la póliza. "
                "El motor Denso rescata el Artículo 9 (Asistencia Mecánica y Remolque) por pura semántica."
            )
        )

        print("\n" + "=" * 70)
        print("💡 CONCLUSIÓN PEDAGÓGICA (Filminas 11, 12, 13 y 14):")
        print("  - El motor Denso comprende el sentido profundo y las paráfrasis del usuario.")
        print("  - BM25 es insustituible para números de formulario, pólizas y cláusulas exactas.")
        print("  - La Búsqueda Híbrida con RRF elimina los puntos ciegos de ambos motores")
        print("    y es el estándar de producción en aseguradoras y sistemas legales.")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
