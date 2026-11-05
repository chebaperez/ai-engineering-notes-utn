#!/usr/bin/env python3
"""
==============================================================================
Clase 3: Sistemas RAG y Recuperación de Conocimiento (UTN FRSR)
Script 1: Ingesta, Segmentación (Chunking) y Almacenamiento Vectorial
Alineación: Filminas 6, 7, 9 y 10 (Circuito 1: Ingesta Offline y Espacio Semántico)
Caso Transversal: Vertex Horizon Seguros (Pólizas 2024 y Exclusiones Legales)
==============================================================================

Objetivos pedagógicos:
  1. Contrastar Chunking Fijo Naive vs. Chunking con Solapamiento (Overlap)
     y demostrar empíricamente el peligro de amputar excepciones legales (CLUST-90).
  2. Generar representaciones vectoriales densas (Embeddings) con LiteLLM
     usando la API de Google Gemini (gemini-embedding-001) o de OpenAI.
  3. Implementar un almacén vectorial en memoria pedagógico y transparente.
  4. Medir y reportar telemetría de producción:
     - Tokens de entrada consumidos (en ingesta y en consulta).
     - Dimensiones del espacio vectorial (ej. 3072 o 1536 dimensiones).
     - Delay de recuperación vectorial (latencia en milisegundos).
     - Score de similitud coseno de cada fragmento recuperado.
"""

import os
import sys
import time
import math
import logging
import warnings
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

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
    print("Para ejecutar llamadas reales contra un proveedor de Embeddings y LLM:")
    print("  1. Obtené una clave gratuita en https://aistudio.google.com/ (Gemini)")
    print("     o en https://platform.openai.com/api-keys (OpenAI).")
    print("  2. Copiá src/.env.example a src/.env y pegá tu clave:")
    print("     cp src/.env.example src/.env")
    print("  3. Volvé a ejecutar este script.")
    print("=" * 70 + "\n")
    sys.exit(1)

if tiene_gemini:
    os.environ["GEMINI_API_KEY"] = GEMINI_KEY
if tiene_openai:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

import litellm
from litellm import embedding

# Silenciar logs ruidosos de depuración
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
# BASE DOCUMENTAL DE PRUEBA: Extracto de Vertex Horizon Seguros 2024
# ==============================================================================
DOCUMENTO_POLIZAS = """MANUAL DE POLIZAS Y PROCEDIMIENTOS OPERATIVOS 2024 - VERTEX HORIZON SEGUROS S.A.

ARTICULO 1: COBERTURA DE DANOS CLIMATICOS EN VIVIENDA URBANA
La poliza Hogar Total 2024 comprende la cobertura integral frente a contingencias meteorologicas como granizo, tempestades severas y anegamiento ordinario en cascos urbanos. La denuncia del evento climatico se canaliza mediante el formulario digital F-22 disponible en la sucursal virtual. Verificada la cobertura, la compania liquidara el 100% del peritaje evaluado con una franquicia fija deducible de 150 USD por evento.
EXCLUSION EXPRESA CLUST-90: Quedan excluidos de toda indemnizacion los siniestros derivados de inundaciones extraordinarias o desbordes de cauces hidricos en predios y zonas rurales que no cuenten con obras de mitigacion y diques de contencion homologados ante la Superintendencia de Seguros en el informe ambiental previo.

ARTICULO 2: PROTOCOLO DE DENUNCIA DE SINIESTROS VEHICULARES Y PLAZOS
Todo siniestro vehicular, sean colisiones simples, destruccion parcial o incidentes en autopista, debe ser denunciado ante el centro de atencion al asegurado dentro de un plazo perentorio improrrogable de setenta y dos horas habiles posteriores a la ocurrencia del hecho. El titular esta obligado a acompanar acta policial certificada, copia del registro de conducir vigente del conductor involucrado y el formulario mecanico de declaracion jurada. La omision del plazo de 72 horas faculta el rechazo administrativo por caducidad del derecho, conforme la clausula operativa RCT-19.

ARTICULO 5: COBERTURA DE ROBO TOTAL Y HURTO DE UNIDADES VEHICULARES
En los supuestos de desapoderamiento total de la unidad asegurada, la compania indemnizara el valor de reposicion vigente en la tabla de referencia del mercado automotor. Sobre el valor de origen se aplica una depreciacion del 25% anual acumulada segun la antiguedad del rodado.
EXCLUSION EXC-114: Se deniega la cobertura de robo total cuando la unidad no contaba con el dispositivo de rastreo satelital homologado por la compania en estado operativo y activo al momento de la ocurrencia del hecho denunciado.
"""


# ==============================================================================
# FUNCIONALIDAD 1: ESTRATEGIAS DE SEGMENTACIÓN (CHUNKING)
# ==============================================================================
def chunk_texto_fijo(texto: str, chunk_size: int, overlap: int = 0) -> List[Dict[str, Any]]:
    """
    Divide un texto en bloques de longitud fija de caracteres sin respetar fronteras semánticas.
    Ilustra la estrategia ingenua y cómo rompe oraciones o cláusulas a la mitad.
    """
    chunks = []
    paso = max(1, chunk_size - overlap)
    idx = 0
    i = 0
    
    while i < len(texto):
        pedazo = texto[i : i + chunk_size].strip()
        if pedazo:
            chunks.append({
                "id": f"chunk_fijo_{idx}",
                "inicio": i,
                "fin": min(i + chunk_size, len(texto)),
                "texto": pedazo,
                "longitud": len(pedazo),
                "estrategia": f"Fijo ({chunk_size}c / overlap {overlap}c)"
            })
            idx += 1
        i += paso
        
    return chunks


def chunk_texto_recursivo(
    texto: str, 
    chunk_size: int = 400, 
    overlap: int = 100,
    separadores: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Segmentación jerárquica respetando fronteras de párrafos ("\\n\\n"),
    líneas ("\\n") o puntos (". "). Añade solapamiento (overlap) para garantizar
    que las excepciones y el contexto circundante no queden amputados.
    """
    if separadores is None:
        separadores = ["\n\n", "\n", ". ", " "]
        
    # Dividir primero por párrafos para mantener unidades conceptuales
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    chunks = []
    idx = 0
    
    buffer_actual = ""
    for parrafo in parrafos:
        # Si el párrafo cabe en el acumulador
        if len(buffer_actual) + len(parrafo) + 2 <= chunk_size:
            buffer_actual = f"{buffer_actual}\n\n{parrafo}".strip()
        else:
            if buffer_actual:
                chunks.append({
                    "id": f"chunk_rec_{idx}",
                    "texto": buffer_actual,
                    "longitud": len(buffer_actual),
                    "estrategia": f"Recursivo-Párrafo ({chunk_size}c / overlap {overlap}c)"
                })
                idx += 1
                # Preservar solapamiento: tomar los últimos 'overlap' caracteres
                cola_overlap = buffer_actual[-overlap:] if overlap > 0 else ""
                buffer_actual = f"{cola_overlap}\n{parrafo}".strip()
            else:
                # Si un solo párrafo excede el chunk_size, se fracciona limpiamente
                sub_chunks = chunk_texto_fijo(parrafo, chunk_size, overlap)
                for sc in sub_chunks:
                    chunks.append({
                        "id": f"chunk_rec_{idx}",
                        "texto": sc["texto"],
                        "longitud": len(sc["texto"]),
                        "estrategia": f"Recursivo-Fallback ({chunk_size}c / overlap {overlap}c)"
                    })
                    idx += 1
                buffer_actual = ""
                
    if buffer_actual:
        chunks.append({
            "id": f"chunk_rec_{idx}",
            "texto": buffer_actual,
            "longitud": len(buffer_actual),
            "estrategia": f"Recursivo-Párrafo ({chunk_size}c / overlap {overlap}c)"
        })
        
    return chunks


# ==============================================================================
# FUNCIONALIDAD 2: ÁLGEBRA VECTORIAL PURA Y EMBEDDINGS DENSOS
# ==============================================================================
def similitud_coseno(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores densos:
    cos(theta) = (A . B) / (||A|| * ||B||)
    Retorna un valor normalizado entre -1.0 y 1.0 (típicamente 0 a 1 en texto).
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Dimensión incompatible: {len(vec_a)} vs {len(vec_b)}")
        
    producto_punto = sum(a * b for a, b in zip(vec_a, vec_b))
    norma_a = math.sqrt(sum(a * a for a in vec_a))
    norma_b = math.sqrt(sum(b * b for b in vec_b))
    
    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0
        
    return producto_punto / (norma_a * norma_b)


def generar_embeddings_lote(textos: List[str], modelo: str) -> Tuple[List[List[float]], int]:
    """
    Invoca la API de Embeddings (Gemini u OpenAI) a través de LiteLLM.
    Retorna la lista de vectores densos y la cantidad de tokens de entrada consumidos.
    """
    t0 = time.time()
    respuesta = embedding(
        model=modelo,
        input=textos
    )
    latencia = time.time() - t0
    
    vectores = [item["embedding"] for item in respuesta.data]
    tokens = respuesta.usage.prompt_tokens if hasattr(respuesta, "usage") and respuesta.usage else 0
    
    return vectores, tokens


# ==============================================================================
# FUNCIONALIDAD 3: ALMACÉN VECTORIAL EN MEMORIA (IN-MEMORY VECTOR STORE)
# ==============================================================================
def crear_base_vectorial(chunks: List[Dict[str, Any]], modelo_embedding: str) -> Dict[str, Any]:
    """
    Indexa los fragmentos calculando sus vectores y persistiendo metadatos.
    Registra telemetría: tiempo de indexación y tokens consumidos.
    """
    print(f"\n⚙️  Indexando {len(chunks)} chunks con el modelo '{modelo_embedding}'...")
    t0 = time.time()
    
    textos = [c["texto"] for c in chunks]
    vectores, tokens_consumidos = generar_embeddings_lote(textos, modelo_embedding)
    duracion = time.time() - t0
    
    dimension = len(vectores[0]) if vectores else 0
    registros = []
    for chunk, vec in zip(chunks, vectores):
        registros.append({
            "id": chunk["id"],
            "texto": chunk["texto"],
            "longitud": chunk["longitud"],
            "vector": vec
        })
        
    base_db = {
        "modelo": modelo_embedding,
        "dimension": dimension,
        "total_registros": len(registros),
        "tokens_indexacion": tokens_consumidos,
        "latencia_indexacion_ms": round(duracion * 1000, 2),
        "registros": registros
    }
    
    print(f"✅ Base vectorial creada exitosamente:")
    print(f"   • Chunks almacenados: {len(registros)}")
    print(f"   • Dimensión del embedding: {dimension} dimensiones")
    print(f"   • Tokens de entrada consumidos: {tokens_consumidos}")
    print(f"   • Tiempo total de indexación: {base_db['latencia_indexacion_ms']} ms")
    
    return base_db


def buscar_en_base_vectorial(
    query: str, 
    base_db: Dict[str, Any], 
    top_k: int = 2
) -> Dict[str, Any]:
    """
    Realiza la búsqueda k-NN (k-Nearest Neighbors) por similitud coseno.
    Mide el delay exacto de recuperación (vector search latency) y tokens de query.
    """
    t0 = time.time()
    
    # 1. Vectorizar la consulta
    vectores_q, tokens_q = generar_embeddings_lote([query], base_db["modelo"])
    vec_query = vectores_q[0]
    
    t_embed_query = time.time() - t0
    
    # 2. Calcular distancias contra todos los vectores indexados (Scan exhaustivo)
    t_busqueda_inicio = time.time()
    resultados = []
    for reg in base_db["registros"]:
        score = similitud_coseno(vec_query, reg["vector"])
        resultados.append({
            "id": reg["id"],
            "score": round(score, 4),
            "texto": reg["texto"]
        })
        
    # 3. Ordenar por score descendente y seleccionar Top-K
    resultados.sort(key=lambda x: x["score"], reverse=True)
    seleccionados = resultados[:top_k]
    
    t_busqueda_fin = time.time()
    latencia_scan_ms = round((t_busqueda_fin - t_busqueda_inicio) * 1000, 3)
    latencia_total_ms = round((t_busqueda_fin - t0) * 1000, 2)
    
    return {
        "query": query,
        "tokens_query": tokens_q,
        "latencia_embedding_ms": round(t_embed_query * 1000, 2),
        "delay_recuperacion_vectorial_ms": latencia_scan_ms,
        "latencia_total_recuperacion_ms": latencia_total_ms,
        "top_k": seleccionados
    }


# ==============================================================================
# DEMOSTRACIÓN 1: EL PELIGRO DEL CHUNKING CIEGO (Filmina 10)
# ==============================================================================
def demo_experimento_chunking():
    """
    Muestra cómo un chunking rígido sin solapamiento parte una cláusula legal y
    separa la cobertura de su exclusión restrictiva.
    """
    print("\n" + "█" * 70)
    print("🔬 EXPERIMENTO 1: DÓNDE SE PIERDEN LAS EXCEPCIONES LEGALES (Filmina 10)")
    print("█" * 70)
    
    # Tomamos exclusivamente el Artículo 1 para aislar la fractura del texto
    art_1 = """ARTICULO 1: COBERTURA DE DANOS CLIMATICOS EN VIVIENDA URBANA
La poliza Hogar Total 2024 comprende la cobertura integral frente a contingencias meteorologicas como granizo, tempestades severas y anegamiento ordinario en cascos urbanos. La denuncia del evento climatico se canaliza mediante el formulario digital F-22 disponible en la sucursal virtual. Verificada la cobertura, la compania liquidara el 100% del peritaje evaluado con una franquicia fija deducible de 150 USD por evento.
EXCLUSION EXPRESA CLUST-90: Quedan excluidos de toda indemnizacion los siniestros derivados de inundaciones extraordinarias o desbordes de cauces hidricos en predios y zonas rurales que no cuenten con obras de mitigacion y diques de contencion homologados ante la Superintendencia de Seguros en el informe ambiental previo."""

    print("\n📄 Texto analizado: Artículo 1 (Regla General + Exclusión CLUST-90)")
    print(f"Longitud total: {len(art_1)} caracteres.")

    # 1. Estrategia Fija Naive (Tamaño 330 caracteres, sin solapamiento)
    chunks_fijos = chunk_texto_fijo(art_1, chunk_size=330, overlap=0)
    print("\n" + "─" * 70)
    print(f"❌ ESTRATEGIA A: CHUNKING FIJO RÍGIDO (330 caracteres / Solapamiento: 0)")
    print("─" * 70)
    for c in chunks_fijos:
        print(f"[{c['id']}] ({c['longitud']} chars):")
        print(f"  \"{c['texto']}\"")
        if "CLUST-90" in c["texto"] and "liquidara el 100%" not in c["texto"]:
            print("  ⚠️  ALERTA PEDAGÓGICA: La exclusión CLUST-90 quedó huérfana de su artículo de origen.")
        elif "liquidara el 100%" in c["texto"] and "EXCLUSION" not in c["texto"]:
            print("  ⚠️  ALERTA PEDAGÓGICA: El chunk afirma cobertura al 100% omitiendo la restricción.")

    # 2. Estrategia con Solapamiento Adecuado (Tamaño 380 caracteres, solapamiento 100)
    chunks_solapados = chunk_texto_recursivo(art_1, chunk_size=420, overlap=120)
    print("\n" + "─" * 70)
    print(f"✅ ESTRATEGIA B: CHUNKING CON SOLAPAMIENTO (420 caracteres / Solapamiento: 120)")
    print("─" * 70)
    for c in chunks_solapados:
        print(f"[{c['id']}] ({c['longitud']} chars):")
        print(f"  \"{c['texto']}\"")
        if "liquidara el 100%" in c["texto"] and "CLUST-90" in c["texto"]:
            print("  🎯 ÉXITO: El chunk preserva la regla general Y su excepción legal juntas.")


# ==============================================================================
# DEMOSTRACIÓN 2: BASE VECTORIAL Y TELEMETRÍA DE RECUPERACIÓN (Filminas 7 y 9)
# ==============================================================================
def demo_vector_store_y_telemetria():
    """
    Construye la base vectorial en memoria y efectúa una consulta de siniestro,
    mostrando las métricas cuantitativas clave de la teoría.
    """
    print("\n" + "█" * 70)
    print("🔍 EXPERIMENTO 2: DEL TEXTO AL ESPACIO VECTORIAL Y TELEMETRÍA DE BÚSQUEDA")
    print(f"   Modelo de Embeddings activo: {MODELO_EMBEDDING}")
    print("█" * 70)

    # Segmentamos el documento completo con solapamiento
    chunks = chunk_texto_recursivo(DOCUMENTO_POLIZAS, chunk_size=380, overlap=100)
    
    # Ingesta offline: Generación de embeddings e indexación
    db_vectorial = crear_base_vectorial(chunks, MODELO_EMBEDDING)
    
    # Consulta crítica: Caso real del incidente de Vertex Horizon
    consulta = "¿La póliza cubre daños por desborde o inundación en un campo rural?"
    
    print("\n" + "─" * 70)
    print(f"🔎 EJECUTANDO RECUPERACIÓN SEMÁNTICA (TOP-K = 2)")
    print(f"   Consulta formulada: \"{consulta}\"")
    print("─" * 70)
    
    resultado = buscar_en_base_vectorial(consulta, db_vectorial, top_k=2)
    
    # Reporte de Métricas y Telemetría
    print("\n📊 TELEMETRÍA DE RECUPERACIÓN:")
    print(f"  • 🪙 Tokens consumidos por la consulta: {resultado['tokens_query']} tokens")
    print(f"  • ⏱️  Latencia de vectorización de query: {resultado['latencia_embedding_ms']} ms")
    print(f"  • ⏱️  Delay de búsqueda vectorial (k-NN scan): {resultado['delay_recuperacion_vectorial_ms']} ms")
    print(f"  • ⏱️  Latencia total del pipeline de búsqueda: {resultado['latencia_total_recuperacion_ms']} ms")
    
    print("\n🎯 FRAGMENTOS (CHUNKS) RECUPERADOS:")
    for i, item in enumerate(resultado["top_k"], 1):
        print(f"\n  [Rank #{i}] Chunk ID: {item['id']} | Similitud Coseno: {item['score']}")
        print(f"  ──────────────────────────────────────────────────────────")
        print(f"  {item['texto']}")


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  UTN FRSR - ELECTIVA LLMs Y MODELOS GENERATIVOS")
    print("  CLASE 3 · DEMO 1: CHUNKING Y ALMACÉN VECTORIAL EN MEMORIA")
    print(f"  Modelo Embedding: {MODELO_EMBEDDING}")
    print(f"  Modelo Generador configurado: {MODELO_GENERADOR}")
    print("█" * 70)

    try:
        demo_experimento_chunking()
        input("\n[Pausa de clase] Presiona Enter para continuar con la indexación vectorial...")
        demo_vector_store_y_telemetria()
        
        print("\n" + "=" * 70)
        print("💡 CONCLUSIONES PEDAGÓGICAS (Filminas 6, 7, 9 y 10):")
        print("  1. El chunking ciego amputa cláusulas: si la exclusión legal queda en")
        print("     otro chunk, el buscador recuperará sólo la cobertura y el LLM indemnizará.")
        print("  2. El solapamiento (overlap) garantiza la continuidad semántica del texto.")
        print("  3. La base vectorial mide similitud matemática en alta dimensionalidad;")
        print("     el delay de búsqueda en memoria es mínimo (<1 ms), siendo la llamada")
        print("     a la API de embeddings el componente dominante de la latencia.")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

