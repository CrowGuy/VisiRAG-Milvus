"""
Milvus + ColQwen2 POC (Refactored)
- Clear module boundaries
- Config-driven (easy to tweak params)
- Cache-friendly (JSONL for page records; optional embedding cache hooks)
- Single responsibility functions
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from pdf2image import convert_from_path

from colpali_engine.models import ColQwen2, ColQwen2Processor
from pymilvus import MilvusClient, DataType
from pymilvus.client.embedding_list import EmbeddingList


# =========================
# Config
# =========================

@dataclass(frozen=True)
class PathsConfig:
    pdf_folder: str = "3d_nand_docs"
    image_output_dir: str = "3d_nand_images"
    page_records_jsonl: str = "3d_nand_page_records.jsonl"


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "/home/randy/Documents/vlm/models/colqwen2-v1.0"
    device: str = "cuda:0"
    torch_dtype: torch.dtype = torch.bfloat16

@dataclass(frozen=True)
class IngestConfig:
    image_batch_size: int = 8          # 4090 建議先 4~16 試
    insert_batch_size: int = 16        # milvus insert payload flush size
    num_workers: int = 0              # 先 0，之後可改成 DataLoader

@dataclass(frozen=True)
class MilvusConfig:
    uri: str = "http://10.0.201.133:19530"
    collection_name: str = "page_3d_nand_aos"
    # schema
    embedding_dim: int = 128
    patches_max_capacity: int = 2048


@dataclass(frozen=True)
class IndexConfig:
    index_type: str = "HNSW"         # AoS supports AUTOINDEX/HNSW per docs
    metric_type: str = "MAX_SIM_IP"  # if instability occurs, consider MAX_SIM_COSINE + normalize
    params: Dict[str, Any] = None

    def __post_init__(self):
        # dataclass frozen -> cannot set; provide default via object.__setattr__
        if self.params is None:
            object.__setattr__(self, "params", {"M": 32, "efConstruction": 200})


@dataclass(frozen=True)
class SearchConfig:
    metric_type: str = "MAX_SIM_IP"
    params: Dict[str, Any] = None
    limit: int = 5
    output_fields: Tuple[str, ...] = ("page_id", "doc_name", "page_number")

    def __post_init__(self):
        if self.params is None:
            object.__setattr__(self, "params", {"ef": 100, "retrieval_ann_ratio": 3})


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    ingest: IngestConfig = IngestConfig()
    milvus: MilvusConfig = MilvusConfig()
    index: IndexConfig = IndexConfig()
    search: SearchConfig = SearchConfig()


# =========================
# Utils: JSONL Cache
# =========================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# =========================
# Ingestion: PDF -> Images -> Page Records (cached)
# =========================

def build_page_records_from_pdfs(
    pdf_folder: str,
    image_output_dir: str,
) -> List[Dict[str, Any]]:
    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")])

    all_page_records: List[Dict[str, Any]] = []
    global_page_counter = 1

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"[ingest] Converting {pdf_file} ...")

        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_output_dir = os.path.join(image_output_dir, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        images = convert_from_path(pdf_path)

        for i, img in enumerate(images):
            page_number = i + 1
            page_id = global_page_counter
            global_page_counter += 1

            image_path = os.path.join(pdf_output_dir, f"page_{page_number}.jpg")
            img.save(image_path, "JPEG")

            all_page_records.append(
                {
                    "page_id": page_id,
                    "doc_name": pdf_file,
                    "page_number": page_number,
                    "image_path": image_path,
                }
            )

    return all_page_records


def load_or_create_page_records(
    pdf_folder: str,
    image_output_dir: str,
    jsonl_path: str,
) -> List[Dict[str, Any]]:
    if os.path.exists(jsonl_path):
        print(f"[cache] Loading page records from {jsonl_path}")
        return read_jsonl(jsonl_path)

    print("[cache] JSONL not found. Building from PDFs...")
    records = build_page_records_from_pdfs(pdf_folder, image_output_dir)
    write_jsonl(jsonl_path, records)
    print(f"[cache] Saved {len(records)} records -> {jsonl_path}")
    return records


# =========================
# Model: ColQwen2 encoder (images + queries)
# =========================

class ColQwen2Encoder:
    """
    Thin wrapper for model + processor.
    Handles:
      - device/dtype placement
      - CPU float32 conversion for Milvus payloads
      - producing patch/token multi-vector outputs
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = ColQwen2.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=cfg.torch_dtype,
            device_map=cfg.device,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(cfg.model_name_or_path)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = self.processor.process_images(list(images)).to(self.device)
        with torch.inference_mode():
            emb = self.model(**batch)
        return emb.detach().to("cpu")  # <<< 關鍵：emb 不要留在 GPU

    def encode_queries(self, queries: Sequence[str]) -> torch.Tensor:
        batch = self.processor.process_queries(list(queries)).to(self.device)
        with torch.inference_mode():
            emb = self.model(**batch)
        return emb

    @staticmethod
    def to_cpu_float32_list(vec: torch.Tensor) -> List[float]:
        return vec.detach().to("cpu", torch.float32).tolist()

    @staticmethod
    def to_cpu_float32_numpy(vec: torch.Tensor):
        return vec.detach().to("cpu", torch.float32).numpy()


# =========================
# Milvus: schema / index / upsert / search
# =========================

class MilvusDocStore:
    def __init__(self, cfg: MilvusConfig):
        self.cfg = cfg
        self.client = MilvusClient(uri=cfg.uri)

    def create_collection_if_needed(self) -> None:
        # MilvusClient has list_collections in newer versions; be defensive.
        try:
            existing = set(self.client.list_collections())
        except Exception:
            existing = set()

        if self.cfg.collection_name in existing:
            print(f"[milvus] Collection exists: {self.cfg.collection_name}")
            return

        schema = self.client.create_schema()
        schema.add_field("page_id", DataType.INT64, is_primary=True)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("doc_name", DataType.VARCHAR, max_length=500)
        schema.add_field("image_path", DataType.VARCHAR, max_length=500)

        struct_schema = self.client.create_struct_field_schema()
        struct_schema.add_field("patch_embedding", DataType.FLOAT_VECTOR, dim=self.cfg.embedding_dim)

        schema.add_field(
            "patches",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct_schema,
            max_capacity=self.cfg.patches_max_capacity,
        )

        self.client.create_collection(self.cfg.collection_name, schema=schema)
        print(f"[milvus] Created collection: {self.cfg.collection_name}")

    def insert_pages_with_patches_batch(
        self,
        records_batch: List[Dict[str, Any]],
        embeddings_batch_cpu: torch.Tensor,  # [B, num_patches, dim] on CPU
    ) -> None:
        payload = []
        # embeddings_batch_cpu[i] 對應 records_batch[i]
        for r, patches in zip(records_batch, embeddings_batch_cpu):
            r2 = dict(r)
            r2["patches"] = [{"patch_embedding": patches[j].to(torch.float32).tolist()}
                             for j in range(patches.shape[0])]
            payload.append(r2)

        self.client.insert(self.cfg.collection_name, payload)

    def build_index_and_load(self, index_cfg: IndexConfig) -> None:
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="patches[patch_embedding]",
            index_type=index_cfg.index_type,
            metric_type=index_cfg.metric_type,
            params=index_cfg.params,
        )
        self.client.create_index(self.cfg.collection_name, index_params)
        self.client.load_collection(self.cfg.collection_name)
        print(f"[milvus] Index built + collection loaded: {self.cfg.collection_name}")

    def search_pages(
        self,
        query_embeddings: torch.Tensor,
        search_cfg: SearchConfig,
    ):
        """
        query_embeddings:
          - [B, n_tokens, dim]  -> batch queries
          - [n_tokens, dim]     -> single query
        return:
          - results: list length B, each is hits list
        """
        if query_embeddings.dim() == 2:
            query_embeddings = query_embeddings.unsqueeze(0)

        data: List[EmbeddingList] = []
        for qb in query_embeddings:  # qb: [n_tokens, dim]
            emb_list = EmbeddingList()
            for v in qb:
                emb_list.add(ColQwen2Encoder.to_cpu_float32_numpy(v))
            data.append(emb_list)

        results = self.client.search(
            collection_name=self.cfg.collection_name,
            data=data,
            anns_field="patches[patch_embedding]",
            search_params={
                "metric_type": search_cfg.metric_type,
                "params": search_cfg.params,
            },
            limit=search_cfg.limit,
            output_fields=list(search_cfg.output_fields),
        )
        return results


# =========================
# Orchestration: offline build + retrieval
# =========================

def load_images(records: List[Dict[str, Any]]) -> List[Image.Image]:
    # Decoupled so you can later stream/batch/load lazily
    return [Image.open(r["image_path"]).convert("RGB") for r in records]

def iter_image_batches(records: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(records), batch_size):
        batch_records = records[i:i+batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch_records]
        yield batch_records, images
        # 立刻關檔案 handle（PIL 有時不會馬上釋放）
        for im in images:
            try:
                im.close()
            except Exception:
                pass


def offline_build(cfg: AppConfig) -> None:
    records = load_or_create_page_records(
        cfg.paths.pdf_folder,
        cfg.paths.image_output_dir,
        cfg.paths.page_records_jsonl,
    )

    encoder = ColQwen2Encoder(cfg.model)
    store = MilvusDocStore(cfg.milvus)
    store.create_collection_if_needed()

    # 逐 batch encode + insert
    bs = cfg.ingest.image_batch_size
    for bi, (records_b, images_b) in enumerate(iter_image_batches(records, bs), 1):
        emb_cpu = encoder.encode_images(images_b)  # [B, patches, dim] on CPU
        store.insert_pages_with_patches_batch(records_b, emb_cpu)

        # 額外保險：讓 cuda caching 回收（通常不必每次做，但 OOM 時有幫助）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if bi % 10 == 0:
            print(f"[ingest] batches={bi} pages_done={bi*bs}/{len(records)}")

    store.build_index_and_load(cfg.index)


def retrieve(cfg: AppConfig, queries: Sequence[str]) -> None:
    encoder = ColQwen2Encoder(cfg.model)
    store = MilvusDocStore(cfg.milvus)

    query_embeddings = encoder.encode_queries(list(queries))
    results = store.search_pages(query_embeddings, cfg.search)

    hits = results[0]
    print(f"\nQuery: '{queries[0]}'")
    for i, hit in enumerate(hits, 1):
        entity = hit.entity
        print(f"{i}. {entity['doc_name']} - Page {entity['page_number']}")
        print(f"   Score: {hit.distance:.4f}\n")

def retrieve_structured(cfg: AppConfig, queries: Sequence[str]) -> List[Dict[str, Any]]:
    encoder = ColQwen2Encoder(cfg.model)
    store = MilvusDocStore(cfg.milvus)

    query_embeddings = encoder.encode_queries(list(queries))  # [B, n_tokens, dim]
    results = store.search_pages(query_embeddings, cfg.search)

    rows: List[Dict[str, Any]] = []
    for q, hits in zip(queries, results):
        for rank, hit in enumerate(hits, 1):
            ent = hit.entity
            rows.append(
                {
                    "question": q,
                    "rank": rank,
                    "doc_name": ent.get("doc_name"),
                    "page_number": ent.get("page_number"),
                    "page_id": ent.get("page_id"),
                    "score": float(hit.distance),
                }
            )
    return rows

def iter_batches(xs: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), batch_size):
        yield list(xs[i : i + batch_size])


def batch_search(
    cfg: AppConfig,
    csv_path: str,
    question_col: str = "question",
    batch_size: int = 16,
    output_csv: str = "batch_retrieval_results.csv",
) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if question_col not in df.columns:
        raise ValueError(f"CSV missing column '{question_col}'. Available={list(df.columns)}")

    df = df.drop_duplicates(subset=[question_col], keep="first")
    queries = [q for q in df[question_col].astype(str).tolist() if q.strip()]

    print(f"[batch] loaded={len(df)} unique_questions={len(queries)} batch_size={batch_size}")

    all_rows: List[Dict[str, Any]] = []
    for bi, q_batch in enumerate(iter_batches(queries, batch_size), 1):
        rows = retrieve_structured(cfg, q_batch)
        all_rows.extend(rows)
        print(f"[batch] batch={bi} queries={len(q_batch)} rows={len(rows)} total_rows={len(all_rows)}")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[batch] wrote -> {output_csv} (rows={len(out_df)})")

# =========================
# Entry
# =========================

if __name__ == "__main__":
    cfg = AppConfig()

    # 1) One-time offline build (PDF -> images -> embeddings -> Milvus)
    #start_time = time.time()
    #offline_build(cfg)
    #end_time = time.time()
    #print(f"[offline] Total time: {end_time - start_time:.2f} seconds")

    # 2) Retrieval
    start_time = time.time()
    retrieve(
        cfg,
        queries=[
            "Why is managing the etch rate and improving Aspect Ratio Dependent Etching (ARDE) crucial for scaling High Aspect Ratio (HAR) etches in 3D NAND channel hole patterning?"
        ],
    )
    end_time = time.time()
    print(f"[retrieve] Total time: {end_time - start_time:.2f} seconds")
    #batch_search(
    #    cfg,
    #    csv_path="golden_QA_3D_nand_original_51.csv",
    #    question_col="question",
    #    batch_size=16,
    #    output_csv="golden51_colqwen2_milvus_topk5.csv",
    #)