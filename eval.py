from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Set, Dict, Tuple, Optional

import pandas as pd


# -------------------------
# Configurable knobs
# -------------------------

@dataclass(frozen=True)
class EvalConfig:
    gt_path: str
    pred_path: str
    question_col_gt: str = "question"
    question_col_pred: str = "question"
    key_pdf_col: str = "key_pdf"
    key_pages_col: str = "key_pages"
    pred_doc_col: str = "doc_name"
    pred_page_col: str = "page_number"
    pred_rank_col: str = "rank"
    ks: Tuple[int, ...] = (1, 3, 5, 10)
    strict_doc_match: bool = True  # True: exact match; False: normalize + loose match


# -------------------------
# Normalization / parsing
# -------------------------

def _normalize_text(s: str) -> str:
    # Normalize whitespace + strip; keep it conservative to avoid false matches
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_doc_name(s: str) -> str:
    # For loose match mode: normalize spaces; you can also lower() if you want
    return _normalize_text(s)


def parse_pages(pages_field: object) -> Set[int]:
    """
    Accepts:
      - "4,5"
      - "3"
      - 3
      - NaN/None -> empty set
    """
    if pages_field is None:
        return set()
    if isinstance(pages_field, float) and pd.isna(pages_field):
        return set()

    s = str(pages_field).strip()
    if not s:
        return set()

    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: Set[int] = set()
    for p in parts:
        # keep digits only
        m = re.match(r"^\d+$", p)
        if m:
            out.add(int(p))
        else:
            # if you have formats like "4-6" in future, you can extend here
            raise ValueError(f"Unsupported key_pages token: {p!r} in {s!r}")
    return out


def doc_match(pred_doc: str, gt_doc: str, strict: bool) -> bool:
    if strict:
        return _normalize_text(pred_doc) == _normalize_text(gt_doc)
    return _normalize_doc_name(pred_doc) == _normalize_doc_name(gt_doc)


# -------------------------
# Core metrics
# -------------------------

def compute_hit_rates(cfg: EvalConfig) -> pd.DataFrame:
    gt = pd.read_csv(cfg.gt_path)
    pred = pd.read_csv(cfg.pred_path)

    # Basic cleanup
    gt[cfg.question_col_gt] = gt[cfg.question_col_gt].astype(str).map(_normalize_text)
    pred[cfg.question_col_pred] = pred[cfg.question_col_pred].astype(str).map(_normalize_text)

    gt[cfg.key_pdf_col] = gt[cfg.key_pdf_col].astype(str).map(_normalize_text)
    pred[cfg.pred_doc_col] = pred[cfg.pred_doc_col].astype(str).map(_normalize_text)

    # Ensure numeric
    pred[cfg.pred_rank_col] = pd.to_numeric(pred[cfg.pred_rank_col], errors="coerce").astype("Int64")
    pred[cfg.pred_page_col] = pd.to_numeric(pred[cfg.pred_page_col], errors="coerce").astype("Int64")

    # Build GT map: question -> (key_pdf, key_pages_set)
    gt_map: Dict[str, Tuple[str, Set[int]]] = {}
    for _, row in gt.iterrows():
        q = row[cfg.question_col_gt]
        key_pdf = row[cfg.key_pdf_col]
        key_pages = parse_pages(row.get(cfg.key_pages_col, None))
        gt_map[q] = (key_pdf, key_pages)

    # Group predictions by question, sort by rank
    pred_grouped = (
        pred.dropna(subset=[cfg.pred_rank_col])
            .sort_values([cfg.question_col_pred, cfg.pred_rank_col], ascending=[True, True])
            .groupby(cfg.question_col_pred, sort=False)
    )

    questions = sorted(gt_map.keys())
    if not questions:
        raise ValueError("No ground-truth questions found.")

    rows = []
    for K in cfg.ks:
        doc_hits = 0
        page_hits = 0
        missing_pred = 0

        for q in questions:
            key_pdf, key_pages = gt_map[q]

            if q not in pred_grouped.groups:
                missing_pred += 1
                continue

            topk = pred_grouped.get_group(q).head(K)

            # doc-level: any doc matches
            doc_hit = any(doc_match(d, key_pdf, cfg.strict_doc_match) for d in topk[cfg.pred_doc_col].tolist())

            # page-level: any (doc matches AND page in key_pages)
            page_hit = False
            if key_pages:
                for _, r in topk.iterrows():
                    d = r[cfg.pred_doc_col]
                    p = r[cfg.pred_page_col]
                    if pd.isna(p):
                        continue
                    if doc_match(d, key_pdf, cfg.strict_doc_match) and int(p) in key_pages:
                        page_hit = True
                        break

            doc_hits += int(doc_hit)
            page_hits += int(page_hit)

        denom = len(questions)  # HitRate is defined over all GT questions
        rows.append(
            {
                "K": K,
                "num_questions": denom,
                "missing_pred_questions": missing_pred,
                "doc_hit_rate": doc_hits / denom,
                "page_hit_rate": page_hits / denom,
            }
        )

    return pd.DataFrame(rows)


# -------------------------
# Optional: per-question debug table
# -------------------------

def build_per_question_debug(cfg: EvalConfig, K: int) -> pd.DataFrame:
    gt = pd.read_csv(cfg.gt_path)
    pred = pd.read_csv(cfg.pred_path)

    gt[cfg.question_col_gt] = gt[cfg.question_col_gt].astype(str).map(_normalize_text)
    pred[cfg.question_col_pred] = pred[cfg.question_col_pred].astype(str).map(_normalize_text)

    gt[cfg.key_pdf_col] = gt[cfg.key_pdf_col].astype(str).map(_normalize_text)
    pred[cfg.pred_doc_col] = pred[cfg.pred_doc_col].astype(str).map(_normalize_text)

    pred[cfg.pred_rank_col] = pd.to_numeric(pred[cfg.pred_rank_col], errors="coerce").astype("Int64")
    pred[cfg.pred_page_col] = pd.to_numeric(pred[cfg.pred_page_col], errors="coerce").astype("Int64")

    gt_map = {
        row[cfg.question_col_gt]: (row[cfg.key_pdf_col], parse_pages(row.get(cfg.key_pages_col, None)))
        for _, row in gt.iterrows()
    }

    pred_grouped = (
        pred.dropna(subset=[cfg.pred_rank_col])
            .sort_values([cfg.question_col_pred, cfg.pred_rank_col], ascending=[True, True])
            .groupby(cfg.question_col_pred, sort=False)
    )

    out = []
    for q, (key_pdf, key_pages) in gt_map.items():
        if q not in pred_grouped.groups:
            out.append(
                {
                    "question": q,
                    "key_pdf": key_pdf,
                    "key_pages": ",".join(map(str, sorted(key_pages))),
                    "doc_hit@K": False,
                    "page_hit@K": False,
                    "topk_docs": "",
                    "topk_pages": "",
                }
            )
            continue

        topk = pred_grouped.get_group(q).head(K)
        docs = topk[cfg.pred_doc_col].tolist()
        pages = [("" if pd.isna(x) else int(x)) for x in topk[cfg.pred_page_col].tolist()]

        doc_hit = any(doc_match(d, key_pdf, cfg.strict_doc_match) for d in docs)

        page_hit = False
        if key_pages:
            for d, p in zip(docs, pages):
                if p == "":
                    continue
                if doc_match(d, key_pdf, cfg.strict_doc_match) and int(p) in key_pages:
                    page_hit = True
                    break

        out.append(
            {
                "question": q,
                "key_pdf": key_pdf,
                "key_pages": ",".join(map(str, sorted(key_pages))),
                f"doc_hit@{K}": bool(doc_hit),
                f"page_hit@{K}": bool(page_hit),
                "topk_docs": " | ".join(docs),
                "topk_pages": ",".join(map(str, pages)),
            }
        )
    return pd.DataFrame(out)


if __name__ == "__main__":
    cfg = EvalConfig(
        gt_path="golden_QA_3D_nand_original_51.csv",
        pred_path="golden51_colqwen2_milvus_topk5.csv",
        ks=(1, 2, 3, 4, 5),
        strict_doc_match=True,
    )

    summary = compute_hit_rates(cfg)
    print(summary.to_string(index=False))

    # If you want a debug dump for K=3:
    # debug = build_per_question_debug(cfg, K=3)
    # debug.to_csv("debug_per_question_k3.csv", index=False, encoding="utf-8-sig")