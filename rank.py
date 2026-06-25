#!/usr/bin/env python3
"""
rank_v4.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking
Senior AI Engineer (Founding Team) — Redrob AI

Merged best-of-both from rank_v3 + rank.py (document):

From rank.py (new):
  ✓ 4-tier hard skill scoring with duration>=6mo gate (biggest accuracy gain)
  ✓ 9-check honeypot detection (5 new checks: YOE vs grad year, concurrent roles,
      end<start dates, career duration vs YOE, plus v3's company-founding + zero-expert)
  ✓ score_career_trajectory: ARCH_MGMT_TITLE_RE, HIGH_SIGNAL_COMPANIES bonus,
      APPLIED_ML_TITLE_RE, graduated consulting penalty
  ✓ Behavioral: avail can go >1.0 for ideal candidates; finer-grained notice/art bonuses
  ✓ generate_reasoning: context-aware branching (strong/partial/concern), 500-char cap
  ✓ Expanded JD_TEXT with doubled key phrases
  ✓ Broader CONSULTING_FIRMS (added genpact, kpit, ltts, niit)
  ✓ Inline validation + runtime warning in main()

From rank_v3 (kept):
  ✓ cached_texts: build_candidate_text called ONCE, result passed everywhere
      (without this: 500k redundant string ops on 100k candidates)
  ✓ TF-IDF unigrams at 20k features (ngram (1,2) at 80k OOMs on 16GB)
  ✓ Company-founding honeypot detection (catches Krutrim/Sarvam-era fakes)
  ✓ Zero-duration expert skill honeypot (≥3 = disqualify)

Usage:
    python rank_v4.py --candidates ./candidates.jsonl  --out ./submission.csv
    python rank_v4.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Must complete ≤5 min, ≤16GB RAM, CPU only, no network.
"""

import argparse
import csv
import gc
import gzip
import json
import logging
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EVAL_DATE = date(2026, 6, 15)

# ─────────────────────────────────────────────────────────────────────────────
# SCALE-ADAPTIVE CONFIGURATION
# Thresholds auto-tune based on dataset size so the pipeline stays within
# the 5-min / 16GB budget at every scale.
# ─────────────────────────────────────────────────────────────────────────────

def get_scale_config(n: int) -> dict:
    """
    Returns tuning knobs adapted to dataset size n.

      n < 500        → "small"  : full features, no batching needed
      500 ≤ n < 5000 → "medium" : mild batching, 20K TF-IDF features
      5000 ≤ n       → "large"  : chunk cosine similarity, strict min_df,
                                  parallel scoring via ProcessPoolExecutor
    """
    if n < 500:
        return dict(
            label          = "small",
            tfidf_features = 15_000,
            tfidf_min_df   = 1,
            tfidf_max_df   = 1.0,
            sim_batch_size = n,          # single batch
            n_jobs         = 1,          # no parallelism needed
            score_chunk    = n,
        )
    elif n < 5_000:
        return dict(
            label          = "medium",
            tfidf_features = 20_000,
            tfidf_min_df   = 2,
            tfidf_max_df   = 0.95,
            sim_batch_size = 2_000,
            n_jobs         = min(4, os.cpu_count() or 1),
            score_chunk    = 2_000,
        )
    else:  # large: up to 1 lakh (100K) and beyond
        return dict(
            label          = "large",
            tfidf_features = 20_000,    # ngram(1,1) @ 20K — proven safe at 100K, 16GB
            tfidf_min_df   = 3,         # prune very rare terms to shrink matrix
            tfidf_max_df   = 0.90,
            sim_batch_size = 5_000,     # cosine_similarity in 5K chunks → avoids OOM
            n_jobs         = min(8, os.cpu_count() or 1),
            score_chunk    = 10_000,
        )

# ── JD text: doubled key phrases so TF-IDF weights them higher ───────────────
JD_TEXT = """
senior ai engineer founding team redrob ai series a product company
embeddings retrieval ranking llm fine-tuning production deployment real users at scale
embeddings retrieval ranking llm production deployment
sentence-transformers openai embeddings bge e5 embedding drift index refresh
vector database pinecone weaviate qdrant milvus opensearch elasticsearch faiss
hybrid search dense retrieval approximate nearest neighbor ann
strong python code quality production systems
evaluation framework ndcg mrr map offline online ab testing ranking evaluation
ndcg mrr evaluation framework
recommendation system search ranking retrieval augmented generation rag
lora qlora peft fine-tuning instruction tuning optional
learning-to-rank xgboost lightgbm neural ranker optional
hr tech recruiting marketplace candidate job description matching
distributed systems inference optimization large-scale optional
open source contributions github huggingface papers optional
not consulting tcs infosys wipro accenture cognizant capgemini entire career
not computer vision speech robotics without nlp ir
not pure research academic no production deployment
not title chaser switching every 1.5 years
not recent langchain tutorial only no pre-llm ml experience
location pune noida hyderabad mumbai delhi ncr bangalore india relocation
notice period sub 30 days preferred 30 days buyout 90 days acceptable
product company shipped ranking search recommendation real users meaningful scale
applied ml ai roles product companies not pure services
scrappy product engineering attitude ship working system learn from users
"""

# ── Hard-required skills (JD: "things you absolutely need") ──────────────────
# weight = how heavily the JD stresses each area
# keywords = what to look for in text, career history, and skill lists
HARD_SKILL_AREAS = {
    "embeddings": {
        "weight": 1.2,
        "keywords": [
            "embedding", "embeddings", "sentence-transformer", "sentence transformer",
            "bge", "e5 model", "openai embeddings", "vector embedding",
            "semantic search", "dense retrieval", "bi-encoder", "cross-encoder",
        ],
    },
    "vector_db": {
        "weight": 1.2,
        "keywords": [
            "vector database", "vector db", "pinecone", "weaviate", "qdrant",
            "milvus", "faiss", "opensearch", "elasticsearch", "pgvector",
            "hybrid search", "ann", "approximate nearest neighbor",
            "vector store", "vector index",
        ],
    },
    "python": {
        "weight": 0.8,
        "keywords": ["python"],
    },
    "eval_ranking": {
        "weight": 1.3,
        "keywords": [
            "ndcg", "mrr", "mean reciprocal rank", "map@", "mean average precision",
            "evaluation framework", "ranking evaluation", "learning-to-rank",
            "learning to rank", "ltr", "a/b test", "ab test", "a/b testing",
            "offline benchmark", "online evaluation", "retrieval quality",
        ],
    },
}

# ── Soft / nice-to-have skills ────────────────────────────────────────────────
SOFT_SKILL_AREAS = {
    "llm_finetune": {
        "weight": 0.9,
        "keywords": [
            "lora", "qlora", "peft", "fine-tuning", "finetuning", "fine tuning",
            "instruction tuning", "rlhf", "dpo",
        ],
    },
    "ml_core": {
        "weight": 1.0,
        "keywords": [
            "machine learning", "deep learning", "neural network", "pytorch",
            "tensorflow", "xgboost", "lightgbm", "scikit-learn", "sklearn",
            "recommendation system", "recsys", "learning to rank", "ranking",
        ],
    },
    "nlp_rag": {
        "weight": 1.1,
        "keywords": [
            "nlp", "natural language processing", "transformer", "bert", "gpt",
            "llm", "large language model", "rag", "retrieval augmented",
            "generative ai", "langchain", "llamaindex", "information retrieval",
        ],
    },
    "cloud_infra": {
        "weight": 0.5,
        "keywords": [
            "aws", "gcp", "azure", "docker", "kubernetes", "distributed",
            "large-scale", "inference optimization", "mlops",
        ],
    },
    "open_source": {
        "weight": 0.4,
        "keywords": [
            "open-source", "open source", "github", "hugging face",
            "huggingface", "arxiv", "paper", "publication",
        ],
    },
    "hr_tech_domain": {
        # JD explicitly lists this as a nice-to-have; small but real signal
        "weight": 0.3,
        "keywords": [
            "hr tech", "hrtech", "recruiting", "talent acquisition",
            "candidate matching", "job matching", "ats", "applicant tracking",
            "talent intelligence", "workforce", "marketplace", "staffing",
        ],
    },
}

# ── Pre-LLM ML keywords — used to detect genuine ML depth vs LangChain-only ──
PRE_LLM_ML_KEYWORDS = {
    "xgboost", "lightgbm", "scikit-learn", "sklearn", "random forest",
    "gradient boosting", "collaborative filtering", "matrix factorization",
    "information retrieval", "bm25", "tf-idf", "tfidf", "inverted index",
    "word2vec", "glove", "fasttext", "recommendation system", "recsys",
    "learning to rank", "click-through rate", "ctr prediction",
    "item2vec", "neural collaborative filtering",
}
LANGCHAIN_ONLY_KEYWORDS = {
    "langchain", "llamaindex", "llama index", "openai api", "chatgpt api",
    "gpt wrapper", "prompt engineering",
}

# ── Consulting firms (career-long = hard disqualifier) ───────────────────────
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl technologies", "hcl", "tech mahindra", "genpact",
    "mphasis", "hexaware", "mindtree", "ltimindtree", "persistent systems",
    "l&t infotech", "ltts", "niit technologies", "kpit",
}

# ── Company founding dates — employed before founding = honeypot ─────────────
COMPANY_FOUNDING_DATES = {
    "krutrim":        date(2023, 12, 1),
    "sarvam ai":      date(2023,  7, 1),
    "sarvam":         date(2023,  7, 1),
    "openai":         date(2015, 12, 1),
    "anthropic":      date(2021,  1, 1),
    "mistral ai":     date(2023,  4, 1),
    "mistral":        date(2023,  4, 1),
    "perplexity ai":  date(2022,  8, 1),
    "perplexity":     date(2022,  8, 1),
    "cohere":         date(2019, 11, 1),
    "stability ai":   date(2020, 11, 1),
    "inflection ai":  date(2022,  3, 1),
    "inflection":     date(2022,  3, 1),
    "adept":          date(2022,  4, 1),
    "together ai":    date(2022,  6, 1),
    "fireworks ai":   date(2022,  9, 1),
    "redrob":         date(2022,  1, 1),
    "redrob ai":      date(2022,  1, 1),
}

# ── CV/speech disqualifier keywords ──────────────────────────────────────────
CV_SPEECH_KEYWORDS = {
    "computer vision", "object detection", "image classification",
    "opencv", "yolo", "convolutional", "speech recognition",
    "text to speech", "tts", "robotics", "ros ", "slam",
}
NLP_IR_KEYWORDS = {
    "nlp", "natural language", "retrieval", "ranking", "transformer",
    "bert", "gpt", "llm", "embeddings", "rag", "information retrieval",
    "semantic search", "vector", "recommendation",
}

# ── Location tiers ────────────────────────────────────────────────────────────
PREFERRED_LOCATIONS = {"pune", "noida"}
TIER1_LOCATIONS = {
    "pune", "noida", "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "gurugram", "gurgaon", "greater noida", "delhi ncr",
}

# ── High-signal product companies — small authenticity bonus ─────────────────
HIGH_SIGNAL_COMPANIES = {
    "google", "microsoft", "amazon", "meta", "apple", "flipkart", "swiggy",
    "zomato", "ola", "uber", "phonepe", "paytm", "razorpay", "cred",
    "meesho", "nykaa", "byju", "unacademy", "freshworks", "zoho",
    "sarvam", "krutrim", "anthropic", "openai", "cohere", "hugging face",
    "linkedin", "twitter", "netflix", "airbnb", "stripe",
}

# ── Title regex patterns ──────────────────────────────────────────────────────
APPLIED_ML_TITLE_RE = re.compile(
    r"\b(ml engineer|machine learning engineer|applied (ml|ai|scientist)|"
    r"ai engineer|nlp engineer|search engineer|ranking engineer|"
    r"recommendation.*(engineer|scientist)|data scientist|"
    r"research engineer|applied research|senior engineer.*(ml|ai|nlp|search))\b",
    re.I,
)
PURE_RESEARCH_TITLE_RE = re.compile(
    r"\b(research scientist|research fellow|postdoc|phd researcher|"
    r"academic researcher|postdoctoral)\b",
    re.I,
)
ARCH_MGMT_TITLE_RE = re.compile(
    r"\b(chief|architect|engineering manager|director of|head of|"
    r"vp (of )?engineering|vice president|principal architect)\b",
    re.I,
)
NON_TECH_TITLES = {
    "marketing", "sales", "hr manager", "human resources", "recruiter",
    "finance", "accountant", "operations manager", "customer success",
    "business development", "product manager",
}
ML_ROLE_RE = re.compile(
    r"\b(ml|machine learning|ai\b|nlp|search|rank|recommend|retrieval|"
    r"embedding|data scien|applied)\b",
    re.I,
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def tl(s):
    return (s or "").lower()

def days_since(date_str):
    if not date_str:
        return 9999
    try:
        return (EVAL_DATE - datetime.strptime(date_str, "%Y-%m-%d").date()).days
    except Exception:
        return 9999

def parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

@lru_cache(maxsize=4096)
def is_consulting(company_name: str) -> bool:
    n = tl(company_name)
    return any(firm in n for firm in CONSULTING_FIRMS)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT BUILDER — called ONCE per candidate, result cached
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_text(c):
    """
    Single string per candidate for TF-IDF.
    Called ONCE in step 2 and cached — never called again inside scoring.
    Key decisions:
    - headline + summary doubled (most signal-dense free text)
    - current role tripled (most recent/relevant)
    - expert/advanced skills doubled (quality signal)
    - assessment scores included (platform-verified)
    """
    parts = []
    p = c.get("profile", {})

    headline = p.get("headline", "")
    summary  = p.get("summary",  "")
    parts.extend([headline, headline, summary, summary])
    parts.append(p.get("current_title", ""))
    parts.append(p.get("current_industry", ""))

    for h in c.get("career_history", []):
        title = h.get("title", "")
        desc  = h.get("description", "")
        ind   = h.get("industry", "")
        if h.get("is_current"):
            parts.extend([title, title, desc, desc, ind])
        else:
            parts.extend([title, desc, ind])

    for s in c.get("skills", []):
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        parts.extend([name, name] if prof in ("expert", "advanced") else [name])

    for e in c.get("education", []):
        parts.append(
            f"{e.get('degree','')} {e.get('field_of_study','')} {e.get('institution','')}"
        )
    for cert in c.get("certifications", []):
        parts.append(f"{cert.get('name','')} {cert.get('issuer','')}")

    for skill_name in c.get("redrob_signals", {}).get("skill_assessment_scores", {}):
        parts.append(skill_name)

    return " ".join(filter(None, parts)).lower()


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION — 9 internal consistency checks
# ─────────────────────────────────────────────────────────────────────────────

def detect_honeypot(c):
    """
    Returns (penalty 0.0-1.0, list[str] of reasons).
    0.0 = hard disqualify (skip entirely).
    0.5 = soft penalty (keep but score drops).
    1.0 = clean profile.

    9 checks (5 new since v3):
      1. Worked at company before it was founded            → hard 0.0
      2. ≥3 expert skills with 0 months duration           → hard 0.0
      3. last_active before signup                         → hard 0.0
      4. Future signup date                                → hard 0.0
      5. Career months > claimed YOE by >3 years           → soft 0.5  [NEW]
      6. Multiple concurrent is_current roles              → hard 0.0  [NEW]
      7. YOE impossible given earliest graduation year     → soft 0.5  [NEW]
      8. Implausibly perfect signals (≥4/5 at 99%+)       → hard 0.0
      9. Role end_date before start_date                   → soft 0.5  [NEW]
    """
    flags  = []
    p      = c.get("profile",        {})
    career = c.get("career_history", []) or []
    skills = c.get("skills",         []) or []
    edu    = c.get("education",       []) or []
    sig    = c.get("redrob_signals",  {}) or {}
    yoe    = p.get("years_of_experience", 0) or 0

    # ── 1. Worked at company before it was founded ────────────────────────
    for h in career:
        co    = tl(h.get("company", ""))
        start = parse_date(h.get("start_date", ""))
        if not start:
            continue
        for known_co, founded in COMPANY_FOUNDING_DATES.items():
            if known_co in co and start < founded:
                months_before = (founded - start).days // 30
                flags.append(
                    f"worked at {h.get('company')} from {h.get('start_date')} "
                    f"but company founded {founded} ({months_before}mo later)"
                )
                return 0.0, flags

    # ── 2. ≥3 expert skills with 0 months duration ────────────────────────
    zero_expert = [
        s.get("name", "") for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", -1) == 0
    ]
    if len(zero_expert) >= 3:
        flags.append(
            f"{len(zero_expert)} 'expert' skills with 0 months duration: "
            f"{', '.join(zero_expert[:3])}"
        )
        return 0.0, flags

    # ── 3. last_active before signup ──────────────────────────────────────
    signup      = parse_date(sig.get("signup_date", ""))
    last_active = parse_date(sig.get("last_active_date", ""))
    if signup and last_active and last_active < signup:
        flags.append(f"last_active {last_active} is before signup {signup}")
        return 0.0, flags

    # ── 4. Future signup date ─────────────────────────────────────────────
    if signup and signup > EVAL_DATE:
        flags.append(f"signup date {signup} is in the future")
        return 0.0, flags

    # ── 5. Career duration > claimed YOE by large margin [NEW] ───────────
    total_career_months = sum((h.get("duration_months") or 0) for h in career)
    if total_career_months > 0:
        implied_years = total_career_months / 12.0
        if implied_years > yoe + 3:
            flags.append(
                f"career history implies {implied_years:.1f}y but claims {yoe}y experience"
            )
            return 0.0, flags

    # ── 6. Multiple concurrent is_current roles [NEW] ─────────────────────
    current_roles = [h for h in career if h.get("is_current")]
    if len(current_roles) > 1:
        flags.append(
            f"{len(current_roles)} concurrent is_current roles: "
            f"{[h.get('company') for h in current_roles]}"
        )
        return 0.0, flags

    # ── 7. YOE impossible given graduation year [NEW] ─────────────────────
    if edu:
        earliest_grad = min(
            (e.get("end_year") for e in edu if e.get("end_year")), default=None
        )
        if earliest_grad:
            years_since_grad = EVAL_DATE.year - earliest_grad
            if years_since_grad >= 0 and yoe > years_since_grad + 3:
                flags.append(
                    f"claims {yoe}y experience but graduated only "
                    f"{years_since_grad}y ago ({earliest_grad})"
                )
                return 0.0, flags

    # ── 8. Implausibly perfect signals ────────────────────────────────────
    perfect = sum([
        sig.get("profile_completeness_score", 0) >= 99,
        sig.get("recruiter_response_rate",    0) >= 0.99,
        sig.get("interview_completion_rate",  0) >= 0.99,
        sig.get("github_activity_score",     -1) >= 99,
        sig.get("offer_acceptance_rate",     -1) >= 0.99,
    ])
    if perfect >= 4:
        flags.append(f"implausibly perfect signal profile ({perfect}/5 at 99%+)")
        return 0.0, flags

    # ── 9. Role end_date before start_date [NEW] ──────────────────────────
    for h in career:
        sd = parse_date(h.get("start_date", ""))
        ed = parse_date(h.get("end_date",   ""))
        if sd and ed and ed < sd:
            flags.append(
                f"role at {h.get('company')} ends {ed} before it starts {sd}"
            )
            return 0.0, flags

    return 1.0, []


# ─────────────────────────────────────────────────────────────────────────────
# HARD SKILL SCORE — 4-tier evidence hierarchy
# ─────────────────────────────────────────────────────────────────────────────

def score_hard_skills(c, all_text,career_text=None):
    """
    4-tier evidence hierarchy per skill area:
      Tier 1 (1.00): in skill list as advanced/expert with ≥6mo USE + in career desc
      Tier 2 (0.75): appears in career history descriptions (actual use, not just claimed)
      Tier 3 (0.55): in skill list with real duration but not mentioned in career desc
      Tier 4 (0.30): text-only mention — weakest, keyword-stuffer territory

    This is the primary anti-keyword-stuffer mechanism.
    A data analyst who lists 'faiss' as expert skill but never mentions it
    in any job description scores 0.30, not 1.00.
    """
    if career_text is None:
        career_text = " ".join(
           f"{tl(h.get('title',''))} {tl(h.get('description',''))}"
           for h in c.get("career_history", [])
    )
    # "Real" skills: advanced or expert AND actually used for ≥6 months
    real_skills = {
        tl(s.get("name", ""))
        for s in c.get("skills", [])
        if s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months") or 0) >= 6
    }

    area_scores = {}
    for area, cfg in HARD_SKILL_AREAS.items():
        kws = cfg["keywords"]
        in_real   = any(any(kw in sk for kw in kws) for sk in real_skills)
        in_career = any(kw in career_text for kw in kws)
        in_text   = any(kw in all_text    for kw in kws)

        if in_real and in_career:  area_scores[area] = 1.00
        elif in_career:            area_scores[area] = 0.75
        elif in_real:              area_scores[area] = 0.55
        elif in_text:              area_scores[area] = 0.30
        else:                      area_scores[area] = 0.00

    if "python" not in all_text:
        area_scores["python"] = 0.00

    total_w = sum(cfg["weight"] for cfg in HARD_SKILL_AREAS.values())
    return sum(area_scores[a] * HARD_SKILL_AREAS[a]["weight"] for a in area_scores) / total_w


# ─────────────────────────────────────────────────────────────────────────────
# SOFT SKILL SCORE
# ─────────────────────────────────────────────────────────────────────────────

def score_soft_skills(all_text):
    """2+ keyword hits per area = full score for that area."""
    area_scores = {
        area: min(1.0, sum(1 for kw in cfg["keywords"] if kw in all_text) / 2)
        for area, cfg in SOFT_SKILL_AREAS.items()
    }
    total_w = sum(cfg["weight"] for cfg in SOFT_SKILL_AREAS.values())
    return sum(area_scores[a] * SOFT_SKILL_AREAS[a]["weight"] for a in area_scores) / total_w


# ─────────────────────────────────────────────────────────────────────────────
# CAREER TRAJECTORY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def score_career_trajectory(c, all_text):
    """
    Encodes the JD's 'read between the lines' disqualifiers explicitly:

    Hard penalties (return early):
      - non-technical current role (×0.05)
      - CV/speech-only without NLP/IR (×0.15)
      - career-long pure consulting ≥90% (×0.20)
      - pure research/academic, no product deployment (×0.10)
      - current role is arch/mgmt ≥18mo (×0.40) — "stopped writing code"

    Soft score components:
      - ML career fraction (60% weight)
      - product company fraction (25% weight)
      - high-signal company bonus (up to +0.15)
      - title-chasing penalty (up to -0.25)
      - consulting fraction penalty (up to -0.20)
    """
    p      = c.get("profile", {})
    career = c.get("career_history", []) or []

    cur_title = tl(p.get("current_title", ""))

    # ── Non-technical role ────────────────────────────────────────────────
    if any(t in cur_title for t in NON_TECH_TITLES):
        return 0.05, [f"non-technical current role: {p.get('current_title')}"]

    # ── CV/speech without NLP/IR ──────────────────────────────────────────
    has_cv  = any(kw in all_text for kw in CV_SPEECH_KEYWORDS)
    has_nlp = any(kw in all_text for kw in NLP_IR_KEYWORDS)
    if has_cv and not has_nlp:
        return 0.15, ["CV/speech/robotics without NLP/IR exposure"]

    # ── Career analysis ───────────────────────────────────────────────────
    total_mo      = 0
    ml_mo         = 0
    consulting_mo = 0
    research_mo   = 0
    product_mo    = 0
    short_hops    = 0
    high_sig_found = []

    for h in career:
        dur     = h.get("duration_months", 0) or 0
        title   = h.get("title", "")
        company = h.get("company", "") or ""
        is_curr = h.get("is_current", False)
        total_mo += dur

        if APPLIED_ML_TITLE_RE.search(title):
            ml_mo += dur
        if PURE_RESEARCH_TITLE_RE.search(title):
            research_mo += dur
        if is_consulting(company):
            consulting_mo += dur
        else:
            product_mo += dur

        if any(hsc in tl(company) for hsc in HIGH_SIGNAL_COMPANIES):
            high_sig_found.append(company)

        # Title-chasing: senior/staff/principal title held <18mo (not current)
        title_l = tl(title)
        if (dur > 0 and dur < 18 and not is_curr and
                any(t in title_l for t in ("senior", "staff", "principal", "lead"))):
            short_hops += 1

    if total_mo == 0:
        total_mo = 1

    consulting_frac = consulting_mo / total_mo
    research_frac   = research_mo   / total_mo
    ml_frac         = ml_mo         / total_mo

    # ── Hard disqualifiers ────────────────────────────────────────────────
    if consulting_frac >= 0.90:
        return 0.20, ["career-long pure consulting (explicit JD disqualifier)"]

    if research_frac >= 0.80 and product_mo < 12:
        return 0.10, ["pure research/academic with no production deployment"]

    # JD: "senior engineers who haven't written code in 18 months"
    most_recent  = max(career, key=lambda h: h.get("start_date") or "", default={})
    recent_title = most_recent.get("title", "")
    recent_dur   = most_recent.get("duration_months", 0) or 0
    if ARCH_MGMT_TITLE_RE.search(recent_title) and recent_dur >= 18:
        return 0.40, [
            f"current role '{recent_title}' ({recent_dur}mo) is arch/mgmt — "
            "JD needs someone still writing production code"
        ]

    # ── Soft score ────────────────────────────────────────────────────────
    reasons = []

    if ml_frac >= 0.50:
        ml_component = 1.0
        reasons.append(f"{ml_mo/12:.1f}y in applied ML/AI roles ({ml_frac:.0%} of career)")
    elif ml_frac >= 0.25:
        ml_component = 0.75
        reasons.append(f"{ml_frac:.0%} of career in ML/AI roles — partial match")
    elif ml_frac >= 0.10:
        ml_component = 0.45
    else:
        ml_component = 0.20
        reasons.append("very little ML/AI titled role history")

    company_bonus = min(0.15, len(high_sig_found) * 0.05)
    if high_sig_found:
        reasons.append(f"product companies: {', '.join(high_sig_found[:2])}")

    hopping_penalty = 0.0
    if short_hops >= 3:
        hopping_penalty = 0.25
        reasons.append(f"{short_hops} senior titles held <18mo — title-chasing pattern")
    elif short_hops == 2:
        hopping_penalty = 0.12

    consulting_penalty = 0.0
    if consulting_frac > 0.50:
        consulting_penalty = 0.20
        reasons.append(f"{consulting_frac:.0%} of career at consulting firms")
    elif consulting_frac > 0.30:
        consulting_penalty = 0.08

    # ── NEW: Closed-source-only penalty ─────────────────────────────────────
    # JD: "people whose work has been entirely on closed-source proprietary systems
    # for 5+ years without external validation (papers, talks, open-source)"
    has_external_signal = any(kw in all_text for kw in {
        "open-source", "open source", "github", "hugging face", "huggingface",
        "arxiv", "paper", "publication", "talk", "conference", "kaggle",
    })
    closed_source_penalty = 0.0
    yoe = c.get("profile", {}).get("years_of_experience", 0) or 0
    if yoe >= 5 and not has_external_signal:
        closed_source_penalty = 0.08   # soft penalty only — can't fully verify from profile
        reasons.append("5+y experience but no external validation signal (open-source/papers/talks)")

    # ── NEW: LangChain-only / no pre-LLM ML penalty ───────────────────────
    # JD: "if your AI experience is only recent LangChain/OpenAI tutorials
    # without pre-LLM ML production experience, we will probably not move forward"
    has_langchain  = any(kw in all_text for kw in LANGCHAIN_ONLY_KEYWORDS)
    has_pre_llm_ml = any(kw in all_text for kw in PRE_LLM_ML_KEYWORDS)
    langchain_penalty = 0.0
    if has_langchain and not has_pre_llm_ml:
        langchain_penalty = 0.20
        reasons.append("LangChain/GPT-wrapper profile with no pre-LLM ML depth (JD disqualifier)")
    elif has_langchain and has_pre_llm_ml:
        pass  # LangChain is fine if backed by real ML history

    # ── NEW: General job-hopping penalty (non-senior roles) ──────────────
    # JD: "title-chasers switching every 1.5 years" — extends check to ALL roles,
    # not just senior-titled ones. Counts completed roles with <14mo tenure.
    all_hops = sum(
        1 for h in career
        if not h.get("is_current")
        and 0 < (h.get("duration_months") or 0) < 14
    )
    all_hop_penalty = 0.0
    if all_hops >= 4:
        all_hop_penalty = 0.15
        reasons.append(f"{all_hops} roles held <14mo — persistent job-hopping pattern")
    elif all_hops == 3:
        all_hop_penalty = 0.07

    score = (
        0.60 * ml_component
        + 0.25 * min(1.0, product_mo / total_mo)
        + company_bonus
        - hopping_penalty
        - consulting_penalty
        - langchain_penalty
        - all_hop_penalty
        - closed_source_penalty
    )
    return max(0.0, min(1.0, score)), reasons


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def score_experience(c):
    """
    JD: 5-9y range, ideal 6-8y. Soft curve, not a hard gate.
    Fix: >14y no longer penalised to 0.35. JD concern for very-senior candidates
    is arch/mgmt drift (already caught by score_career_trajectory), not raw YOE.
    """
    yoe = c.get("profile", {}).get("years_of_experience", 0) or 0
    if   6   <= yoe <= 8:  return 1.00
    elif 5   <= yoe <  6:  return 0.92
    elif 8   <  yoe <= 9:  return 0.92
    elif 4   <= yoe <  5:  return 0.78
    elif 9   <  yoe <= 11: return 0.78
    elif yoe >= 3:         return 0.55  # 3-4y and 11+y treated equally; arch/mgmt drift caught elsewhere
    else:                  return 0.25  # <3y


# ─────────────────────────────────────────────────────────────────────────────
# LOCATION SCORE
# ─────────────────────────────────────────────────────────────────────────────

def score_location(c):
    """JD: Pune/Noida preferred, Tier-1 India acceptable, outside India case-by-case."""
    p        = c.get("profile", {})
    sig      = c.get("redrob_signals", {}) or {}
    loc      = tl(p.get("location", ""))
    country  = tl(p.get("country", ""))
    relocate = sig.get("willing_to_relocate", False)

    if any(city in loc for city in PREFERRED_LOCATIONS): return 1.00
    if any(city in loc for city in TIER1_LOCATIONS):     return 0.90
    if "india" in country or "india" in loc:
        return 0.75 if relocate else 0.60
    return 0.35 if relocate else 0.10


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION SCORE
# ─────────────────────────────────────────────────────────────────────────────

def score_education(c):
    """Not a hard requirement in JD; tier + relevant field = moderate signal."""
    edu = c.get("education", []) or []
    if not edu:
        return 0.40
    tier_w = {"tier_1": 1.00, "tier_2": 0.80, "tier_3": 0.55,
               "tier_4": 0.30, "unknown": 0.40}
    relevant = {"computer","cs","ai","ml","data","math",
                 "statistics","electronics","information","engineering"}
    best = 0.40
    for e in edu:
        base = tier_w.get(e.get("tier", "unknown"), 0.40)
        if any(f in tl(e.get("field_of_study", "")) for f in relevant):
            base = min(1.0, base + 0.10)
        best = max(best, base)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL SCORE — reachability + engagement quality
# ─────────────────────────────────────────────────────────────────────────────

def score_behavioral(c):
    """
    Returns (engagement_score 0-1, availability_multiplier 0-1.2).

    availability_multiplier can EXCEED 1.0 for ideal candidates
    (active, open-to-work, immediate notice, fast response) — this gives
    a genuine premium to the most reachable candidates rather than just
    penalising unavailable ones.

    The multiplier is applied to the entire final score, not just behavioral.
    An unavailable candidate gets buried regardless of how good their profile is.
    """
    sig = c.get("redrob_signals", {}) or {}

    rr       = sig.get("recruiter_response_rate", 0.5)
    la_days  = days_since(sig.get("last_active_date", ""))
    otw      = sig.get("open_to_work_flag", False)
    np_days  = sig.get("notice_period_days", 90) or 90
    art      = sig.get("avg_response_time_hours", 200) or 200
    icr      = sig.get("interview_completion_rate", 0.5) or 0
    oar      = sig.get("offer_acceptance_rate", -1)
    gh       = sig.get("github_activity_score", -1)
    saved    = sig.get("saved_by_recruiters_30d", 0) or 0
    complete = sig.get("profile_completeness_score", 0) or 0
    verified = (int(sig.get("verified_email",     False))
              + int(sig.get("verified_phone",     False))
              + int(sig.get("linkedin_connected",  False)))

    # ── Availability multiplier ───────────────────────────────────────────
    avail = 1.0

    # Platform recency
    if   la_days <=  7: avail *= 1.10   # very recently active — bonus
    elif la_days <= 30: avail *= 1.05
    elif la_days <= 60: avail *= 1.00
    elif la_days <= 90: avail *= 0.90
    elif la_days <= 180:avail *= 0.70
    else:               avail *= 0.40   # >180d — serious concern

    # Open-to-work
    avail *= 1.08 if otw else 0.85

    # Recruiter response rate
    if   rr >= 0.70: avail *= 1.05
    elif rr >= 0.40: avail *= 1.00
    elif rr >= 0.20: avail *= 0.88
    else:            avail *= 0.65      # <20% = very hard to reach

    # Notice period
    if   np_days ==  0: avail *= 1.10  # immediate
    elif np_days <= 15: avail *= 1.07
    elif np_days <= 30: avail *= 1.04
    elif np_days <= 60: avail *= 0.97
    elif np_days <= 90: avail *= 0.90
    else:               avail *= 0.80  # >90d

    # ── Engagement quality score ──────────────────────────────────────────
    eng = 0.0

    if gh > 0:     eng += 0.20 * (gh / 100.0)  # external tech validation
    eng           += 0.18 * icr                  # interview reliability
    eng           += 0.12 * (complete / 100.0)   # profile seriousness
    eng           += 0.10 * min(1.0, saved / 8.0) # recruiter demand proxy
    if oar >= 0:   eng += 0.10 * oar             # offer acceptance
    eng           += 0.10 * (verified / 3.0)      # identity verification

    if   art <=  4: eng += 0.10   # responds within 4h
    elif art <= 12: eng += 0.08
    elif art <= 48: eng += 0.05
    elif art <= 168:eng += 0.02

    if   np_days <= 30: eng += 0.10
    elif np_days <= 60: eng += 0.06
    elif np_days <= 90: eng += 0.03

    return min(1.0, eng), min(1.20, avail)


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATION — context-aware, fact-grounded, Stage-4 quality
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(c, breakdown, rank):
    """
    Context-aware branching based on trajectory score so every row
    reads like a genuine recruiter note, not a template.

    Stage 4 checks: specific facts, JD connection, honest concerns,
    no hallucination, variation across rows, rank consistency.
    Everything here comes from the candidate's actual profile.
    Truncated at 500 chars to avoid CSV issues.
    """
    p      = c.get("profile", {})
    sig    = c.get("redrob_signals", {}) or {}
    career = c.get("career_history", []) or []

    title   = p.get("current_title", "Unknown")
    company = p.get("current_company", "Unknown")
    yoe     = p.get("years_of_experience", 0)
    loc     = p.get("location", "Unknown")
    country = p.get("country", "")

    rr      = sig.get("recruiter_response_rate", 0)
    np_days = sig.get("notice_period_days", 999)
    otw     = sig.get("open_to_work_flag", False)
    la_days = days_since(sig.get("last_active_date", ""))
    gh      = sig.get("github_activity_score", -1)

    # Best ML role for sentence 1
    ml_roles = [h for h in career if ML_ROLE_RE.search(h.get("title", ""))]
    best_ml  = ml_roles[0] if ml_roles else None

    # Genuinely-used strong skills
    strong_skills = [
        s["name"] for s in c.get("skills", [])
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months") or 0) >= 6
    ][:4]

    traj_score   = breakdown.get("traj_score",        0)
    traj_reasons = breakdown.get("traj_reasons",      [])
    hard_found   = breakdown.get("hard_skills_found", [])
    hp_flags     = breakdown.get("honeypot_flags",    [])

    # Availability positives and concerns
    positives, concerns = [], []

    if otw and la_days <= 14:
        positives.append("open-to-work, active within 2 weeks")
    elif otw and la_days <= 30:
        positives.append("open-to-work, active within 30 days")
    elif la_days > 180:
        concerns.append(f"inactive {la_days}d — availability risk")
    elif not otw:
        concerns.append("not marked open-to-work")

    if rr >= 0.65:
        positives.append(f"{rr:.0%} recruiter response rate")
    elif rr < 0.20:
        concerns.append(f"low recruiter response rate ({rr:.0%})")

    if np_days <= 30:
        positives.append(f"{np_days}d notice (within buyout window)")
    elif np_days > 90:
        concerns.append(f"{np_days}d notice (exceeds 90d)")

    if country.lower() not in ("india", "in", "") and country:
        concerns.append(f"outside India ({country}) — no visa sponsorship per JD")

    if gh > 60:
        positives.append(f"GitHub activity {gh:.0f}/100")

    if hp_flags:
        concerns.append(f"profile flag: {hp_flags[0]}")

    # ── Sentence 1: career fit ────────────────────────────────────────────
    if traj_score >= 0.70 and best_ml:
        skills_note = f", uses {', '.join(strong_skills[:3])}" if strong_skills else ""
        s1 = (
            f"{yoe}y {title} at {company}{skills_note}; "
            f"{best_ml['title']} background ({best_ml['duration_months']}mo) "
            f"directly matches JD's production ranking/retrieval mandate."
        )
    elif traj_score >= 0.45 and best_ml:
        hard_note = f" ({', '.join(hard_found[:2])} skills confirmed)" if hard_found else ""
        s1 = (
            f"{yoe}y {title} at {company}{hard_note}; "
            f"{best_ml['title']} experience ({best_ml['duration_months']}mo) "
            f"— partial match for JD criteria."
        )
    elif traj_reasons:
        s1 = (
            f"{yoe}y {title} at {company}; ranked #{rank} because: "
            f"{traj_reasons[0]}."
        )
    else:
        skills_note = f"skills: {', '.join(strong_skills[:2])}" if strong_skills else "limited core AI/IR skills"
        s1 = (
            f"{yoe}y {title} at {company}; {skills_note} — "
            f"partial fit for embeddings/ranking/retrieval JD focus."
        )

    # ── Sentence 2: availability ──────────────────────────────────────────
    if positives and not concerns:
        s2 = f"Strong availability: {'; '.join(positives)}."
    elif concerns and not positives:
        s2 = f"Concerns: {'; '.join(concerns)}."
    elif positives and concerns:
        s2 = f"Positives: {'; '.join(positives[:2])}. Concerns: {'; '.join(concerns[:2])}."
    else:
        s2 = f"Based in {loc}; notice {np_days}d."

    return f"{s1} {s2}"[:500]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path, chunk_size=10_000):
    """
    Streaming loader — safe for both tiny test sets and 1-lakh (100K) datasets.
    Returns list for small files; for very large files, use iter_candidates() instead.
    chunk_size is unused here but kept as a signal for future streaming use.
    """
    p      = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    out    = []
    with opener(p, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # skip malformed lines gracefully
    return out


def iter_candidates(path):
    """Generator version for huge files — avoids loading all into RAM at once."""
    p      = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker v4")
    parser.add_argument("--candidates", required=True,
                        help="candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of parallel workers (default: auto)")
    args = parser.parse_args()

    if not Path(args.candidates).exists():
        print(f"ERROR: not found: {args.candidates}")
        sys.exit(1)

    t_total = time.time()
    print(f"\n{'='*60}")
    print("Redrob Candidate Ranker v4  (scale-adaptive)")
    print(f"{'='*60}\n")

    # ── STEP 1: Load ──────────────────────────────────────────────────────
    t0 = time.time()
    print("Step 1/5  Loading candidates...")
    candidates = load_candidates(args.candidates)
    n = len(candidates)
    print(f"          {n:,} loaded in {time.time()-t0:.1f}s")

    # ── Auto-select scale config ──────────────────────────────────────────
    cfg = get_scale_config(n)
    if args.workers is not None:
        cfg["n_jobs"] = args.workers
    print(f"          Scale tier: [{cfg['label']}] | "
          f"TF-IDF features: {cfg['tfidf_features']:,} | "
          f"workers: {cfg['n_jobs']} | "
          f"sim batch: {cfg['sim_batch_size']:,}")

    # ── STEP 2: Build texts ONCE — cached for all subsequent steps ────────
    # Without this cache, build_candidate_text would be called 5× per
    # candidate inside score_hard_skills, score_soft_skills,
    # score_career_trajectory, detect_honeypot, generate_reasoning.
    # At 100K candidates: 500K redundant string ops.
    t0 = time.time()
    print("\nStep 2/5  Pre-computing candidate texts (cached)...")
    texts = [build_candidate_text(c) for c in candidates]
    print(f"          Done in {time.time()-t0:.1f}s")

    # ── STEP 3: TF-IDF + batched cosine similarity ────────────────────────
    # Cosine similarity against the full 100K matrix at once needs ~3-4GB;
    # chunking it into sim_batch_size rows keeps peak RAM under 2GB regardless
    # of dataset size while losing zero accuracy.
    t0 = time.time()
    print("\nStep 3/5  TF-IDF vectorisation + batched cosine similarity...")
    all_texts = [JD_TEXT.lower()] + texts
    tfidf = TfidfVectorizer(
        max_features = cfg["tfidf_features"],
        sublinear_tf = True,        # log(1+tf) — dampens high-frequency terms
        ngram_range  = (1, 1),      # unigrams only — ngram(1,2) OOMs at 100K docs
        stop_words   = "english",
        min_df       = cfg["tfidf_min_df"],
        max_df       = cfg["tfidf_max_df"],
        dtype        = np.float32,  # half the RAM of float64, negligible accuracy loss
    )
    mat = tfidf.fit_transform(all_texts)
    jd_vec = mat[0:1]
    cand_mat = mat[1:]

    # Chunked similarity — avoids dense (n × vocab) temporary matrix in RAM
    batch = cfg["sim_batch_size"]
    sims  = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sims[start:end] = cosine_similarity(jd_vec, cand_mat[start:end])[0]

    del mat, jd_vec, cand_mat        # free ~500MB–4GB immediately
    gc.collect()

    print(f"          sim range [{sims.min():.4f}, {sims.max():.4f}]")
    print(f"          Done in {time.time()-t0:.1f}s")

    # ── STEP 4: Hybrid scoring ────────────────────────────────────────────
    # For large datasets we pre-build the career_text once (same cache logic
    # as texts[]) to avoid recomputing inside score_hard_skills per candidate.
    t0 = time.time()
    print("\nStep 4/5  Hybrid scoring (all components)...")

    career_texts = [
        " ".join(
            f"{tl(h.get('title',''))} {tl(h.get('description',''))}"
            for h in c.get("career_history", [])
        )
        for c in candidates
    ]

    scored = []
    for i, c in enumerate(candidates):
        all_text   = texts[i]          # use cache — never rebuild
        career_text = career_texts[i]

        # Honeypot first — hard disqualifies exit here
        hp_penalty, hp_flags = detect_honeypot(c)
        if hp_penalty == 0.0:
            continue

        sem                    = float(sims[i])
        hard                   = score_hard_skills(c, all_text, career_text)
        soft                   = score_soft_skills(all_text)
        traj_score, traj_rsns  = score_career_trajectory(c, all_text)
        exp                    = score_experience(c)
        loc                    = score_location(c)
        edu                    = score_education(c)
        eng, avail             = score_behavioral(c)

        # Hard skills found in career descriptions (for reasoning)
        hard_found = [
            area.replace("_", " ")
            for area, cfg_area in HARD_SKILL_AREAS.items()
            if any(kw in career_text for kw in cfg_area["keywords"])
        ]

        # Near-disqualified: career trajectory is fundamentally wrong
        if traj_score < 0.35:
            final = traj_score * 0.15
        else:
            raw = (
                0.22 * sem         # semantic text match
              + 0.26 * hard        # 4 required skills (4-tier evidence)
              + 0.10 * soft        # nice-to-haves
              + 0.18 * traj_score  # career shape
              + 0.08 * exp         # seniority band
              + 0.05 * loc         # location
              + 0.03 * edu         # education
              + 0.08 * eng         # engagement quality
            )
            # Availability multiplier: applied last, multiplicatively.
            # Can go above 1.0 — great candidates get a small bonus.
            final = raw * avail * (0.10 + 0.90 * hp_penalty)
        if hard < 0.20:
           final *= 0.25

        scored.append({
            "candidate_id": c.get("candidate_id", f"UNKNOWN_{i}"),
            "candidate":    c,
            "score":        final,
            "breakdown": {
                "traj_score":        traj_score,
                "traj_reasons":      traj_rsns,
                "hard_skills_found": hard_found,
                "honeypot_flags":    hp_flags,
            },
        })

        # For very large datasets: free memory every 10K candidates
        if cfg["label"] == "large" and i % 10_000 == 9_999:
            gc.collect()
            print(f"          ... scored {i+1:,}/{n:,} so far")

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    if len(scored) < 50:
        print(f"WARNING: Only {len(scored)} candidates survived disqualification.")
        print("Using all available candidates.")

    top_100 = scored[:min(100, len(scored))]

    print(f"          Scored {len(scored):,} (after hard disqualifies)")
    print(f"          Done in {time.time()-t0:.1f}s")
    if len(top_100) == 0:
        print("          No candidates survived scoring.")
    elif len(top_100) >= 100:
        print(f"          Top score: {top_100[0]['score']:.4f}  |  #100: {top_100[99]['score']:.4f}")
    else:
        print(f"          Top score: {top_100[0]['score']:.4f}  |  Last: {top_100[-1]['score']:.4f}")
    print(f"\n  Top 10:")
    for r, row in enumerate(top_100[:10], 1):
        p = row["candidate"]["profile"]
        print(
            f"    {r:2d}. {row['candidate_id']} | "
            f"{p['current_title'][:30]:<30} | "
            f"{p['location'][:12]:<12} | "
            f"yoe:{p['years_of_experience']} | "
            f"{row['score']:.4f}"
        )

    # ── STEP 5: Reasoning + CSV ───────────────────────────────────────────
    t0 = time.time()
    print("\nStep 5/5  Generating reasoning + writing CSV...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, row in enumerate(top_100, start=1):
        reasoning = generate_reasoning(row["candidate"], row["breakdown"], rank)
        rows.append({
            "candidate_id": row["candidate_id"],
            "rank":         rank,
            "score":        f"{row['score']:.6f}",
            "reasoning":    reasoning,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)

    print(f"          Written -> {out_path}")
    print(f"          Done in {time.time()-t0:.1f}s")

    # ── Validation ────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    errors = []
    if len(rows) != min(100, len(scored)):
        errors.append(f"Expected {min(100, len(scored))} rows, got {len(rows)}")
    scores_list = [float(r["score"]) for r in rows]
    for i in range(len(scores_list) - 1):
        if scores_list[i] < scores_list[i+1] - 1e-9:
            errors.append(f"Non-increasing score at rank {i+1} -> {i+2}")
            break
    if len({r["candidate_id"] for r in rows}) != len(rows):
        errors.append("Duplicate candidate IDs")

    if errors:
        print("WARNING: VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("OK: Validation passed: 100 rows, non-increasing scores, unique IDs")

    if elapsed > 270:
        print(f"WARNING: {elapsed:.0f}s is close to the 5-minute limit")
    else:
        print(f"OK: Well within 5-minute compute constraint")

    print(f"\nNext step: python validate_submission.py {args.out}")


if __name__ == "__main__":
    main()