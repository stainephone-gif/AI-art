"""Общий цикл прогона: модели × повторы × тексты, с кешем, докачкой и метаданными."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .common import (LLMClient, Prompt, code_version, dump_json, extract_json, load_config,
                     load_corpus, load_json, load_prompt, model_entries, now_iso, run_directory,
                     write_meta)

ProcessFn = Callable[[Optional[Dict[str, Any]], pd.Series, Dict[str, Any]], Dict[str, Any]]


def build_argparser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", action="append", help="id или slug модели из конфига (можно несколько); по умолчанию все")
    ap.add_argument("--repeats", type=int, default=None, help="число повторов (по умолчанию из конфига)")
    ap.add_argument("--limit", type=int, default=0, help="только первые N текстов (для проверки)")
    ap.add_argument("--ids", type=str, default=None, help="список id через запятую")
    ap.add_argument("--run-name", type=str, default=None, help="имя каталога прогона вместо <slug>_<date>")
    ap.add_argument("--date", type=str, default=None, help="дата в имени каталога (YYYYMMDD), по умолчанию сегодня")
    ap.add_argument("--force", action="store_true", help="переклассифицировать уже готовые тексты (кеш ответов при этом не сбрасывается)")
    ap.add_argument("--no-cache", action="store_true", help="не использовать кеш сырых ответов")
    return ap


def run_kind(kind: str, process: ProcessFn, args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg = load_config(args.config)
    corpus = load_corpus(cfg)
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",") if x.strip()}
        corpus = corpus[corpus["id"].isin(wanted)]
    if args.limit:
        corpus = corpus.head(args.limit)
    prompt: Prompt = load_prompt(cfg, kind)
    repeats = args.repeats or int(cfg.get("repeats", 2))
    seed = int(cfg.get("seed", 0))
    version = code_version(cfg.get("_root"))
    written: List[Dict[str, Any]] = []

    for model in model_entries(cfg, args.model):
        rdir = run_directory(cfg, model["slug"], args.run_name, args.date)
        client = LLMClient(cfg, model["id"], cache_dir=None if args.no_cache else rdir / "cache")
        write_meta(rdir, cfg, model, {kind: prompt})
        for rep in range(1, repeats + 1):
            out_path = rdir / f"{kind}_r{rep}.json"
            existing = {r["id"]: r for r in load_json(out_path)} if out_path.exists() else {}
            results: Dict[int, Dict[str, Any]] = dict(existing)
            todo = [row for _, row in corpus.iterrows()
                    if args.force or row["id"] not in existing or existing[row["id"]].get("status") != "ok"]
            print(f"[{kind}] {model['id']} повтор {rep}/{repeats}: {len(todo)} текстов "
                  f"(готово {len(corpus) - len(todo)}), prompt {prompt.name}#{prompt.hash}")
            for n, row in enumerate(todo, 1):
                print(f"  {n}/{len(todo)} id={row['id']} {str(row['title'])[:60]}")
                resp = client.complete(prompt.system, prompt.user(row["text"]), seed=seed + rep - 1, repeat=rep)
                rec: Dict[str, Any] = {
                    "id": int(row["id"]), "title": row["title"], "source": row["source"],
                    "n_words": int(row["n_words"]), "model": model["id"], "slug": model["slug"],
                    "repeat": rep, "prompt": prompt.name, "prompt_hash": prompt.hash,
                    "codebook_hash": prompt.codebook_hash, "code_version": version,
                    "timestamp": now_iso(), "cached": resp.cached,
                }
                data = None
                if resp.error:
                    rec["status"], rec["error"] = "api_error", resp.error
                else:
                    data = extract_json(resp.content)
                    if data is None:
                        rec["status"] = "parse_error"
                        rec["raw_content"] = resp.content[:2000]
                    else:
                        rec["status"] = "ok"
                rec.update(process(data, row, cfg))
                results[int(row["id"])] = rec
                if n % 10 == 0:
                    dump_json([results[k] for k in sorted(results)], out_path)
            ordered = [results[k] for k in sorted(results)]
            dump_json(ordered, out_path)
            written.extend(ordered)
            ok = sum(1 for r in ordered if r.get("status") == "ok")
            print(f"  -> {out_path} ({ok}/{len(ordered)} ok)")
        write_meta(rdir, cfg, model, {kind: prompt}, extra={f"{kind}_finished_at": now_iso(), f"{kind}_repeats": repeats})
    return written
