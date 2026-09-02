"""Общие утилиты: конфиг, корпус, промпты, LLM-клиент, каталог прогонов.

Все модули пайплайна читают один config.yaml. Пути в конфиге относительны
каталогу, в котором лежит сам config.yaml.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from . import __version__

CLASSES: List[str] = ["COMP", "IIT", "PRED", "GWT", "ENACT", "PAN", "EMERG"]
ALL_CLASSES: List[str] = CLASSES + ["UND"]
LEVELS = {"explicit", "scientific_metaphor", "meta_metaphor"}
HP_STANCES = {"denial", "attribution", "open_question", "reframing"}
HP_SUBJECTS = {"машина", "материя", "человек", "неясно"}


# --------------------------------------------------------------------------- #
# Конфиг и пути
# --------------------------------------------------------------------------- #

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_root"] = str(path.parent)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(cfg: Dict[str, Any], p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return Path(cfg.get("_root", ".")) / p


def reports_dir(cfg: Dict[str, Any]) -> Path:
    d = resolve(cfg, cfg["paths"]["reports_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(cfg: Dict[str, Any]) -> Path:
    d = resolve(cfg, cfg["paths"]["runs_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def code_version(root: Optional[str | Path] = None) -> str:
    """Версия кода: git-коммит (+dirty) или версия пакета, если git недоступен."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return f"{out}{'-dirty' if dirty else ''} (pipeline {__version__})"
    except Exception:  # noqa: BLE001
        return f"pipeline {__version__}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Корпус
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)


def clean_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).replace("_x000D_", " ").replace("\r", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def load_corpus(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Корпус в единой форме: id, source, title, author, text, n_words.

    id — позиция строки в исходном xlsx (0..N-1); она же используется во всех
    прогонах, в legacy-результатах (поле index) и в таблице ручной разметки.
    """
    c = cfg["corpus"]
    df = pd.read_excel(resolve(cfg, c["path"]), sheet_name=c.get("sheet", 0))
    text_col = c.get("text_column", "descr_clean")
    fb_col = c.get("fallback_text_column", "descr")
    texts = []
    for _, row in df.iterrows():
        t = clean_text(row.get(text_col)) if text_col in df.columns else ""
        if not t and fb_col in df.columns:
            t = clean_text(row.get(fb_col))
        texts.append(t)
    out = pd.DataFrame({
        "id": range(len(df)),
        "source": df[c.get("source_column", "__source")].astype(str) if c.get("source_column", "__source") in df.columns else "unknown",
        "title": df[c.get("title_column", "title")].astype(str) if c.get("title_column", "title") in df.columns else [f"Item_{i}" for i in range(len(df))],
        "author": df[c.get("author_column", "descr_author")] if c.get("author_column", "descr_author") in df.columns else None,
        "text": texts,
    })
    out["n_words"] = out["text"].map(count_words)
    return out


# --------------------------------------------------------------------------- #
# Промпты
# --------------------------------------------------------------------------- #

@dataclass
class Prompt:
    name: str
    system: str
    user_template: str
    rendered: str
    hash: str
    codebook_hash: str

    def user(self, text: str) -> str:
        return self.user_template.replace("{text}", text)


_SECTION_RE = re.compile(r"<!--\s*section:(?P<name>[\w-]+)\s*-->(?P<body>.*?)<!--\s*/section:(?P=name)\s*-->", re.S)


def codebook_sections(codebook_text: str) -> Dict[str, str]:
    return {m.group("name"): m.group("body").strip() for m in _SECTION_RE.finditer(codebook_text)}


def load_codebook(cfg: Dict[str, Any]) -> str:
    p = resolve(cfg, cfg["paths"]["prompts_dir"]) / cfg["prompts"]["codebook"]
    return p.read_text(encoding="utf-8")


def render_prompt_text(raw: str, codebook_text: str) -> str:
    sections = codebook_sections(codebook_text)

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in sections:
            raise KeyError(f"В кодбуке нет раздела section:{name}")
        return sections[name]

    return re.sub(r"\{\{\s*codebook:([\w-]+)\s*\}\}", sub, raw)


def _block(text: str, tag: str) -> str:
    m = re.search(rf"<!--\s*{tag}\s*-->(.*?)<!--\s*/{tag}\s*-->", text, re.S)
    if not m:
        raise ValueError(f"В промпте нет блока <!-- {tag} -->")
    return m.group(1).strip()


def load_prompt(cfg: Dict[str, Any], kind: str) -> Prompt:
    """kind: 'theory' | 'hardproblem'. Возвращает промпт с подставленным кодбуком и хешем."""
    pdir = resolve(cfg, cfg["paths"]["prompts_dir"])
    name = cfg["prompts"][kind]
    raw = (pdir / name).read_text(encoding="utf-8")
    codebook = load_codebook(cfg)
    rendered = render_prompt_text(raw, codebook)
    return Prompt(
        name=name,
        system=_block(rendered, "system"),
        user_template=_block(rendered, "user"),
        rendered=rendered,
        hash=sha256(rendered)[:16],
        codebook_hash=sha256(codebook)[:16],
    )


# --------------------------------------------------------------------------- #
# JSON из ответа модели
# --------------------------------------------------------------------------- #

def extract_json(content: str) -> Optional[Dict[str, Any]]:
    """Достаёт первый сбалансированный JSON-объект из ответа (с учётом ограждений и преамбулы)."""
    if not content:
        return None
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break
    try:
        return json.loads(s[start:])
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# LLM-клиент с кешем сырых ответов
# --------------------------------------------------------------------------- #

@dataclass
class LLMResponse:
    content: str
    raw: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    error: Optional[str] = None


class LLMClient:
    """Chat-completions клиент (OpenRouter-совместимый) с файловым кешем.

    Кеш лежит в каталоге прогона: повторный запуск с тем же промптом и текстом
    не обращается к API, так что прерванный прогон можно продолжить.
    Ключ кеша включает model, system, user, temperature, seed и repeat.
    """

    def __init__(self, cfg: Dict[str, Any], model_id: str, cache_dir: Optional[Path] = None):
        self.cfg = cfg
        llm = cfg.get("llm", {})
        self.provider = llm.get("provider", "openrouter")
        self.model = model_id
        self.base_url = llm.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        self.temperature = float(llm.get("temperature", 0.0))
        self.max_tokens = int(llm.get("max_tokens", 4000))
        self.timeout = int(llm.get("timeout", 120))
        self.max_retries = int(llm.get("max_retries", 4))
        self.delay = float(llm.get("request_delay", 1.0))
        self.json_mode = bool(llm.get("response_format_json", True))
        self.send_seed = bool(llm.get("send_seed", True))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = None
        if self.provider == "openrouter":
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(cfg.get("_root", ".")) / ".env")
            except ImportError:
                pass
            env = llm.get("api_key_env", "OPENROUTER_API_KEY")
            self._api_key = os.getenv(env)
            if not self._api_key:
                raise RuntimeError(
                    f"Не задан {env}. Положите ключ в .env или переменную окружения, "
                    "либо используйте llm.provider: mock для проверки пайплайна без API."
                )
        elif self.provider == "mock":
            from .mock_llm import MockLLM
            self._mock = MockLLM()
        else:
            raise ValueError(f"Неизвестный llm.provider: {self.provider}")

    def _cache_key(self, system: str, user: str, seed: int, repeat: int) -> str:
        return sha256("|".join([self.model, system, user, str(self.temperature), str(seed), str(repeat)]))

    def complete(self, system: str, user: str, seed: int = 0, repeat: int = 1) -> LLMResponse:
        key = self._cache_key(system, user, seed, repeat)
        if self.cache_dir:
            cp = self.cache_dir / f"{key}.json"
            if cp.exists():
                d = json.loads(cp.read_text(encoding="utf-8"))
                return LLMResponse(content=d.get("content", ""), raw=d.get("raw", {}), cached=True)
        if self.provider == "mock":
            content = self._mock.complete(system, user)
            resp = LLMResponse(content=content, raw={"provider": "mock"})
        else:
            resp = self._call_api(system, user, seed)
        if self.cache_dir and resp.error is None:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps({"model": self.model, "seed": seed, "repeat": repeat,
                            "content": resp.content, "raw": resp.raw}, ensure_ascii=False, indent=1),
                encoding="utf-8")
        return resp

    def _call_api(self, system: str, user: str, seed: int) -> LLMResponse:
        import requests

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("HTTP_REFERER", "https://github.com/stainephone-gif/AI-art"),
            "X-Title": os.getenv("X_TITLE", "AI-art consciousness classifier v3"),
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.send_seed:
            payload["seed"] = int(seed)
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    time.sleep(self.delay)
                    return LLMResponse(content=content, raw={
                        "id": data.get("id"), "model": data.get("model"),
                        "usage": data.get("usage"), "provider": data.get("provider"),
                        "system_fingerprint": data.get("system_fingerprint"),
                    })
                if r.status_code == 429 or r.status_code >= 500:
                    wait = 5 * (2 ** attempt)
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    print(f"    {last_err}; ждём {wait}s")
                    time.sleep(wait)
                    continue
                if r.status_code == 400 and self.json_mode and "response_format" in r.text:
                    # провайдер не поддерживает json_object — повторяем без него
                    payload.pop("response_format", None)
                    self.json_mode = False
                    continue
                return LLMResponse(content="", error=f"HTTP {r.status_code}: {r.text[:300]}")
            except requests.RequestException as e:  # noqa: PERF203
                last_err = str(e)
                time.sleep(2 ** attempt)
        return LLMResponse(content="", error=last_err or "max retries exceeded")


# --------------------------------------------------------------------------- #
# Каталог прогона
# --------------------------------------------------------------------------- #

def model_entries(cfg: Dict[str, Any], only: Optional[List[str]] = None) -> List[Dict[str, str]]:
    entries = []
    for m in cfg.get("models", []):
        if isinstance(m, str):
            m = {"id": m}
        m = dict(m)
        m.setdefault("slug", re.sub(r"[^\w.-]+", "_", m["id"].split("/")[-1]))
        if only and m["id"] not in only and m["slug"] not in only:
            continue
        entries.append(m)
    return entries


def run_directory(cfg: Dict[str, Any], slug: str, run_name: Optional[str] = None, date: Optional[str] = None) -> Path:
    """runs/<slug>_<YYYYMMDD>/ (или runs/<run_name>/)."""
    base = runs_dir(cfg)
    name = run_name or f"{slug}_{date or datetime.now().strftime('%Y%m%d')}"
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_meta(run_dir: Path, cfg: Dict[str, Any], model: Dict[str, str], prompts: Dict[str, Prompt], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """meta.json прогона: модель, хеши промптов, дата, версия кода, снимок конфига."""
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    meta.update({
        "model": model["id"],
        "slug": model["slug"],
        "provider": cfg.get("llm", {}).get("provider"),
        "temperature": cfg.get("llm", {}).get("temperature", 0.0),
        "seed": cfg.get("seed"),
        "updated_at": now_iso(),
        "code_version": code_version(cfg.get("_root")),
        "pipeline_version": __version__,
        "config_snapshot": snapshot,
    })
    meta.setdefault("created_at", meta["updated_at"])
    meta.setdefault("prompts", {})
    for kind, p in prompts.items():
        meta["prompts"][kind] = {"file": p.name, "hash": p.hash, "codebook_hash": p.codebook_hash}
    if extra:
        meta.update(extra)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def list_runs(cfg: Dict[str, Any], kind: str = "theory") -> List[Dict[str, Any]]:
    """Все v3-прогоны: [{run_dir, meta, files: {repeat: path}}], без legacy-каталогов."""
    out = []
    base = runs_dir(cfg)
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith("legacy"):
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        files = {}
        for f in sorted(d.glob(f"{kind}_r*.json")):
            m = re.match(rf"{kind}_r(\d+)\.json", f.name)
            if m:
                files[int(m.group(1))] = f
        if files:
            out.append({"run_dir": d, "meta": json.loads(meta_path.read_text(encoding="utf-8")), "files": files})
    return out


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj: Any, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
