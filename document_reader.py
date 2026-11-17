#!/usr/bin/env python3
"""
document_reader.py  (single file: engine + web dashboard)
==========================================================

Run the dashboard :  streamlit run document_reader.py
Run the CLI       :  python document_reader.py search "valve"

Read technical documents, make them searchable, and summarize them.

Features
--------
* Reads: PDF, DOCX, PPTX, TXT, MD, RST, HTML, CSV, JSON, LOG, code files
* Stores text in a local SQLite database with full-text search (FTS5, BM25 ranking)
* Incremental indexing (unchanged files are skipped)
* Summaries:
    - extractive (offline, no dependencies, default)
    - AI summary using Claude (optional, needs ANTHROPIC_API_KEY)
* Optional "focus" so a summary is built around a topic/question

Install (only what you need)
----------------------------
    pip install pypdf python-docx python-pptx beautifulsoup4 anthropic reportlab openpyxl streamlit pandas

Usage
-----
    python document_reader.py index  ./docs              # index a folder (recursive)
    python document_reader.py list                       # show indexed documents
    python document_reader.py search "pressure relief valve" -n 10
    python document_reader.py search "torque spec" --any # match ANY word
    python document_reader.py summarize manual.pdf -s 8
    python document_reader.py summarize manual.pdf --focus "safety requirements"
    python document_reader.py summarize manual.pdf --ai  # Claude summary
    python document_reader.py show manual.pdf --page 3   # print text of a page
    python document_reader.py remove manual.pdf
    python document_reader.py lists spec.pdf [--only motor,valve]   # equipment/motor/valve/instrument/line lists
    python document_reader.py inputs spec.pdf [--ai]      # design inputs table -> xlsx/pdf/csv
    python document_reader.py scope spec.pdf [--ai]       # sorted scope lists -> xlsx/pdf/csv
    python document_reader.py report manual.pdf -k "torque, safety" -o out.pdf --csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Configuration (edit freely)
# --------------------------------------------------------------------------- #
DB_PATH = os.environ.get("DOCREADER_DB", "document_index.db")
CHUNK_CHARS = 1200          # approx. size of each searchable chunk
CHUNK_OVERLAP = 150         # overlap between chunks so phrases aren't cut
AI_MODEL = os.environ.get("DOCREADER_MODEL", "claude-sonnet-5-5")
AI_CHUNK_CHARS = 30000      # size of text pieces sent to the AI per request

TEXT_EXTS = {
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".xml", ".ini", ".cfg", ".toml", ".py", ".js", ".ts", ".java", ".c",
    ".cpp", ".h", ".cs", ".go", ".rs", ".sql", ".sh", ".tex",
}
SUPPORTED_EXTS = TEXT_EXTS | {".pdf", ".docx", ".pptx", ".html", ".htm"}


def _pip_hint(pkg: str) -> str:
    """Exact install command for the Python that is running this script."""
    return f'{pkg} is not installed in this Python.  Run:  "{sys.executable}" -m pip install {pkg}'


# --------------------------------------------------------------------------- #
# 1. Text extraction  ->  list of (page_or_section_number, text)
# --------------------------------------------------------------------------- #
def _read_text_file(path: Path) -> list[tuple[int, str]]:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return [(1, path.read_text(encoding=enc))]
        except (UnicodeDecodeError, UnicodeError):
            continue
    return [(1, path.read_bytes().decode("utf-8", errors="ignore"))]


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(_pip_hint("pypdf"))
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise RuntimeError("PDF is password protected")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages.append((i, page.extract_text() or ""))
        except Exception:
            pages.append((i, ""))
    if not any(t.strip() for _, t in pages):
        print(f"  ! {path.name}: no text found (scanned PDF? needs OCR)", file=sys.stderr)
    return pages


def _read_docx(path: Path) -> list[tuple[int, str]]:
    try:
        import docx
    except ImportError:
        raise RuntimeError(_pip_hint("python-docx"))
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:                      # technical docs have lots of tables
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            parts.append(" | ".join(cells))
    return [(1, "\n".join(parts))]


def _read_pptx(path: Path) -> list[tuple[int, str]]:
    try:
        from pptx import Presentation
    except ImportError:
        raise RuntimeError(_pip_hint("python-pptx"))
    prs = Presentation(str(path))
    slides = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = [sh.text_frame.text for sh in slide.shapes
                 if getattr(sh, "has_text_frame", False) and sh.has_text_frame]
        slides.append((i, "\n".join(texts)))
    return slides


def _read_html(path: Path) -> list[tuple[int, str]]:
    raw = _read_text_file(path)[0][1]
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    except ImportError:                          # fallback: crude tag stripper
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return [(1, text)]


def extract_pages(path: Path) -> list[tuple[int, str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext == ".pptx":
        return _read_pptx(path)
    if ext in (".html", ".htm"):
        return _read_html(path)
    if ext in TEXT_EXTS:
        return _read_text_file(path)
    raise RuntimeError(f"Unsupported file type: {ext}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\x7f", "•")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"-\n(\w)", r"\1", text)       # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into ~size char chunks, preferring paragraph/sentence boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# --------------------------------------------------------------------------- #
# 3. Database (SQLite + FTS5)
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    path      TEXT UNIQUE NOT NULL,
    name      TEXT NOT NULL,
    sha256    TEXT NOT NULL,
    n_pages   INTEGER NOT NULL,
    n_chars   INTEGER NOT NULL,
    indexed   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id INTEGER NOT NULL,
    page   INTEGER NOT NULL,
    text   TEXT NOT NULL,
    PRIMARY KEY (doc_id, page)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    content,
    doc_id UNINDEXED,
    page   UNINDEXED,
    tokenize = 'porter unicode61'
);
"""


@dataclass
class Hit:
    doc_name: str
    doc_path: str
    page: int
    snippet: str
    score: float


class DocumentStore:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.executescript(SCHEMA)
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Your SQLite build lacks FTS5 support: {e}")

    # ---- indexing ---------------------------------------------------------
    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def add_file(self, path: Path, force: bool = False) -> str:
        path = path.resolve()
        digest = self._sha256(path)
        row = self.conn.execute("SELECT id, sha256 FROM docs WHERE path=?", (str(path),)).fetchone()
        if row and row["sha256"] == digest and not force:
            return "skipped (unchanged)"
        if row:
            self._delete_doc(row["id"])

        pages = [(n, clean_text(t)) for n, t in extract_pages(path)]
        total_chars = sum(len(t) for _, t in pages)
        cur = self.conn.execute(
            "INSERT INTO docs(path, name, sha256, n_pages, n_chars) VALUES (?,?,?,?,?)",
            (str(path), path.name, digest, len(pages), total_chars),
        )
        doc_id = cur.lastrowid
        for page_no, text in pages:
            if not text:
                continue
            self.conn.execute("INSERT INTO pages VALUES (?,?,?)", (doc_id, page_no, text))
            for piece in chunk_text(text):
                self.conn.execute(
                    "INSERT INTO chunks(content, doc_id, page) VALUES (?,?,?)",
                    (piece, doc_id, page_no),
                )
        self.conn.commit()
        return f"indexed ({len(pages)} pages/sections, {total_chars:,} chars)"

    def add_path(self, target: Path, force: bool = False) -> None:
        files = self._collect(target)
        if not files:
            print("No supported files found.")
            return
        for f in files:
            try:
                print(f"- {f.name}: {self.add_file(f, force)}")
            except Exception as e:
                print(f"- {f.name}: FAILED ({e})", file=sys.stderr)

    @staticmethod
    def _collect(target: Path) -> list[Path]:
        if target.is_file():
            return [target] if target.suffix.lower() in SUPPORTED_EXTS else []
        return sorted(p for p in target.rglob("*")
                      if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                      and not any(part.startswith(".") for part in p.parts))

    def _delete_doc(self, doc_id: int) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM docs WHERE id=?", (doc_id,))

    def remove(self, name_or_path: str) -> bool:
        doc = self.find_doc(name_or_path)
        if not doc:
            return False
        self._delete_doc(doc["id"])
        self.conn.commit()
        return True

    # ---- lookup -----------------------------------------------------------
    def list_docs(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, name, path, n_pages, n_chars, indexed FROM docs ORDER BY name").fetchall()

    def find_doc(self, name_or_path: str) -> sqlite3.Row | None:
        q = self.conn.execute
        row = q("SELECT * FROM docs WHERE path=? OR name=?",
                (str(Path(name_or_path).resolve()), name_or_path)).fetchone()
        if row:
            return row
        rows = q("SELECT * FROM docs WHERE name LIKE ?", (f"%{name_or_path}%",)).fetchall()
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            print("Multiple matches, be more specific:", file=sys.stderr)
            for r in rows:
                print("  ", r["name"], file=sys.stderr)
        return None

    def doc_text(self, doc_id: int) -> str:
        rows = self.conn.execute(
            "SELECT page, text FROM pages WHERE doc_id=? ORDER BY page", (doc_id,)).fetchall()
        return "\n\n".join(r["text"] for r in rows)

    def page_text(self, doc_id: int, page: int) -> str | None:
        r = self.conn.execute(
            "SELECT text FROM pages WHERE doc_id=? AND page=?", (doc_id, page)).fetchone()
        return r["text"] if r else None

    # ---- search -----------------------------------------------------------
    @staticmethod
    def _build_query(query: str, match_any: bool) -> str:
        """Turn user text into a safe FTS5 query. Keep "quoted phrases" intact."""
        parts = re.findall(r'"([^"]+)"|(\S+)', query)
        terms = []
        for phrase, word in parts:
            if phrase:
                terms.append('"' + phrase.replace('"', "") + '"')
            else:
                w = re.sub(r"[^\w\-\*]", "", word)
                if not w:
                    continue
                prefix = w.endswith("*")
                w = w.rstrip("*")
                if w:
                    terms.append(f'"{w}"' + ("*" if prefix else ""))
        return (" OR " if match_any else " AND ").join(terms)

    def search(self, query: str, limit: int = 10, match_any: bool = False,
               doc_filter: str | None = None) -> list[Hit]:
        fts = self._build_query(query, match_any)
        if not fts:
            return []
        sql = """
            SELECT d.name, d.path, c.page,
                   snippet(chunks, 0, '[[', ']]', ' … ', 28) AS snip,
                   bm25(chunks) AS score
            FROM chunks c JOIN docs d ON d.id = c.doc_id
            WHERE chunks MATCH ?
        """
        args: list = [fts]
        if doc_filter:
            sql += " AND d.name LIKE ?"
            args.append(f"%{doc_filter}%")
        sql += " ORDER BY score LIMIT ?"
        args.append(limit)
        try:
            rows = self.conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError as e:
            print(f"Search error: {e}", file=sys.stderr)
            return []
        return [Hit(r["name"], r["path"], r["page"],
                    r["snip"].replace("\n", " "), r["score"]) for r in rows]


# --------------------------------------------------------------------------- #
# 4. Summarization
# --------------------------------------------------------------------------- #
STOPWORDS = set("""
a about above after again all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few
for from further had has have having he her here hers him his how i if in into is
it its just me more most my no nor not now of off on once only or other our out over
own same she should so some such than that the their them then there these they this
those through to too under until up very was we were what when where which while who
whom why will with would you your shall may must per via etc eg ie
""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/\.]{1,}")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\"'])|\n{2,}|\n(?=\s*(?:[-•*]|\d+[.)])\s)")


def _words(text: str) -> list[str]:
    return [w.lower().strip(".-_/") for w in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    out = []
    for s in SENT_SPLIT.split(text):
        s = " ".join(s.split())
        if 25 <= len(s) <= 600:               # drop fragments and giant blobs
            out.append(s)
    return out


def extractive_summary(text: str, n_sentences: int = 7, focus: str | None = None) -> str:
    """Offline summary: score sentences by word importance (+ boost for focus terms)."""
    sentences = split_sentences(text)
    if not sentences:
        return text[:800].strip()
    if len(sentences) <= n_sentences:
        return "\n".join(f"• {s}" for s in sentences)

    freq = Counter(w for w in _words(text) if w not in STOPWORDS and len(w) > 2)
    if not freq:
        return "\n".join(f"• {s}" for s in sentences[:n_sentences])
    top = max(freq.values())
    focus_terms = {w for w in _words(focus) if w not in STOPWORDS} if focus else set()

    scored = []
    for idx, s in enumerate(sentences):
        ws = [w for w in _words(s) if w not in STOPWORDS and len(w) > 2]
        if not ws:
            continue
        score = sum(freq[w] / top for w in ws) / (len(ws) ** 0.6)   # favor dense, not just long
        if focus_terms:
            hits = sum(1 for w in ws if w in focus_terms)
            score += 2.0 * hits / max(len(focus_terms), 1)
            if hits == 0:
                score *= 0.4
        if idx < 3:                                                 # intro sentences matter
            score *= 1.15
        scored.append((score, idx, s))

    best = sorted(scored, reverse=True)[:n_sentences]
    best.sort(key=lambda t: t[1])                                    # restore original order
    return "\n".join(f"• {s}" for _, _, s in best)


def keywords(text: str, n: int = 12) -> list[str]:
    freq = Counter(w for w in _words(text) if w not in STOPWORDS and len(w) > 3)
    return [w for w, _ in freq.most_common(n)]


def ai_summary(text: str, focus: str | None = None, length: str = "about 10 bullet points") -> str:
    """Summarize with Claude. Long docs are summarized in pieces, then combined."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(_pip_hint("anthropic"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable first.")
    client = anthropic.Anthropic()

    def ask(prompt: str) -> str:
        msg = client.messages.create(
            model=AI_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    focus_line = f"Focus especially on: {focus}.\n" if focus else ""
    pieces = [text[i:i + AI_CHUNK_CHARS] for i in range(0, len(text), AI_CHUNK_CHARS)]

    if len(pieces) == 1:
        return ask(f"Summarize this technical document in {length}. {focus_line}"
                   f"Keep exact figures, part numbers, units and requirements.\n\n{text}")

    partials = []
    for i, p in enumerate(pieces, 1):
        print(f"  summarizing part {i}/{len(pieces)}...", file=sys.stderr)
        partials.append(ask(f"Summarize part {i} of {len(pieces)} of a technical document. "
                            f"{focus_line}Keep exact figures and requirements.\n\n{p}"))
    combined = "\n\n".join(partials)
    return ask(f"Below are partial summaries of one technical document. Merge them into a "
               f"single coherent summary in {length}. {focus_line}\n\n{combined}")


# --------------------------------------------------------------------------- #
# 4b. Keyword extraction  +  PDF report
# --------------------------------------------------------------------------- #
@dataclass
class Match:
    keyword: str
    doc: str
    page: int
    text: str


def parse_keywords(raw: str) -> list[str]:
    """'torque, safety; "pressure relief valve"'  ->  ['torque', 'safety', 'pressure relief valve']"""
    out, seen = [], set()
    for p in re.split(r"[,\n;]+", raw or ""):
        p = " ".join(p.strip().strip('"').split())
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def keyword_regex(kw: str) -> re.Pattern:
    """Case-insensitive, whole-word, tolerant of plurals and line-wrapped phrases."""
    body = r"\s+".join(re.escape(t) for t in kw.split())
    left = r"\b" if kw[0].isalnum() else ""
    right = r"(?:e?s|ed|ing)?\b" if kw[-1].isalnum() else ""
    return re.compile(left + body + right, re.I)


def _sentences(text: str) -> list[str]:
    out = []
    for s in SENT_SPLIT.split(text):
        if not s or not s.strip():
            continue
        parts = s.split("\n") if len(s) > 500 else [s]      # table rows / unpunctuated lines
        out += [" ".join(p.split()) for p in parts if p.strip()]
    return out


def find_keyword_matches(store: "DocumentStore", keywords: list[str],
                         doc_names: list[str] | None = None,
                         context: int = 1, max_per_keyword: int = 25) -> list[Match]:
    """Find every passage containing each keyword (plus `context` sentences around it)."""
    docs = store.list_docs()
    if doc_names:
        docs = [d for d in docs if d["name"] in set(doc_names)]
    regs = {k: keyword_regex(k) for k in keywords}
    counts: Counter = Counter()
    matches: list[Match] = []
    for d in docs:
        rows = store.conn.execute(
            "SELECT page, text FROM pages WHERE doc_id=? ORDER BY page", (d["id"],)).fetchall()
        for r in rows:
            sents = _sentences(r["text"])
            for kw, rx in regs.items():
                i, last_hi = 0, 0
                while i < len(sents):
                    if counts[kw] >= max_per_keyword:
                        break
                    if rx.search(sents[i]):
                        lo, hi = max(last_hi, i - context), min(len(sents), i + context + 1)
                        last_hi = hi
                        matches.append(Match(kw, d["name"], r["page"], " ".join(sents[lo:hi])))
                        counts[kw] += 1
                        i = hi                      # skip past context: no duplicate passages
                    else:
                        i += 1
    order = {k: n for n, k in enumerate(keywords)}
    matches.sort(key=lambda m: order[m.keyword])    # group by keyword, keep doc/page order
    return matches


def matches_to_rows(matches: list[Match]) -> list[dict]:
    return [{"Keyword": m.keyword, "Document": m.doc, "Page": m.page, "Text": m.text}
            for m in matches]


def matches_to_csv(matches: list[Match]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["Keyword", "Document", "Page", "Text"])
    w.writeheader()
    w.writerows(matches_to_rows(matches))
    return buf.getvalue().encode("utf-8-sig")       # utf-8-sig so Excel opens it correctly


def _register_pdf_font():
    """Use a Unicode TrueType font if one exists (° µ Ω ≥ … in technical docs)."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    dj = "/usr/share/fonts/truetype/dejavu/"
    candidates = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/ariali.ttf"),
        (dj + "DejaVuSans.ttf", dj + "DejaVuSans-Bold.ttf", dj + "DejaVuSans-Oblique.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial Italic.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
    ]
    for reg, bold, ital in candidates:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("DocFont", reg))
                pdfmetrics.registerFont(TTFont("DocFont-Bold", bold))
                pdfmetrics.registerFont(TTFont("DocFont-Italic", ital if os.path.exists(ital) else reg))
                pdfmetrics.registerFontFamily("DocFont", normal="DocFont", bold="DocFont-Bold",
                                              italic="DocFont-Italic", boldItalic="DocFont-Bold")
                return "DocFont", "DocFont-Bold", True
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold", False


def build_pdf_report(matches: list[Match], keywords: list[str], doc_names: list[str],
                     target, title: str = "Keyword Extraction Report",
                     key_points: bool = True) -> None:
    """Write a PDF to `target` (a file path or a BytesIO)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (HRFlowable, KeepTogether, Paragraph,
                                        SimpleDocTemplate, Spacer, Table, TableStyle)
    except ImportError:
        raise RuntimeError(_pip_hint("reportlab"))

    base, bold, unicode_ok = _register_pdf_font()

    def clean(s: str) -> str:
        s = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", s)
        return s if unicode_ok else s.encode("latin-1", "replace").decode("latin-1")

    def esc(s: str) -> str:
        return html.escape(clean(s), quote=False)

    regs = {k: keyword_regex(k) for k in keywords}

    def highlight(text: str, rx: re.Pattern) -> str:
        text, out, pos = clean(text), [], 0
        for m in rx.finditer(text):
            out.append(html.escape(text[pos:m.start()], quote=False))
            out.append('<font color="#B45309"><b>%s</b></font>' % html.escape(m.group(), quote=False))
            pos = m.end()
        out.append(html.escape(text[pos:], quote=False))
        return "".join(out)

    ss = getSampleStyleSheet()
    grey, blue = colors.HexColor("#6B7280"), colors.HexColor("#0F4C81")
    s_title = ParagraphStyle("T", parent=ss["Title"], fontName=bold, fontSize=20, leading=24,
                             alignment=0, textColor=colors.HexColor("#1F2937"))
    s_meta = ParagraphStyle("M", parent=ss["Normal"], fontName=base, fontSize=9, leading=13, textColor=grey)
    s_h = ParagraphStyle("H", parent=ss["Heading2"], fontName=bold, fontSize=14, leading=18,
                         spaceBefore=16, spaceAfter=4, textColor=blue)
    s_body = ParagraphStyle("B", parent=ss["Normal"], fontName=base, fontSize=10, leading=14, spaceAfter=3)
    s_cell = ParagraphStyle("C", parent=s_body, fontSize=9, leading=12, spaceAfter=0)
    s_cellb = ParagraphStyle("CB", parent=s_cell, fontName=bold)
    s_kp = ParagraphStyle("K", parent=s_body, leftIndent=10, bulletIndent=0)

    story = [
        Paragraph(esc(title), s_title),
        Paragraph(esc(f"Generated {datetime.now():%d %b %Y, %H:%M}  ·  {len(doc_names)} document(s)  ·  "
                      f"{len(keywords)} keyword(s)  ·  {len(matches)} match(es)"), s_meta),
        Spacer(1, 3),
        Paragraph("<b>Documents:</b> " + esc(", ".join(doc_names) or "-"), s_meta),
        Spacer(1, 10),
    ]

    # overview table
    rows = [[Paragraph("Keyword", s_cellb), Paragraph("Matches", s_cellb), Paragraph("Found in", s_cellb)]]
    for k in keywords:
        ms = [m for m in matches if m.keyword == k]
        found = ", ".join(sorted({m.doc for m in ms})) or "—"
        rows.append([Paragraph(esc(k), s_cell), Paragraph(str(len(ms)), s_cell), Paragraph(esc(found), s_cell)])
    t = Table(rows, colWidths=[55 * mm, 22 * mm, None], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EDF5")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # one section per keyword
    for n, k in enumerate(keywords, 1):
        ms = [m for m in matches if m.keyword == k]
        story.append(Paragraph(esc(f"{n}. {k}  —  {len(ms)} match(es)"), s_h))
        if not ms:
            story.append(Paragraph("No matches found in the selected documents.", s_meta))
            continue
        if key_points and len(ms) >= 5:
            pts = extractive_summary(" ".join(m.text for m in ms), 3, k)
            story.append(Paragraph("<b>Key points</b>", s_body))
            for line in pts.splitlines():
                line = line.lstrip("• ").strip()
                if line:
                    story.append(Paragraph(highlight(line, regs[k]), s_kp, bulletText="•"))
            story.append(Spacer(1, 4))
        for m in ms:
            story.append(KeepTogether([
                Paragraph(esc(f"{m.doc}  ·  page/section {m.page}"), s_meta),
                Paragraph(highlight(m.text, regs[k]), s_body),
                HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#E5E7EB"),
                           spaceBefore=3, spaceAfter=5),
            ]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(base, 8)
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.drawString(18 * mm, 10 * mm, clean(title)[:90])
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(target, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=18 * mm, title=clean(title)
                      ).build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------- #
# 4c. Scope extraction: engineering / supply / design inputs / performance ...
#     Reads the document clause by clause (headings, bullets, table rows,
#     sentences) and sorts each clause into a category.
#     >>> EDIT `CATEGORIES` below to add/rename categories or tune the cues. <<<
# --------------------------------------------------------------------------- #
MODAL = re.compile(r"\b(shall|must|will|should|required|requires?|to be|is to|are to)\b", re.I)

_UNITS = (r"(?:m3/h|m³/h|m3/hr|m³/hr|m3/s|m³/s|Nm3/h|Nm³/h|t/h|t/hr|kg/h|kg/hr|kg/cm2|kg/cm²|kg/m3|kg/m³|"
          r"t/m3|t/m³|g/cm3|g/cm³|g/cc|mg/Nm3|mg/Nm³|kcal/kg|kJ/kg|MJ/kg|m/sec|m/s|l/s|"
          r"TPH|TPD|lpm|lps|gpm|barg?|mbar|MPa|kPa|Pa|psi|mm ?WC|mWC|mlc|mm|cm|km|µm|microns?|m2|m²|m3|m³|m|"
          r"°\s?C|deg\.? ?C|oC|kW|MW|kVA|kV|Hz|rpm|HP|BHP|%|dB\(?A?\)?|ppm|mps|"
          r"hrs?|hours?|minutes?|mins?|seconds?|secs?|days?|months?|years?|nos?\.?|sets?|kg|MT|tonnes?|MWh|kWh|kN|Nm|inch(?:es)?|"
          r"(?-i:K|V|N|t)|(?-i:A)(?![\w)]))")
NUM_UNIT = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?(?:\s*(?:-|–|to)\s*\d[\d,]*(?:\.\d+)?)?)\s*("
                      + _UNITS + r"(?:\s?\((?:g|a|abs|gauge)\))?)(?![A-Za-z])", re.I)

STD_RX = re.compile(r"\b(IS|IEC|IEEE|ASME|ASTM|API|ANSI|BS|DIN|EN|ISO|NFPA|AWS|AISC|AWWA|MSS|NEMA|TEMA|JIS|CEA|IBC)"
                    r"\b[\s\-:./]*([A-Z]{0,2}\d[\w.\-/:]*(?:\s*\((?:Part|Pt\.?)\s*\d+\))?)")


def _rx(*patterns):
    return [re.compile(p, re.I) for p in patterns]


CATEGORIES = [
    {"name": "Engineering Scope", "code": "ES", "w": 1.0,
     "desc": "design & engineering work the bidder must perform: calculations, layouts, P&IDs, data sheets, surveys, analyses",
     "heading": _rx(r"engineering", r"detail(ed)? design", r"design (and|&) (engineering|detailing)"),
     "cues": _rx(r"\bdesign(ed|ing|s)?\b", r"\bengineering\b", r"\bcalculations?\b", r"\blayouts?\b",
                 r"\bp&?i&?ds?\b|\bpiping and instrumentation\b", r"\bdata ?sheets?\b", r"\bsizing\b",
                 r"\bgeneral arrangement\b|\bga drawings?\b", r"\b3-?d (model|modelling)\b",
                 r"\b(surge|stress|hydraulic|flexibility|seismic|transient) analysis\b",
                 r"\b(soil|geotechnical) investigation\b|\btopographical survey\b", r"\bbasic engineering\b")},
    {"name": "Supply Scope", "code": "SS", "w": 1.0,
     "desc": "equipment, materials and items the bidder must supply / furnish (bill of quantities, quantities, accessories)",
     "heading": _rx(r"scope of supply", r"\bsupply\b", r"bill of (quantit|material)", r"\bBOQ\b|\bBOM\b",
                    r"equipment list", r"schedule of (quantit|supply)", r"items? (covered|included)"),
     "cues": _rx(r"\b(supply|supplied|supplying|furnish(ed|ing)?)\b", r"\bscope of supply\b",
                 r"\bbill of (quantities|materials)\b|\bBOQ\b|\bBOM\b",
                 r"\b\d+\s*(?:nos?\b\.?|sets?\b|lots?\b|numbers?\b)", r"\baccessories\b|\bappurtenances\b",
                 r"\bcomplete with\b|\bcomplete in all respects\b", r"\bwith all (necessary )?(fittings|accessories)\b")},
    {"name": "Mechanical & Piping Scope", "code": "MP", "w": 1.0,
     "desc": "mechanical equipment, piping, valves, pumps, coating, insulation and related mechanical items",
     "heading": _rx(r"mechanical", r"piping", r"\bpumps?\b", r"\bvalves?\b"),
     "cues": _rx(r"\bpip(e|es|ing)\b", r"\bvalves?\b", r"\bpumps?\b", r"\bflanges?\b", r"\bweld(ing|ed|s)?\b",
                 r"\bpaint(ing)?\b|\bcoating\b|\bprimer\b", r"\binsulation\b|\blagging\b|\bcladding\b",
                 r"\bexpansion joints?\b", r"\bpipe supports?\b|\bhangers?\b|\bgaskets?\b|\bfasteners?\b",
                 r"\b(mild steel|stainless steel|cast iron|ni-?hard|hdpe|rubber lined|carbon steel)\b",
                 r"\bslurry\b|\bconveying\b|\bhoppers?\b|\bsilos?\b|\btanks?\b|\bstrainers?\b|\bconveyors?\b|\bblowers?\b|\bcompressors?\b")},
    {"name": "Civil & Structural Scope", "code": "CS", "w": 1.0,
     "desc": "foundations, structures, buildings, roads, drains and other civil works",
     "heading": _rx(r"civil", r"structural", r"foundations?", r"architectur"),
     "cues": _rx(r"\bfoundations?\b", r"\brcc\b|\bconcrete\b|\bgrouting\b", r"\bstructural steel\b|\bstructures?\b",
                 r"\bpedestals?\b|\bpipe racks?\b|\btrestles?\b|\bsupport structures?\b",
                 r"\bbuildings?\b|\bshelters?\b|\broads?\b|\bculverts?\b|\bpaving\b|\bfencing\b",
                 r"\bdrain(age|s)?\b|\bexcavation\b|\bpiling\b|\bcivil\b|\bearthwork\b")},
    {"name": "Electrical Scope", "code": "EL", "w": 1.0,
     "desc": "motors, cables, switchgear, panels, earthing, lighting and power supply items",
     "heading": _rx(r"electrical", r"\bmotors?\b", r"cabl(e|ing)", r"earthing", r"switchgear|\bMCC\b"),
     "cues": _rx(r"\bmotors?\b", r"\bcables?\b|\bcable trays?\b|\bcabling\b", r"\bearthing\b|\bgrounding\b",
                 r"\bswitchgears?\b|\bMCC\b|\bPCC\b", r"\btransformers?\b", r"\bVFD\b|\bvariable frequency\b",
                 r"\b\d+(\.\d+)?\s*kV\b|\bLT\b|\bHT\b", r"\blighting\b|\blight fittings?\b",
                 r"\b(local control|junction|control) (panels?|boxes?|stations?)\b", r"\bpower supply\b|\bUPS\b")},
    {"name": "C&I Scope", "code": "CI", "w": 1.0,
     "desc": "control & instrumentation, PLC/DCS, field instruments, interlocks and automation",
     "heading": _rx(r"instrument", r"control (and|&) instrumentation", r"\bC ?& ?I\b", r"automation", r"\bPLC\b|\bDCS\b|\bSCADA\b"),
     "cues": _rx(r"\bPLC\b|\bDCS\b|\bSCADA\b|\bHMI\b", r"\binstrument(s|ation)?\b",
                 r"\b(level|pressure|flow|temperature) (transmitters?|switch(es)?|gauges?|indicators?|sensors?)\b",
                 r"\binterlocks?\b|\bprotections?\b and\b", r"\bcontrol systems?\b|\bcontrol philosophy\b",
                 r"\bflow ?meters?\b|\bsensors?\b|\bI/?O\b|\bfield instruments?\b", r"\bautomation\b|\bautomatic(ally)?\b")},
    {"name": "Design Inputs", "code": "DI", "w": 1.0,
     "desc": "design basis & site/process data: capacities, flows, quantities, pressures, temperatures, site conditions, ash/coal data",
     "heading": _rx(r"design (basis|data|input|parameters?|criteria|conditions?)", r"basic (data|design)",
                    r"site (conditions?|data|information)", r"(process|operating|input|technical|salient) (data|conditions?|parameters?)",
                    r"meteorolog|climat", r"(ash|coal|water) (quantity|characteristics|properties|analysis)"),
     "cues": [NUM_UNIT] + _rx(r"\b(ambient|relative humidity|altitude|seismic zone|wind (speed|zone)|rainfall|site elevation)\b",
                 r"\b(ash|coal|fly ash|bottom ash|slurry|water) (quantity|generation|flow|content|production|rate|concentration|density|characteristics|analysis)\b",
                 r"\bdesign (pressure|temperature|capacity|flow|life|head|velocity)\b",
                 r"\b(capacity|flow ?rate|concentration|velocity|density|viscosity|specific gravity|particle size|head|elevation|distance|pH)\b")},
    {"name": "Performance Requirements", "code": "PR", "w": 1.0,
     "desc": "guaranteed / required performance: capacity, efficiency, availability, noise, consumption, emission limits, PG tests, LDs",
     "heading": _rx(r"performance", r"guarantee", r"efficiency|availability"),
     "cues": _rx(r"(?<!bank )\bguarantee[sd]?\b(?! period)",
                 r"\bperformance (guarantee|test|requirement|parameter)s?\b|\bperformance\b",
                 r"\bshall (achieve|meet|maintain|deliver|be capable of|not exceed|limit)\b",
                 r"\b(not (less|more|greater|higher|lower) than|at least|not exceed(ing)?|maximum of|minimum of|up to)\b[^.]{0,60}\d",
                 r"\b(availability|efficiency|noise level|vibration|power consumption|water consumption|dust emission|leakage|throughput|turndown)\b",
                 r"\bliquidated damages\b|\bPG tests?\b|\bperformance tests?\b",
                 r"\bcapable of\b|\bcontinuous(ly)? (operation|duty|rating)\b")},
    {"name": "Battery Limits & Interfaces", "code": "BL", "w": 1.0,
     "desc": "scope boundaries, terminal/tie-in points, interfaces with other packages or the owner",
     "heading": _rx(r"battery limits?", r"interfaces?", r"tie-?ins?", r"terminal points?", r"scope boundar", r"plant boundary"),
     "cues": _rx(r"\bbattery limits?\b", r"\binterface(s|d)?\b", r"\btie-?ins?\b", r"\bterminal points?\b",
                 r"\bboundary\b|\bOPB\b|\bplant boundary\b", r"\bpoint of connection\b|\bpoints? of (supply|delivery)\b")},
    {"name": "Exclusions / Owner's Scope", "code": "EX", "w": 2.0,
     "desc": "items NOT in the bidder's scope: by owner / others, excluded work, free-issue items",
     "heading": _rx(r"exclusions?", r"out of scope", r"not in (the )?(bidder|contractor|vendor|supplier)", r"by (owner|purchaser|others)",
                    r"(owner|purchaser|employer)'?s scope"),
     "cues": _rx(r"\bnot (be )?(included )?in (the )?(bidder|contractor|vendor|supplier|seller)'?s? scope",
                 r"\bexclud(e|ed|es|ing)\b", r"\bby (the )?(owner|purchaser|others|employer|client)\b",
                 r"\b(owner|purchaser|employer)'?s? scope\b", r"\bout of (the )?scope\b",
                 r"\bwill be (provided|arranged|supplied|done|carried out) by (the )?(owner|purchaser|employer|others)\b",
                 r"\bfree[- ]issue\b|\bnot (a )?part of\b")},
    {"name": "Codes & Standards", "code": "CD", "w": 1.0,
     "desc": "applicable codes, standards and statutory requirements",
     "heading": _rx(r"codes?\b", r"standards?\b", r"statutory", r"regulations?", r"reference documents?"),
     "cues": [STD_RX] + _rx(r"\bin accordance with\b", r"\bconform(s|ing)? to\b", r"\bcodes? (and|&) standards\b",
                            r"\bstatutory\b|\bregulations?\b|\bapplicable (codes?|standards?)\b")},
    {"name": "Inspection & Testing", "code": "IT", "w": 1.0,
     "desc": "quality plans, inspection, factory/site tests, NDT, witness points",
     "heading": _rx(r"inspection", r"testing", r"quality (assurance|control|plan)", r"\bQAP\b", r"tests? (and|&) (inspection|trials?)"),
     "cues": _rx(r"\binspection\b", r"\btest(s|ing|ed)?\b", r"\bQAP\b|\bQA\b|\bQC\b|\bFAT\b|\bSAT\b",
                 r"\bNDT\b|\bradiograph(y|ic)\b|\bdye penetrant\b|\bultrasonic\b", r"\bhydro(static)? tests?\b|\bleak tests?\b",
                 r"\bwitness(ed|ing)?\b|\bhold points?\b|\bTPI\b|\bthird[- ]party inspection\b", r"\bmill test\b|\bTC\b|\btest certificates?\b")},
    {"name": "Erection, Commissioning & O&M", "code": "EC", "w": 1.0,
     "desc": "erection, installation, commissioning, trial run, training, handover and O&M obligations",
     "heading": _rx(r"erection", r"installation", r"commissioning", r"construction", r"\bsite (work|activit)", r"operation (and|&) maintenance",
                    r"\bO ?& ?M\b", r"training", r"trial (run|operation)"),
     "cues": _rx(r"\berect(ion|ed|ing)?\b", r"\binstall(ation|ed|ing)?\b", r"\bcommission(ing|ed)\b",
                 r"\btrial (run|operation)\b", r"\bat site\b|\bsite (work|activities|office|storage|mobili[sz]ation)\b",
                 r"\btraining\b", r"\bhand(ing)? ?over\b|\btake ?over\b", r"\bO ?& ?M\b|\boperation and maintenance\b",
                 r"\bsupervision\b|\bmanpower\b|\bT ?& ?P\b")},
    {"name": "Documentation & Drawings", "code": "DD", "w": 1.0,
     "desc": "documents, drawings, manuals and data the bidder must submit and get approved",
     "heading": _rx(r"documen", r"submittals?", r"manuals?", r"data ?books?", r"drawing (submission|approval)"),
     "cues": _rx(r"\bsubmit(ted|s|ting)?\b", r"\bapproval\b|\bapproved\b|\bfor (review|information|approval)\b",
                 r"\bmanuals?\b|\bas-?built\b|\bdata ?books?\b|\bdocumentation\b", r"\bhard cop(y|ies)\b|\bsoft cop(y|ies)\b",
                 r"\bdocument (list|schedule|control)\b|\bdrawing (list|schedule)\b", r"\bcategory[ -]?(i|1|ii|2)\b")},
    {"name": "Spares & Tools", "code": "SP", "w": 1.0,
     "desc": "mandatory / commissioning / operational spares, special tools and consumables",
     "heading": _rx(r"spares?", r"tools? (and|&) tackles", r"consumables?"),
     "cues": _rx(r"\bspares?\b|\bspare parts?\b", r"\btools? (and|&) tackles\b|\bspecial tools?\b", r"\bconsumables?\b",
                 r"\b(commissioning|mandatory|operational|insurance|startup|start-up) spares\b", r"\bfirst fill\b|\blubricants?\b")},
    {"name": "Warranty & Commercial", "code": "WC", "w": 1.0,
     "desc": "warranty, delivery/completion schedule, payment, insurance, bank guarantees, penalties",
     "heading": _rx(r"warrant", r"commercial", r"payment", r"delivery", r"schedule", r"insurance", r"general conditions", r"bid\b"),
     "cues": _rx(r"\bwarrant(y|ies)\b|\bdefects? liability\b", r"\bdelivery (schedule|period)\b|\bcompletion (period|schedule|time)\b|\bmilestones?\b",
                 r"\bpayment\b|\binvoice\b", r"\binsurance\b", r"\bbank guarantee\b|\bguarantee period\b|\bsecurity deposit\b|\bearnest money\b",
                 r"\bpenalt(y|ies)\b|\bvalidity\b|\bterms and conditions\b")},
    {"name": "General Requirements", "code": "GR", "w": 1.0,
     "desc": "any other binding requirement that fits none of the above",
     "heading": [], "cues": []},
]
CAT_BY_NAME = {c["name"]: c for c in CATEGORIES}


@dataclass
class ReqItem:
    category: str
    text: str
    section: str = ""
    page: int = 0
    doc: str = ""
    confidence: float = 0.0       # 0..1 (rule based) ; 1.0 for AI items
    param: str = ""               # Design Inputs: parameter name
    value: str = ""               # Design Inputs: value
    source: str = "rule"          # "rule" or "ai"
    ref: str = ""


# ---- reading the document structure --------------------------------------- #
HEAD_NUM = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+([A-Z][^\n]{2,100})$")
HEAD_WORD = re.compile(r"^\s*(?:ANNEXURE|ANNEX|APPENDIX|ATTACHMENT|SCHEDULE|SECTION|PART|CHAPTER|EXHIBIT)\b[^\n]{0,100}$", re.I)
BULLET = re.compile(r"^\s*(?:[-•*▪●○■►–◦·✓✔➢➤\x7f\x80-\x9f\uf000-\uf8ff\u2022-\u2023\u25a0-\u25ff]|\(?\d{1,2}[.)]|\(?[a-z][.)]|\(?(?:ii|iii|iv|vi|vii|viii|ix|x)[.)])\s+(.*\S)")


def _heading(line: str):
    """Return (level, title) if the line looks like a heading, else None."""
    if len(line) > 110 or line.endswith((".", ",", ";", ":")):
        return None
    m = HEAD_NUM.match(line)
    if m:
        title = m.group(2).strip()
        if (len(title.split()) <= 14 and not MODAL.search(title) and not NUM_UNIT.fullmatch(title)
                and sum(ch.isdigit() for ch in title) < len(title) / 3):
            num = re.sub(r"(\.0+)+$", "", m.group(1))
            return num.count(".") + 1, title
        return None
    if HEAD_WORD.match(line) and not MODAL.search(line):
        return 1, line.strip()
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 4 and line.isupper() and len(line) <= 90 and sum(c.isdigit() for c in line) < 6:
        return 1, line.strip()
    return None


def _is_row(line: str) -> bool:
    """Table row / parameter line (own clause instead of being merged into a paragraph)."""
    if line.count("|") >= 1 and len(line.split("|")) >= 2:
        return True
    return (len(line) < 70 and len(line.split()) <= 8 and NUM_UNIT.search(line) is not None
            and not line.endswith(".") and line[0].isupper() and not MODAL.search(line))


def strip_repeating_lines(pages):
    """Drop page headers/footers (lines that repeat on many pages, e.g. 'Page 5 of 90')."""
    if len(pages) < 4:
        return pages
    norm = lambda l: re.sub(r"\d+", "#", l.strip().lower())
    cnt: Counter = Counter()
    for _, t in pages:
        for l in {norm(x) for x in t.split("\n") if len(x.strip()) >= 4}:
            cnt[l] += 1
    limit = max(3, int(0.4 * len(pages)))
    bad = {l for l, c in cnt.items() if c >= limit and len(l) < 120}
    return [(p, "\n".join(x for x in t.split("\n") if not x.strip() or norm(x) not in bad)) for p, t in pages]


def iter_clauses(pages):
    """Yield clauses: {page, path (heading trail), lead (lead-in sentence), text, bullet, row}."""
    path: list[tuple[int, str]] = []
    st = {"lead": "", "buf": [], "page": 0, "kind": "para"}
    out: list[dict] = []

    def flush():
        buf, kind, page = st["buf"], st["kind"], st["page"]
        st["buf"], st["kind"] = [], "para"
        if not buf:
            return
        text = buf[0] if kind == "row" else " ".join(" ".join(buf).split())
        if len(text) < 12:
            return
        if text.endswith(":"):
            st["lead"] = text
            return
        out.append({"page": page, "path": [t for _, t in path], "lead": st["lead"],
                    "text": text, "bullet": kind == "bullet", "row": kind == "row"})
        if kind == "para":
            st["lead"] = ""

    for page, text in pages:
        for raw in text.split("\n"):
            line = raw.strip()
            if not line:
                flush()
                continue
            h = _heading(line)
            if h:
                flush()
                level, title = h
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, title))
                st["lead"] = ""
                continue
            m = BULLET.match(line)
            if m:
                flush()
                st.update(buf=[m.group(1)], page=page, kind="bullet")
                continue
            if _is_row(line):
                flush()
                st.update(buf=[line], page=page, kind="row")
                flush()
                continue
            if not st["buf"]:
                st["page"] = page
            st["buf"].append(line)
    flush()
    return out


# ---- classification -------------------------------------------------------- #
def classify(path: list[str], lead: str, text: str):
    """Return (category_name, score) or (None, 0). Heading + lead-in give context."""
    near = path[-1] if path else ""
    parents = " ".join(path[:-1])
    best, best_score = None, 0.0
    for cat in CATEGORIES:
        if not cat["cues"] and not cat["heading"]:
            continue
        s = 0.0
        if any(r.search(near) for r in cat["heading"]):
            s += 3.0
        elif parents and any(r.search(parents) for r in cat["heading"]):
            s += 1.5
        if lead and any(r.search(lead) for r in cat["heading"]):
            s += 1.0
        hits_text = sum(1 for r in cat["cues"] if r.search(text))
        hits_lead = sum(1 for r in cat["cues"] if lead and not r.search(text) and r.search(lead))
        s += (min(hits_text, 4) + 0.5 * min(hits_lead, 2)) * cat["w"]
        if s > best_score:
            best, best_score = cat["name"], s
    if best_score >= 2.0:
        return best, best_score
    return None, best_score


def split_param_value(t: str, row: bool = False):
    """'Ambient temperature : 45 °C' -> ('Ambient temperature', '45 °C')"""
    t = t.strip()
    if "|" in t:
        cells = [x.strip() for x in t.split("|") if x.strip()]
        if len(cells) >= 2 and re.search(r"\d", " ".join(cells[1:])):
            return cells[0], " ".join(cells[1:])
    m = re.match(r"^\s*([A-Za-z][^:=|]{2,70}?)\s*(?::|=|\s[-–]\s)\s*(.*\d.*)$", t)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip(".")
    m = re.match(r"^(?:[Tt]he\s+)?([A-Za-z][A-Za-z /()\-]{3,60}?)\s+(?:shall be|will be|is|are)\s+((?:[<>≤≥~+\-]?\d).*)$", t)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip(".")
    m = re.match(r"^(?:[Tt]he\s+)?([A-Za-z][A-Za-z /()\-]{3,50}?)\s+of\s+((?:[<>≤≥~+\-]?\d).*)$", t)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip(".")
    if row:
        m = re.match(r"^([A-Za-z][A-Za-z /()\-]{2,50}?)\s+((?:[<>≤≥+\-]?\d).*)$", t)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return "", ""


def get_pages(store: "DocumentStore", doc_id: int) -> list[tuple[int, str]]:
    return [(r["page"], r["text"]) for r in store.conn.execute(
        "SELECT page, text FROM pages WHERE doc_id=? ORDER BY page", (doc_id,))]


def dedupe_items(items: list[ReqItem]) -> list[ReqItem]:
    seen, out = set(), []
    for it in items:
        key = (it.category, re.sub(r"\W+", "", it.text.lower())[:200])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def rule_based_extract(pages, doc_name: str) -> list[ReqItem]:
    """Offline extraction: no internet, no API key. Heuristic - always review the result."""
    items: list[ReqItem] = []
    for c in iter_clauses(strip_repeating_lines(pages)):
        units = [c["text"]] if (c["bullet"] or c["row"]) else _sentences(c["text"])
        for u in units:
            u = u.strip()
            if len(u) < 20 or u.endswith(":"):
                continue
            cat, score = classify(c["path"], c["lead"], u)
            param, value = "", ""
            if cat is None or cat == "General Requirements":
                param, value = split_param_value(u, c["row"])
                if param and NUM_UNIT.search(value):               # 'Parameter : value' line
                    cat, score = "Design Inputs", 2.0
                elif MODAL.search(u) and len(u) >= 40:
                    cat, score = "General Requirements", 1.0
                else:
                    continue
            if cat == "Design Inputs" and not param:
                param, value = split_param_value(u, c["row"])
            items.append(ReqItem(cat, u[:700], " > ".join(c["path"][-2:]), c["page"], doc_name,
                                 round(min(score / 6.0, 1.0), 2), param, value, "rule"))
    return dedupe_items(items)


def extract_standards(pages) -> list[dict]:
    found: dict[str, set] = {}
    counts: Counter = Counter()
    for page, text in pages:
        for m in STD_RX.finditer(text):
            num = m.group(2).strip(" .,-:;/")
            if not re.search(r"\d", num) or len(num) > 25:
                continue
            key = f"{m.group(1)} {num}"
            counts[key] += 1
            found.setdefault(key, set()).add(page)
    return [{"standard": k, "count": counts[k], "pages": ", ".join(str(p) for p in sorted(found[k])[:12])}
            for k in sorted(found)]


def group_items(items: list[ReqItem]) -> dict[str, list[ReqItem]]:
    groups: dict[str, list[ReqItem]] = {c["name"]: [] for c in CATEGORIES}
    for it in items:
        if it.category not in groups:
            it.category = "General Requirements"
        groups[it.category].append(it)
    for name, lst in groups.items():
        lst.sort(key=lambda i: i.page)
        for n, it in enumerate(lst, 1):
            it.ref = f"{CAT_BY_NAME[name]['code']}-{n:03d}"
    return groups


# ---- AI extraction (Claude) ------------------------------------------------ #
_CLAUDE = {"client": None}


def _claude_text(prompt: str, max_tokens: int = 8000) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(_pip_hint("anthropic"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable first.")
    if _CLAUDE["client"] is None:
        _CLAUDE["client"] = anthropic.Anthropic()
    msg = _CLAUDE["client"].messages.create(model=AI_MODEL, max_tokens=max_tokens,
                                            messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _parse_items(raw: str) -> list[dict]:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
        data = data.get("items", []) if isinstance(data, dict) else data
        return [d for d in data if isinstance(d, dict) and d.get("text")]
    except Exception:
        pass
    items, dec, pos = [], json.JSONDecoder(), raw.find("{")        # salvage a truncated reply
    while pos != -1:
        try:
            obj, end = dec.raw_decode(raw, pos)
            items.append(obj)
            pos = raw.find("{", end)
        except json.JSONDecodeError:
            pos = raw.find("{", pos + 1)
    return [d for d in items if isinstance(d, dict) and d.get("text")]


def _match_category(name: str) -> str:
    n = (name or "").strip().lower()
    for c in CATEGORIES:
        if c["name"].lower() == n:
            return c["name"]
    for c in CATEGORIES:
        if n and (n in c["name"].lower() or c["name"].lower() in n):
            return c["name"]
    return "General Requirements"


def ai_extract(pages, doc_name: str, progress=None) -> list[ReqItem]:
    cats = "\n".join(f"- {c['name']}: {c['desc']}" for c in CATEGORIES)
    chunks, cur = [], ""
    for page, text in pages:
        block = f"\n[[Page {page}]]\n{text}\n"
        if cur and len(cur) + len(block) > AI_CHUNK_CHARS:
            chunks.append(cur)
            cur = ""
        cur += block
    if cur.strip():
        chunks.append(cur)
    items: list[ReqItem] = []
    for i, ch in enumerate(chunks, 1):
        if progress:
            progress(i, len(chunks))
        prompt = (
            "You are a tender / technical-specification analyst. Below is part of a specification "
            "(pages are marked [[Page N]]). Extract every scope item, requirement, data point and "
            "commitment and sort each into exactly one of these categories:\n"
            f"{cats}\n\nRules:\n"
            "- Classify by what the clause MEANS, not by keywords.\n"
            "- One item per distinct requirement / list entry; keep exact figures, units, quantities, standards.\n"
            "- 'text' must be concise (max ~45 words) and faithful to the source. Do not invent anything.\n"
            "- For Design Inputs also give 'parameter' and 'value' (with unit).\n"
            "- Skip boilerplate, headers/footers, tables of contents and pure formatting text.\n"
            "- Use the heading or clause number in 'section' when visible.\n"
            'Return ONLY JSON, no commentary: {"items":[{"category":"<exact category name>","text":"...",'
            '"section":"...","page":<int>,"parameter":"","value":""}]}\n\n'
            f"SPECIFICATION TEXT:\n{ch}")
        for d in _parse_items(_claude_text(prompt)):
            pg = d.get("page")
            items.append(ReqItem(_match_category(d.get("category", "")), str(d["text"]).strip()[:700],
                                 str(d.get("section", "") or ""), int(pg) if str(pg).isdigit() else 0,
                                 doc_name, 1.0, str(d.get("parameter", "") or ""),
                                 str(d.get("value", "") or ""), "ai"))
    return dedupe_items(items)


# ---- exports: Excel + PDF + CSV ------------------------------------------- #
def _conf_label(it: ReqItem) -> str:
    if it.source == "ai":
        return "AI"
    return "High" if it.confidence >= 0.8 else "Medium" if it.confidence >= 0.5 else "Low"


def scope_rows(items: list[ReqItem]) -> list[dict]:
    rows = []
    for name, lst in group_items(items).items():
        for it in lst:
            rows.append({"Category": name, "Ref": it.ref, "Requirement": it.text,
                         "Parameter": it.param, "Value": it.value, "Section": it.section,
                         "Page": it.page, "Document": it.doc, "Confidence": _conf_label(it)})
    return rows


def scope_csv(items: list[ReqItem]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["Category", "Ref", "Requirement", "Parameter", "Value",
                                        "Section", "Page", "Document", "Confidence"])
    w.writeheader()
    w.writerows(scope_rows(items))
    return buf.getvalue().encode("utf-8-sig")


_XL_BAD = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _xl_safe(v):
    """Make a value safe for an Excel cell (no illegal control chars, no accidental formulas)."""
    if v is None:
        return ""
    if isinstance(v, str):
        v = _XL_BAD.sub(" ", v)
        return " " + v if v.startswith("=") else v
    return v


def export_scope_xlsx(items: list[ReqItem], standards: list[dict], target, title: str, doc_names: list[str]) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise RuntimeError(_pip_hint("openpyxl"))
    groups = group_items(items)
    wb = Workbook()
    head_fill = PatternFill("solid", fgColor="0F4C81")
    head_font = Font(bold=True, color="FFFFFF")
    wrap = Alignment(wrap_text=True, vertical="top")

    safe = _xl_safe

    def sheet(name, headers, rows, widths):
        name = re.sub(r"[\\/*?:\[\]]", "-", name)[:31]
        ws = wb.create_sheet(name)
        ws.append(headers)
        for c in ws[1]:
            c.fill, c.font, c.alignment = head_fill, head_font, Alignment(vertical="center")
        for r in rows:
            ws.append([safe(v) for v in r])
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = wrap
        ws.freeze_panes = "A2"
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        return ws

    ws = wb.active
    ws.title = "Summary"
    ws["A1"], ws["A1"].font = _xl_safe(title), Font(bold=True, size=14)
    ws["A2"] = _xl_safe("Documents: " + ", ".join(doc_names))
    ws["A3"] = f"Generated {datetime.now():%d %b %Y %H:%M}. Review all rule-based items before use."
    ws.append([])
    ws.append(["Category", "Items", "Sheet"])
    for c in ws[5]:
        c.fill, c.font = head_fill, head_font
    for name, lst in groups.items():
        ws.append([name, len(lst), re.sub(r"[\\/*?:\[\]]", "-", name)[:31] if lst else "-"])
    ws.append(["Codes & standards referenced", len(standards), "Standards"])
    ws.append(["TOTAL ITEMS", len(items), ""])
    ws.cell(ws.max_row, 1).font = Font(bold=True)
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 32

    allrows = scope_rows(items)
    sheet("All Items", ["Category", "Ref", "Requirement", "Parameter", "Value", "Section", "Page", "Document", "Confidence"],
          [[r["Category"], r["Ref"], r["Requirement"], r["Parameter"], r["Value"], r["Section"], r["Page"],
            r["Document"], r["Confidence"]] for r in allrows], [24, 9, 80, 24, 20, 30, 7, 24, 11])
    for name, lst in groups.items():
        if not lst:
            continue
        if name == "Design Inputs":
            sheet(name, ["Ref", "Parameter", "Value", "Full text", "Section", "Page", "Confidence"],
                  [[i.ref, i.param, i.value, i.text, i.section, i.page, _conf_label(i)] for i in lst],
                  [9, 30, 24, 70, 30, 7, 11])
        else:
            sheet(name, ["Ref", "Requirement", "Section", "Page", "Document", "Confidence"],
                  [[i.ref, i.text, i.section, i.page, i.doc, _conf_label(i)] for i in lst],
                  [9, 90, 30, 7, 24, 11])
    if standards:
        sheet("Standards", ["Standard", "Mentions", "Pages"],
              [[s["standard"], s["count"], s["pages"]] for s in standards], [26, 10, 40])
    wb.save(target)


def build_scope_pdf(items: list[ReqItem], standards: list[dict], target, title: str, doc_names: list[str]) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (HRFlowable, KeepTogether, PageBreak, Paragraph,
                                        SimpleDocTemplate, Spacer, Table, TableStyle)
    except ImportError:
        raise RuntimeError(_pip_hint("reportlab"))
    base, bold, unicode_ok = _register_pdf_font()
    clean = lambda s: (lambda t: t if unicode_ok else t.encode("latin-1", "replace").decode("latin-1"))(
        re.sub(r"[\x00-\x08\x0b-\x1f]", " ", str(s)))
    esc = lambda s: html.escape(clean(s), quote=False)
    ss = getSampleStyleSheet()
    grey, blue = colors.HexColor("#6B7280"), colors.HexColor("#0F4C81")
    s_title = ParagraphStyle("T", parent=ss["Title"], fontName=bold, fontSize=20, leading=24, alignment=0,
                             textColor=colors.HexColor("#1F2937"))
    s_meta = ParagraphStyle("M", parent=ss["Normal"], fontName=base, fontSize=8.5, leading=12, textColor=grey)
    s_h = ParagraphStyle("H", parent=ss["Heading2"], fontName=bold, fontSize=13.5, leading=17,
                         spaceBefore=14, spaceAfter=4, textColor=blue)
    s_body = ParagraphStyle("B", parent=ss["Normal"], fontName=base, fontSize=9.5, leading=13, spaceAfter=1)
    s_cell = ParagraphStyle("C", parent=s_body, fontSize=9, leading=12, spaceAfter=0)
    s_cellb = ParagraphStyle("CB", parent=s_cell, fontName=bold)
    groups = group_items(items)

    def table(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EDF5")),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return t

    story = [Paragraph(esc(title), s_title),
             Paragraph(esc(f"Generated {datetime.now():%d %b %Y, %H:%M}  ·  {len(items)} item(s)  ·  "
                           f"Source: {', '.join(doc_names)}"), s_meta),
             Paragraph("Auto-extracted - review before use.", s_meta), Spacer(1, 8)]
    rows = [[Paragraph("Category", s_cellb), Paragraph("Items", s_cellb)]]
    for name, lst in groups.items():
        rows.append([Paragraph(esc(name), s_cell), Paragraph(str(len(lst)), s_cell)])
    rows.append([Paragraph("Codes & standards referenced", s_cell), Paragraph(str(len(standards)), s_cell)])
    story.append(table(rows, [100 * mm, 25 * mm]))

    for name, lst in groups.items():
        if not lst:
            continue
        story.append(Paragraph(esc(f"{name}  ({len(lst)})"), s_h))
        if name == "Design Inputs":
            drows = [[Paragraph(h, s_cellb) for h in ("Ref", "Parameter", "Value / requirement", "Page")]]
            for i in lst:
                drows.append([Paragraph(i.ref, s_cell), Paragraph(esc(i.param or "-"), s_cell),
                              Paragraph(esc(i.value or i.text), s_cell), Paragraph(str(i.page), s_cell)])
            story.append(table(drows, [18 * mm, 50 * mm, None, 14 * mm]))
            continue
        for i in lst:
            sec = f" · {i.section}" if i.section else ""
            story.append(KeepTogether([
                Paragraph(f"<b>{esc(i.ref)}</b>&nbsp;&nbsp;{esc(i.text)}", s_body),
                Paragraph(esc(f"page {i.page}{sec}"), s_meta),
                HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#E5E7EB"), spaceBefore=2, spaceAfter=4)]))
    if standards:
        story.append(Paragraph(esc(f"Codes & standards referenced  ({len(standards)})"), s_h))
        srows = [[Paragraph(h, s_cellb) for h in ("Standard", "Mentions", "Pages")]]
        srows += [[Paragraph(esc(s["standard"]), s_cell), Paragraph(str(s["count"]), s_cell),
                   Paragraph(esc(s["pages"]), s_cell)] for s in standards]
        story.append(table(srows, [50 * mm, 22 * mm, None]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(base, 8)
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.drawString(18 * mm, 10 * mm, clean(title)[:90])
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(target, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=18 * mm, title=clean(title)).build(story, onFirstPage=footer, onLaterPages=footer)


def run_scope_extraction(store: "DocumentStore", doc, use_ai: bool = False, progress=None):
    """Return (items, standards) for one indexed document row."""
    pages = get_pages(store, doc["id"])
    if not any(t.strip() for _, t in pages):
        raise RuntimeError("No readable text in this document (scanned PDF? it needs OCR first).")
    items = ai_extract(pages, doc["name"], progress) if use_ai else rule_based_extract(pages, doc["name"])
    return items, extract_standards(pages)


# --------------------------------------------------------------------------- #
# 4d. Design inputs: pipe size, pump capacity, head, pressure, motor rating ...
#     One row per value found anywhere in the document.
#     >>> EDIT `EQUIPMENT`, `FAMILY_GROUP` and the keyword lists to tune. <<<
# --------------------------------------------------------------------------- #
EQUIPMENT = [
    ("Pump", r"\bpumps?\b|\bpumping\b"),
    ("Pipeline", r"\bpipelines?\b|\bpiping\b|\bpipes?\b|\bheaders?\b|\bline\b"),
    ("Valve", r"\bvalves?\b"),
    ("Motor", r"\bmotors?\b"),
    ("Tank / Sump", r"\btanks?\b|\bsumps?\b|\bpits?\b|\bbasins?\b|\breservoirs?\b"),
    ("Silo / Hopper", r"\bsilos?\b|\bhoppers?\b|\bbunkers?\b"),
    ("Blower / Compressor", r"\bblowers?\b|\bcompressors?\b|\bfans?\b"),
    ("Conveyor", r"\bconveyors?\b"),
    ("Ash dyke / pond", r"\bash (dyke|pond)\b|\bdyke\b"),
    ("Slurry", r"\bslurry\b"),
    ("Strainer / Filter", r"\bstrainers?\b|\bfilters?\b"),
    ("Agitator / Mixer", r"\bagitators?\b|\bmixers?\b"),
    ("Transformer", r"\btransformers?\b"),
    ("Cable", r"\bcables?\b"),
]
_EQ_RX = [(n, re.compile(p, re.I)) for n, p in EQUIPMENT]

FAMILY_GROUP = {
    "Capacity": "Hydraulic & Process", "Flow": "Hydraulic & Process", "Head": "Hydraulic & Process",
    "Pressure": "Hydraulic & Process", "Temperature": "Hydraulic & Process", "Velocity": "Hydraulic & Process",
    "Concentration": "Hydraulic & Process", "Density": "Hydraulic & Process", "Specific gravity": "Hydraulic & Process",
    "pH": "Hydraulic & Process", "Particle size": "Hydraulic & Process", "Efficiency / availability": "Hydraulic & Process",
    "Duration": "Hydraulic & Process", "Volume": "Hydraulic & Process", "Weight": "Hydraulic & Process",
    "Pipe size": "Piping & Mechanical", "Pipe schedule": "Piping & Mechanical", "Thickness": "Piping & Mechanical",
    "Dimension": "Piping & Mechanical", "Distance / length": "Piping & Mechanical",
    "Elevation / level": "Piping & Mechanical", "Quantity": "Piping & Mechanical",
    "Configuration": "Piping & Mechanical", "Speed": "Piping & Mechanical",
    "Power": "Electrical", "Voltage": "Electrical", "Frequency": "Electrical",
    "Ambient temperature": "Site & Environmental", "Seismic zone": "Site & Environmental",
    "Wind speed": "Site & Environmental", "Humidity": "Site & Environmental", "Altitude": "Site & Environmental",
    "Rainfall": "Site & Environmental",
}
GROUP_ORDER = ["Hydraulic & Process", "Piping & Mechanical", "Electrical", "Site & Environmental", "Other"]

_PIPE_SIZE = re.compile(
    r"\b(?:DN|NB|NPS)\s*[-:]?\s*\d{2,4}\b|\b\d{2,4}(?:\.\d+)?\s*(?:mm\s*)?(?:NB|DN)\b"
    r"|\b\d{1,3}(?:\.\d+)?\s*(?:inch(?:es)?|\")\s*(?:NB|DN|dia(?:meter)?|OD|ID|pipe)?"
    r"|\b(?:OD|ID)\s*[-:]?\s*\d{2,4}(?:\.\d+)?\s*(?:mm)?", re.I)
_SCHEDULE = re.compile(r"\bSch(?:edule)?[\s.\-]*(?:\d{1,3}S?|STD|XXS|XS)\b", re.I)
_DUTY = re.compile(r"\b\d\s*[xX×]\s*\d{2,3}\s*%|\b\d+\s*(?:working|duty|running)\s*\+\s*\d+\s*(?:standby|stand-by|spare)\b", re.I)
_SEISMIC = re.compile(r"seismic zone\s*[-:]?\s*([IVX]+|\d)\b", re.I)
_PH = re.compile(r"\bpH\s*[:=]?\s*(\d+(?:\.\d+)?(?:\s*(?:-|–|to)\s*\d+(?:\.\d+)?)?)")
_SG = re.compile(r"(?:specific gravity|\bS\.?G\.?)\s*[:=]?\s*(\d+(?:\.\d+)?)", re.I)
_COMMERCIAL = re.compile(r"liquidated|penalt|payment|invoice|bank guarantee|warrant|delivery schedule|earnest|"
                         r"security deposit|retention money|validity|defects? liability|\bLD\b", re.I)
_QUAL = re.compile(r"\b(design|operating|working|maximum|max|minimum|min|test|hydro ?test|rated|normal|peak|average|shut-?off)\b", re.I)


def _nearest_equipment(sentence: str, pos: int, fallback: str = "") -> str:
    best, best_d = "", 10 ** 6
    for name, rx in _EQ_RX:
        for m in rx.finditer(sentence):
            d = pos - m.end() if m.end() <= pos else (m.start() - pos) + 15     # prefer mentions before the value
            if name in ("Slurry", "Ash dyke / pond"):
                d += 25                                                          # context nouns, not equipment
            if d < best_d:
                best, best_d = name, d
    if best:
        return best
    for name, rx in _EQ_RX:
        if fallback and rx.search(fallback):
            return name
    return ""


def _family_for(unit: str, before: str, after: str):
    """Return (family, strong) for a number+unit match; strong=True if a keyword confirmed it."""
    u = unit.lower().replace(" ", "")
    b, a = before.lower(), after.lower()
    ctx = b + " " + a
    kw = lambda pat, text=ctx: re.search(pat, text) is not None
    if u in ("m3/h", "m³/h", "m3/hr", "m³/hr", "m3/s", "m³/s", "nm3/h", "nm³/h", "t/h", "t/hr", "kg/h", "kg/hr",
             "tph", "tpd", "lpm", "lps", "l/s", "gpm"):
        return "Flow", kw(r"capacity|flow|discharge|throughput|rate|generation|quantity")
    if u in ("mwc", "mlc"):
        return "Head", True
    if u.startswith(("kg/cm", "bar", "mbar", "mpa", "kpa", "psi", "mmwc")) or u == "pa":
        return "Pressure", kw(r"pressure|mawp|hydro|shut-?off|rating|class")
    if u in ("°c", "degc", "deg.c", "oc", "k"):
        if kw(r"ambient|atmospheric|air temp|dry bulb|wet bulb"):
            return "Ambient temperature", True
        return "Temperature", kw(r"temperature|temp\b")
    if u in ("m/s", "m/sec", "mps"):
        if kw(r"wind"):
            return "Wind speed", True
        return "Velocity", kw(r"velocity|speed|flow")
    if u in ("kw", "mw", "kva", "hp", "bhp"):
        return "Power", kw(r"power|rating|rated|motor|kw|output|load|capacity")
    if u == "rpm":
        return "Speed", True
    if u in ("kv", "v"):
        return "Voltage", kw(r"volt|supply|phase")
    if u == "hz":
        return "Frequency", True
    if u in ("kg/m3", "kg/m³", "t/m3", "t/m³", "g/cm3", "g/cm³", "g/cc"):
        return "Density", True
    if u in ("mg/nm3", "mg/nm³", "ppm"):
        return "Concentration", True
    if u in ("µm", "micron", "microns"):
        return "Particle size", kw(r"particle|size|fineness|mesh")
    if u in ("m3", "m³"):
        return "Volume", kw(r"volume|capacity|storage|holding")
    if u in ("kg", "mt", "t", "tonne", "tonnes", "kn", "n"):
        return "Weight", kw(r"weight|load|mass")
    if u == "%":
        if kw(r"humidity"):
            return "Humidity", True
        if kw(r"concentration|solids?|by weight|wt|consistency|dilution"):
            return "Concentration", True
        if kw(r"efficiency|availability|margin|tolerance|moisture|loss"):
            return "Efficiency / availability", True
        return "Efficiency / availability", False
    if u in ("hrs", "hr", "hours", "hour", "minutes", "minute", "mins", "min", "seconds", "second", "secs", "sec",
             "days", "day", "months", "month", "years", "year"):
        return "Duration", kw(r"design life|\blife\b|storage|autonomy|retention|hold|operation|run|continuous|trial|buffer|per day|shift")
    if u in ("nos", "no", "nos.", "no.", "sets", "set"):
        return "Quantity", True
    if u in ("mm", "cm", "m", "km", "m2", "m²", "inch", "inches"):
        if kw(r"thick|\bwt\b|corrosion allowance|lining"):
            return "Thickness", True
        if kw(r"\b(pipe|piping|line|header|nb|dn|dia|diameter|nominal|bore|size|od|id)\b") and u in ("mm", "inch", "inches", "cm"):
            return "Pipe size", True
        if kw(r"head|tdh|lift"):
            return "Head", True
        if kw(r"elevation|level|height|depth|altitude|\brl\b|above|below|msl"):
            return "Elevation / level", True
        if kw(r"distance|length|route|away|lead|from .* to|run of"):
            return "Distance / length", True
        if u == "km":
            return "Distance / length", False
        if kw(r"rainfall|rain"):
            return "Rainfall", True
        return "Dimension", False
    return "Other", False


def _display_name(family: str, equipment: str, qual: str) -> str:
    fam = family
    if family == "Flow" and equipment in ("Pump", "Blower / Compressor", "Silo / Hopper", "Tank / Sump"):
        fam = "Capacity"
    if family in ("Pressure", "Temperature", "Flow", "Capacity", "Head", "Velocity", "Power") and qual:
        fam = f"{qual.capitalize()} {fam.lower()}"
    if equipment and FAMILY_GROUP.get(family) != "Site & Environmental":
        return f"{equipment}: {fam}"
    return fam


def _spans_overlap(spans, a, b):
    return any(not (b <= s or a >= e) for s, e in spans)


def extract_design_inputs(pages, doc_name: str) -> list[dict]:
    """Offline extraction of design input values (pipe size, pump capacity, head, ...)."""
    rows: list[dict] = []
    for c in iter_clauses(strip_repeating_lines(pages)):
        heading = " ".join(c["path"][-2:])
        units = [c["text"]] if (c["bullet"] or c["row"]) else _sentences(c["text"])
        for u in units:
            u = u.strip()
            if len(u) < 8 or _COMMERCIAL.search(u):
                continue
            label, _ = split_param_value(u, c["row"])
            hits = []                                   # (start, end, value, unit, family, strong)
            spans: list[tuple[int, int]] = []

            def take(rx, family, value_group=0, unit_text=""):
                for m in rx.finditer(u):
                    if _spans_overlap(spans, m.start(), m.end()):
                        continue
                    spans.append((m.start(), m.end()))
                    hits.append((m.start(), m.end(), m.group(value_group).strip(), unit_text, family, True))

            take(_PIPE_SIZE, "Pipe size")
            take(_SCHEDULE, "Pipe schedule")
            take(_DUTY, "Configuration")
            take(_SEISMIC, "Seismic zone", 1)
            take(_PH, "pH", 1)
            take(_SG, "Specific gravity", 1)
            for m in NUM_UNIT.finditer(u):
                if _spans_overlap(spans, m.start(), m.end()):
                    continue
                unit = m.group(2).strip()
                fam, strong = _family_for(unit, u[max(0, m.start() - 55):m.start()], u[m.end():m.end() + 25])
                if fam in ("Other", "Dimension", "Weight") and not strong and not label:
                    continue                            # too ambiguous to be a design input
                if fam == "Duration" and not strong:
                    continue
                spans.append((m.start(), m.end()))
                hits.append((m.start(), m.end(), re.sub(r"\s+", " ", m.group(1)).strip(), unit, fam, strong))
            if not hits:
                continue
            hits.sort()
            for start, end, value, unit, fam, strong in hits:
                eq = _nearest_equipment(u, start, heading + " " + c["lead"])
                if fam in ("Power", "Voltage", "Frequency", "Speed") and re.search(r"\bmotors?\b", u, re.I):
                    eq = "Motor"
                if fam == "Quantity" and not eq:
                    continue
                before = u[max(0, start - 80):start]
                qm = _QUAL.findall(before)
                qual = qm[-1].lower() if qm else ""
                group = "Electrical" if eq == "Motor" else FAMILY_GROUP.get(fam, "Other")
                name = _display_name(fam, eq, qual)
                if label and len(hits) == 1 and not MODAL.search(label) and len(label.split()) <= 8:
                    name = label[0].upper() + label[1:]
                conf = "High" if (strong or label) else ("Medium" if eq else "Low")
                rows.append({"Group": group, "Parameter": name, "Value": value, "Unit": unit,
                             "Applies to": eq, "Source text": u[:400], "Section": " > ".join(c["path"][-2:]),
                             "Pages": str(c["page"]), "Confidence": conf, "Method": "rules"})
    return finalize_design_rows(rows)


def finalize_design_rows(rows: list[dict]) -> list[dict]:
    """Merge duplicates (same parameter/value/unit), sort by group, add references."""
    merged: dict[tuple, dict] = {}
    for r in rows:
        key = (r["Parameter"].lower(), str(r["Value"]).lower(), str(r["Unit"]).lower(), str(r.get("Applies to", "")).lower())
        if key in merged:
            pg = set(str(merged[key]["Pages"]).split(", ")) | set(str(r["Pages"]).split(", "))
            merged[key]["Pages"] = ", ".join(sorted((p for p in pg if p), key=lambda x: int(x) if x.isdigit() else 0))
        else:
            merged[key] = dict(r)
    out = list(merged.values())
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    first_page = lambda r: int(re.findall(r"\d+", str(r["Pages"]) or "0")[0]) if re.findall(r"\d+", str(r["Pages"])) else 0
    out.sort(key=lambda r: (order.get(r["Group"], 99), r["Parameter"].split(":")[0], first_page(r)))
    for n, r in enumerate(out, 1):
        r["Ref"] = f"DI-{n:03d}"
    return out


def ai_design_inputs(pages, doc_name: str, progress=None) -> list[dict]:
    chunks, cur = [], ""
    for page, text in pages:
        block = f"\n[[Page {page}]]\n{text}\n"
        if cur and len(cur) + len(block) > AI_CHUNK_CHARS:
            chunks.append(cur)
            cur = ""
        cur += block
    if cur.strip():
        chunks.append(cur)
    rows: list[dict] = []
    for i, ch in enumerate(chunks, 1):
        if progress:
            progress(i, len(chunks))
        prompt = (
            "You are a mechanical / process engineer reading a technical specification. Extract every DESIGN INPUT: "
            "a concrete design value, size, rating or condition that engineers need, for example pipe size (NB/DN/OD, schedule, "
            "thickness), pump capacity, head, flow, pressure, temperature, velocity, slurry concentration, density, distances, "
            "elevations, storage times, quantities and duty/standby configuration, motor rating/voltage/speed, site and "
            "environmental data (ambient temperature, humidity, wind, seismic zone, altitude), materials of construction.\n"
            "Rules:\n- One item per value. Keep exact numbers and units; ranges as '1.8 - 2.5'.\n"
            "- 'parameter' = short engineering name (e.g. 'Slurry pump capacity', 'Ash slurry pipe size').\n"
            "- 'applies_to' = the equipment/system it belongs to.\n"
            "- 'group' must be one of: Hydraulic & Process, Piping & Mechanical, Electrical, Site & Environmental, Other.\n"
            "- 'text' = the source sentence, shortened if long. Do not invent values. Skip commercial terms (payment, LDs, warranty).\n"
            'Return ONLY JSON: {"items":[{"parameter":"","value":"","unit":"","applies_to":"","group":"","text":"",'
            '"section":"","page":<int>}]}\n\n'
            f"SPECIFICATION TEXT:\n{ch}")
        for d in _parse_design_items(_claude_text(prompt)):
            pg = d.get("page")
            grp = str(d.get("group", "") or "Other")
            rows.append({"Group": grp if grp in GROUP_ORDER else "Other",
                         "Parameter": str(d.get("parameter", "")).strip(), "Value": str(d.get("value", "")).strip(),
                         "Unit": str(d.get("unit", "") or "").strip(), "Applies to": str(d.get("applies_to", "") or "").strip(),
                         "Source text": str(d.get("text", "") or "")[:400], "Section": str(d.get("section", "") or ""),
                         "Pages": str(int(pg)) if str(pg).isdigit() else "", "Confidence": "AI", "Method": "ai"})
    return finalize_design_rows([r for r in rows if r["Parameter"] and r["Value"]])


def _parse_design_items(raw: str) -> list[dict]:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
        data = data.get("items", []) if isinstance(data, dict) else data
        return [d for d in data if isinstance(d, dict) and d.get("parameter")]
    except Exception:
        pass
    items, dec, pos = [], json.JSONDecoder(), raw.find("{")
    while pos != -1:
        try:
            obj, end = dec.raw_decode(raw, pos)
            items.append(obj)
            pos = raw.find("{", end)
        except json.JSONDecodeError:
            pos = raw.find("{", pos + 1)
    return [d for d in items if isinstance(d, dict) and d.get("parameter")]


def run_design_extraction(store: "DocumentStore", doc, use_ai: bool = False, progress=None) -> list[dict]:
    pages = get_pages(store, doc["id"])
    if not any(t.strip() for _, t in pages):
        raise RuntimeError("No readable text in this document (scanned PDF? it needs OCR first).")
    return ai_design_inputs(pages, doc["name"], progress) if use_ai else extract_design_inputs(pages, doc["name"])


DESIGN_COLS = ["Ref", "Group", "Parameter", "Value", "Unit", "Applies to", "Source text", "Section", "Pages", "Confidence"]


def design_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=DESIGN_COLS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def export_design_xlsx(rows: list[dict], target, title: str) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise RuntimeError(_pip_hint("openpyxl"))
    wb = Workbook()
    head_fill, head_font = PatternFill("solid", fgColor="0F4C81"), Font(bold=True, color="FFFFFF")
    safe = _xl_safe

    def sheet(name, data, first=False):
        ws = wb.active if first else wb.create_sheet()
        ws.title = re.sub(r"[\\/*?:\[\]]", "-", name)[:31]
        ws.append(DESIGN_COLS)
        for cell in ws[1]:
            cell.fill, cell.font = head_fill, head_font
        for r in data:
            ws.append([safe(r.get(c, "")) for c in DESIGN_COLS])
        for i, w in enumerate([9, 22, 34, 14, 12, 20, 70, 30, 10, 11], 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions

    sheet("Design Inputs", rows, first=True)
    for g in GROUP_ORDER:
        sub = [r for r in rows if r.get("Group") == g]
        if sub:
            sheet(g, sub)
    sm = wb.create_sheet("Summary", 0)
    sm["A1"], sm["A1"].font = _xl_safe(title), Font(bold=True, size=14)
    sm["A2"] = f"Generated {datetime.now():%d %b %Y %H:%M}. Check Low / Medium confidence rows against the document."
    sm.append([])
    sm.append(["Group", "Values"])
    for cell in sm[4]:
        cell.fill, cell.font = head_fill, head_font
    for g in GROUP_ORDER:
        sm.append([g, sum(1 for r in rows if r.get("Group") == g)])
    sm.append(["TOTAL", len(rows)])
    sm.column_dimensions["A"].width = 32
    wb.save(target)


def build_design_pdf(rows: list[dict], target, title: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        raise RuntimeError(_pip_hint("reportlab"))
    base, bold, unicode_ok = _register_pdf_font()
    clean = lambda s: (lambda t: t if unicode_ok else t.encode("latin-1", "replace").decode("latin-1"))(
        re.sub(r"[\x00-\x08\x0b-\x1f]", " ", str(s if s is not None else "")))
    esc = lambda s: html.escape(clean(s), quote=False)
    ss = getSampleStyleSheet()
    grey, blue = colors.HexColor("#6B7280"), colors.HexColor("#0F4C81")
    s_title = ParagraphStyle("T", parent=ss["Title"], fontName=bold, fontSize=20, leading=24, alignment=0,
                             textColor=colors.HexColor("#1F2937"))
    s_meta = ParagraphStyle("M", parent=ss["Normal"], fontName=base, fontSize=8.5, leading=12, textColor=grey)
    s_h = ParagraphStyle("H", parent=ss["Heading2"], fontName=bold, fontSize=13.5, leading=17,
                         spaceBefore=14, spaceAfter=4, textColor=blue)
    s_cell = ParagraphStyle("C", parent=ss["Normal"], fontName=base, fontSize=8.5, leading=11)
    s_cellb = ParagraphStyle("CB", parent=s_cell, fontName=bold)
    story = [Paragraph(esc(title), s_title),
             Paragraph(esc(f"Generated {datetime.now():%d %b %Y, %H:%M}  ·  {len(rows)} value(s)"), s_meta),
             Paragraph("Auto-extracted - verify against the source document before use.", s_meta), Spacer(1, 6)]
    for g in GROUP_ORDER:
        sub = [r for r in rows if r.get("Group") == g]
        if not sub:
            continue
        story.append(Paragraph(esc(f"{g}  ({len(sub)})"), s_h))
        data = [[Paragraph(h, s_cellb) for h in ("Ref", "Parameter", "Value", "Unit", "Applies to", "Page")]]
        for r in sub:
            data.append([Paragraph(esc(r.get("Ref", "")), s_cell), Paragraph(esc(r.get("Parameter", "")), s_cell),
                         Paragraph(f"<b>{esc(r.get('Value', ''))}</b>", s_cell), Paragraph(esc(r.get("Unit", "")), s_cell),
                         Paragraph(esc(r.get("Applies to", "")), s_cell), Paragraph(esc(r.get("Pages", "")), s_cell)])
        t = Table(data, colWidths=[16 * mm, 56 * mm, 24 * mm, 22 * mm, 34 * mm, 14 * mm], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EDF5")),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(t)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(base, 8)
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.drawString(18 * mm, 10 * mm, clean(title)[:90])
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(target, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=18 * mm, title=clean(title)).build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------- #
# 4e. Engineering lists: equipment / motor / valve / instrument / line list
#     >>> EDIT the vocabularies (EQUIP_NOUNS, _VALVE_TYPES, ...) to tune. <<<
# --------------------------------------------------------------------------- #
_NUMW = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
         "eight": "8", "nine": "9", "ten": "10"}
_MOD_STOP = set("""the a an of for with and or to in on by each all any every shall will be is are was were been being
supply supplied supplying furnish furnished provide provided include includes including included nos no set sets complete
along per as at from type types new following proposed existing minimum maximum total number qty quantity bidder contractor
vendor owner purchaser working standby stand-by duty spare spares its their this that these those suitable required
requirement requirements necessary such also not than more less between within which where when under over using use used
based driven operated""".split()) | set(_NUMW)
_NOT_TAG = {"IS", "IEC", "IEEE", "ASME", "ASTM", "API", "ANSI", "BS", "DIN", "EN", "ISO", "NFPA", "AWS", "AISC", "AWWA",
            "MSS", "NEMA", "TEMA", "JIS", "CEA", "IBC", "SCH", "DN", "NB", "PN", "IP", "RTD", "SG", "CL", "NPS"}

EQUIP_NOUNS = (r"pumps?|blowers?|compressors?|fans?|agitators?|mixers?|ejectors?|silos?|hoppers?|tanks?|sumps?|"
               r"thickeners?|cyclones?|strainers?|filters?|heat exchangers?|coolers?|cranes?|hoists?|conveyors?|"
               r"feeders?|screens?|vessels?|transformers?|dryers?|crushers?|mills?|dust collectors?|air receivers?|"
               r"receivers?|chillers?|dosing systems?")
_EQ_NOUN = re.compile(r"\b(" + EQUIP_NOUNS + r")\b(?!\s+(?:house|room|building|foundations?|station|shed|motors?))", re.I)

_VALVE_T = (r"gate|globe|check|non[- ]return|ball|butterfly|plug|knife[- ]?gate|diaphragm|pinch|control|safety|relief|"
            r"pressure relief|pressure reducing|pressure safety|solenoid|needle|foot|dump|isolation|slurry|drain|vent|"
            r"air release|sluice|slide gate|rotary|flap|swing check|dual plate|y-?type|angle|regulating|float|bypass")
_VALVE_UNIT = r"(?:(?:%s)[\s-]+)?(?:%s)" % (_VALVE_T, _VALVE_T)
_VALVE_RX = re.compile(r"\b(" + _VALVE_UNIT + r"(?:\s*(?:,|/|&|\band\b|\bor\b)\s*" + _VALVE_UNIT + r")*)\s+valves?\b", re.I)
_VALVE_ABBR = {"NRV": "Non-return valve", "PRV": "Pressure relief valve", "PSV": "Pressure safety valve",
               "SDV": "Shut-down valve", "MOV": "Motor operated valve", "PCV": "Pressure control valve",
               "FCV": "Flow control valve", "LCV": "Level control valve"}
_VALVE_ABBR_RX = re.compile(r"\b(" + "|".join(_VALVE_ABBR) + r")s?\b")

_INST_RX = re.compile(
    r"\b((?:differential pressure|pressure|level|flow|temperature|density|ph|conductivity|vibration|speed|position|"
    r"limit|proximity|torque|weight|load|radar level|ultrasonic level|magnetic flow|electromagnetic flow|"
    r"ultrasonic flow|slurry flow)[\s-]+(?:transmitters?|gauges?|switch(?:es)?|indicators?|sensors?|elements?|meters?|"
    r"analy[sz]ers?|controllers?|recorders?|detectors?|probes?|transducers?))\b|"
    r"\b(thermocouples?|rotameters?|orifice plates?|venturi|load cells?|flow ?meters?|annunciators?|"
    r"RTDs?|DPTs?)\b", re.I)

_PIPE_NOUN = re.compile(r"\b(pipelines?|piping|pipes?|headers?|lines?)\b", re.I)
_MAT_RX = re.compile(
    r"\b(?:SG iron|cast iron|ductile iron|stainless steel|carbon steel|mild steel|Ni-?hard|high chrome|alloy steel|"
    r"bronze|brass|gunmetal|rubber lined|ceramic lined|basalt lined|alumina lined|cast basalt|"
    r"API 5L(?:\s*Gr\.?\s*[A-Z0-9]+)?|IS ?1239|IS ?3589|IS ?2062|"
    r"(?-i:CI|WCB|WCC|LCB|CF8M|CF8|CF3M|MS|SS ?3\d{2}L?|SS ?4\d{2}|HDPE|GRP|FRP|PVC|CPVC|UPVC|"
    r"(?:ASTM )?A ?\d{3}(?:\s*Gr\.?\s*[A-Z0-9]+)?))\b", re.I)
_RATING_RX = re.compile(r"\b(?:ANSI\s*)?(?:class|cl\.?)\s*[-:]?\s*(?:150|300|600|900|1500|2500)\b|"
                        r"\b(?:150|300|600|900|1500|2500)\s*(?:#|lbs?\b)|\bPN\s*[-:]?\s*\d{1,3}\b", re.I)
_END_RX = re.compile(r"\b(double flanged|flanged|butt[- ]?weld(?:ed|ing)?|socket[- ]?weld|screwed|threaded|wafer|lug(?:ged)?(?: type)?)\b", re.I)
_ACT_RX = re.compile(r"\b(manual(?:ly)?|hand ?wheel|gear(?:ed)? operated|pneumatic(?:ally)?(?: operated| actuated)?|"
                     r"electric(?:al)?(?: actuator| operated)?|motori[sz]ed|motor operated|electro-?hydraulic|hydraulic(?:ally)?(?: operated)?|"
                     r"solenoid operated|self[- ]actuated|actuator)\b", re.I)
_SERVICE_RX = re.compile(r"\b((?:ash |fire |raw |service |dm |demineral(?:i[sz]ed)? |cooling |sealing |gland |potable |"
                         r"fluidi[sz]ing |compressed |instrument |bottom ash |fly ash )?(?:slurry|water|air|steam|oil|ash|drain|vent|sludge|condensate))\b", re.I)
_SIGNAL_RX = re.compile(r"4\s*[-–]\s*20\s*mA(?:\s*DC)?(?:\s*with HART|\s*HART)?|\bHART\b|\bProfibus(?: PA| DP)?\b|\bModbus(?: RTU| TCP)?\b|"
                        r"Foundation Fieldbus|potential[- ]free contacts?|dry contacts?|\bSPDT\b|\bDPDT\b|\b24\s*V\s*DC\b", re.I)
_ACCURACY_RX = re.compile(r"±\s*\d+(?:\.\d+)?\s*%(?:\s*(?:of\s*)?(?:FS|FSD|span|reading|full scale))?", re.I)
_PROT_RX = re.compile(r"\bIP\s*\d{2}[A-Z]?\b|\bNEMA\s*\d+X?\b|\bEx\s?[a-z]{1,2}\b(?:\s*II[ABC])?|flame ?proof|intrinsically safe|weather ?proof", re.I)
_RANGE_RX = re.compile(r"\b0\s*(?:-|–|to)\s*\d[\d.,]*\s*(?:kg/cm2(?:\s*\(g\))?|barg?|mbar|MPa|kPa|mm ?WC|mWC|°\s?C|m3/h|m/s|%|mm|m|TPH)?", re.I)
_TAG_RX = re.compile(r"\b[A-Z]{2,5}-\d{2,4}[A-Z]?\b")
_MOTOR_RX = re.compile(r"\bmotors?\b(?!\s*(?:operated|control|starters?|protection|actuated))", re.I)
_VOLT_RX = re.compile(r"\b\d{3,5}\s*V\b|\b\d+(?:\.\d+)?\s*kV\b")
_FREQ_RX = re.compile(r"\b\d{2}\s*Hz\b", re.I)
_RPM_RX = re.compile(r"\b\d{3,4}\s*rpm\b", re.I)
_POLE_RX = re.compile(r"\b(\d{1,2})\s*[- ]?poles?\b", re.I)
_ENCL_RX = re.compile(r"\bTEFC\b|\bTEAO\b|\bTEWAC\b|\bODP\b|flame ?proof|\bIP\s*\d{2}\b|\bEx\s?d\b")
_INSUL_RX = re.compile(r"\bclass\s*[BFH]\b(?:\s*insulation)?|insulation class\s*[BFH]", re.I)
_EFF_RX = re.compile(r"\bIE\s?[1-4]\b|\bEff\s?[123]\b")
_START_RX = re.compile(r"\bDOL\b|direct[- ]on[- ]line|star[- ]delta|soft[- ]starter|\bVFD\b|\bVSD\b|variable frequency drive|autotransformer", re.I)
_DUTY_S_RX = re.compile(r"\bS[1-9]\b|continuous duty|continuous rating")
_SIZE_MM = re.compile(r"\b(\d{2,4})\s*mm\s*(?:dia(?:meter)?|bore)\b", re.I)
_LINING_RX = re.compile(r"\b(?:ceramic|rubber|basalt|alumina|HDPE|cement mortar|polyurethane|PU|epoxy|glass|cast basalt)[\s-]*(?:lined|lining|coated|coating)\b|\bFBE\b|\b3LPE\b", re.I)
_FROM_TO = re.compile(r"\bfrom\s+(?:the\s+)?(.{3,60}?)\s+to\s+(?:the\s+)?(.{3,60}?)(?=[,.;]|\s+(?:shall|with|for|via|including|through)\b|$)", re.I)
_LENGTH_RX = re.compile(r"\b(?:length(?: of)?(?: approx\.?| about| around)?\s*[:=]?\s*)?(\d+(?:[.,]\d+)?)\s*(km|m)\s*(?:long|length)\b|\blength\s*(?:of|is|=|:)?\s*(?:approx\.?\s*)?(\d+(?:[.,]\d+)?)\s*(km|m)\b", re.I)
_INSULATION_RX = re.compile(r"\b(un-?insulated|non-?insulated|without insulation|no insulation|insulated|insulation|lagging|heat traced|tracing)\b", re.I)
_LINE_NO_RX = re.compile(r"\b(?:\d{1,3}\s*[\"”]?\s*-\s*)?([A-Z]{2,5}-\d{2,4}(?:-[A-Z0-9]{2,5})?)\b")
_QTY_RX = re.compile(r"\b(\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:nos?\b\.?|numbers?\b|sets?\b|off\b|units?\b)", re.I)
_NUM_ONLY_QTY = re.compile(r"^\s*(?:\(\s*)?(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:no\.?\s+)?$", re.I)

LIST_SPECS = {
    "Equipment List": {"code": "EQ", "key": ["Equipment"], "soft": [],
        "cols": ["Equipment", "Qty", "Configuration", "Capacity / flow", "Head", "Power", "Material"],
        "desc": "major equipment (pumps, blowers, tanks, silos, conveyors ...) with quantity, capacity, head, power, material"},
    "Motor List": {"code": "MT", "key": ["Driven equipment", "Rating"], "soft": ["Rating"],
        "cols": ["Driven equipment", "Qty", "Rating", "Voltage", "Frequency", "Speed", "Poles", "Enclosure / IP",
                 "Insulation", "Efficiency class", "Starting", "Duty"],
        "desc": "electric motors: driven equipment, kW/HP rating, voltage, frequency, speed, poles, enclosure/IP, insulation class, efficiency class, starting method, duty"},
    "Valve List": {"code": "VL", "key": ["Valve type", "Size", "Rating / class"], "soft": ["Size", "Rating / class"],
        "cols": ["Valve type", "Size", "Rating / class", "End connection", "Body material", "Actuation", "Service", "Qty", "Standard"],
        "desc": "valves: type, size, pressure class, end connection, body material, actuation, service, quantity, standard"},
    "Instrument List": {"code": "IN", "key": ["Instrument", "Application"], "soft": ["Application"],
        "cols": ["Instrument", "Application", "Range", "Output signal", "Accuracy", "Protection", "Material", "Qty", "Tag"],
        "desc": "field instruments: type, application, range, output signal, accuracy, protection class, wetted material, quantity, tag"},
    "Line List": {"code": "LN", "key": ["Line / service", "Size"], "soft": ["Size"],
        "cols": ["Line / service", "Line no.", "Size", "Material", "Schedule / thickness", "Lining / coating",
                 "Design pressure", "Design temperature", "From", "To", "Length", "Insulation"],
        "desc": "piping / pipelines: service, line number, size, material, schedule/thickness, lining, design pressure/temperature, from, to, length, insulation"},
}
TAIL_COLS = ["Source text", "Section", "Pages", "Confidence"]


def list_columns(name: str) -> list[str]:
    return ["Ref"] + LIST_SPECS[name]["cols"] + TAIL_COLS


# ---- small helpers --------------------------------------------------------- #
def _uniq_join(values, sep: str = ", ") -> str:
    seen, out = set(), []
    for v in values:
        v = re.sub(r"\s+", " ", str(v)).strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return sep.join(out)


def _find_all(rx, text: str) -> str:
    return _uniq_join(m.group(0) for m in rx.finditer(text))


def _singular(w: str) -> str:
    w = w.lower()
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _modifiers(prefix: str, n: int = 3) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-/&]*|\d+|[,;:()]", prefix)
    mods: list[str] = []
    for t in reversed(toks):
        if not t[0].isalpha() or t.lower() in _MOD_STOP:
            break
        mods.append(t)
        if len(mods) == n:
            break
    return list(reversed(mods))


def _name(mods: list[str], noun: str) -> str:
    parts = [m if m.isupper() else m.lower() for m in mods] + [_singular(noun)]
    s = " ".join(parts)
    return s[0].upper() + s[1:]


def _tag(text: str) -> str:
    return _uniq_join(m.group(0) for m in _TAG_RX.finditer(text) if m.group(0).split("-")[0] not in _NOT_TAG)


def _materials(text: str) -> str:
    return _find_all(_MAT_RX, text)


def _qty_config(w: str) -> tuple[str, str]:
    cfg = _find_all(_DUTY, w)
    m = _QTY_RX.search(w)
    qty = _NUMW.get(m.group(1).lower(), m.group(1)) if m else ""
    if not qty and cfg:
        a = re.match(r"(\d+)\s*[xX×]", cfg)
        b = re.match(r"(\d+)\s*(?:working|duty|running)\s*\+\s*(\d+)", cfg, re.I)
        qty = a.group(1) if a else (str(int(b.group(1)) + int(b.group(2))) if b else "")
    return qty, cfg


def _nums(w: str) -> list[tuple[str, str]]:
    out = []
    for m in NUM_UNIT.finditer(w):
        unit = m.group(2).strip()
        fam, _ = _family_for(unit, w[max(0, m.start() - 55):m.start()], w[m.end():m.end() + 25])
        out.append((fam, re.sub(r"\s+", " ", m.group(1)).strip() + " " + unit))
    return out


def _first_fam(nums, *fams) -> str:
    for fam, val in nums:
        if fam in fams:
            return val
    return ""


def _window(u: str, spans: list, i: int) -> str:
    lo = spans[i - 1][1] if i else 0
    hi = spans[i + 1][0] if i + 1 < len(spans) else len(u)
    return u[lo:hi]


def _size(w: str) -> str:
    m = _PIPE_SIZE.search(w)
    if m:
        return m.group(0).strip()
    m = _SIZE_MM.search(w)
    return (m.group(1) + " mm dia") if m else ""


def _conf(filled: int, cue: bool) -> str:
    return "High" if filled >= 3 else ("Medium" if filled >= 1 or cue else "Low")


_SUPPLY_CUE = re.compile(r"\b(supply|furnish|provide|include|scope)\b", re.I)


def _base_row(name: str, u: str, ctx: dict, conf: str) -> dict:
    row = {c: "" for c in LIST_SPECS[name]["cols"]}
    row.update({"Source text": u[:400], "Section": ctx["section"], "Pages": str(ctx["page"]), "Confidence": conf})
    return row


# ---- extractors (one sentence -> rows) ------------------------------------ #
def _x_equipment(u: str, ctx: dict) -> list[dict]:
    spans = [(m.start(), m.end(), _name(_modifiers(u[:m.start()]), m.group(1))) for m in _EQ_NOUN.finditer(u)]
    rows = []
    for i, (s, e, nm) in enumerate(spans):
        w = _window(u, spans, i)
        nums = _nums(w)
        qty, cfg = _qty_config(w)
        cap, head, pw = _first_fam(nums, "Flow", "Volume"), _first_fam(nums, "Head"), _first_fam(nums, "Power")
        mat = _materials(w)
        filled = sum(1 for v in (qty, cfg, cap, head, pw, mat) if v)
        r = _base_row("Equipment List", u, ctx, _conf(filled, bool(_SUPPLY_CUE.search(u))))
        r.update({"Equipment": nm, "Qty": qty, "Configuration": cfg, "Capacity / flow": cap, "Head": head,
                  "Power": pw, "Material": mat})
        rows.append(r)
    return rows


def _x_motor(u: str, ctx: dict) -> list[dict]:
    ms = list(_MOTOR_RX.finditer(u))
    if not ms:
        return []
    nums = _nums(u)
    rating = _first_fam(nums, "Power")
    voltage, freq, rpm = _find_all(_VOLT_RX, u), _find_all(_FREQ_RX, u), _find_all(_RPM_RX, u)
    pm = _POLE_RX.search(u)
    poles = (pm.group(1) + " pole") if pm else ""
    encl, insul = _find_all(_ENCL_RX, u), _find_all(_INSUL_RX, u)
    eff, start, duty = _find_all(_EFF_RX, u), _find_all(_START_RX, u), _find_all(_DUTY_S_RX, u)
    driven_spans = [(m.start(), m.end(), _name(_modifiers(u[:m.start()]), m.group(1))) for m in _EQ_NOUN.finditer(u)]
    pos = ms[0].start()
    driven = ""
    if driven_spans:
        driven = min(driven_spans, key=lambda t: (pos - t[1]) if t[1] <= pos else (t[0] - pos) + 20)[2]
    else:
        for m in _EQ_NOUN.finditer(ctx.get("lead", "") or ""):
            driven = _name(_modifiers(ctx["lead"][:m.start()]), m.group(1))
    qty, _ = _qty_config(u)
    filled = sum(1 for v in (rating, voltage, freq, rpm, poles, encl, insul, eff, start, duty) if v)
    if not (filled or qty):
        return []
    r = _base_row("Motor List", u, ctx, _conf(filled, False))
    r.update({"Driven equipment": driven, "Qty": qty, "Rating": rating, "Voltage": voltage, "Frequency": freq,
              "Speed": rpm, "Poles": poles, "Enclosure / IP": encl, "Insulation": insul,
              "Efficiency class": eff, "Starting": start, "Duty": duty})
    return [r]


def _x_valve(u: str, ctx: dict) -> list[dict]:
    spans = []                                   # (start, end, [type names])
    for m in _VALVE_RX.finditer(u):
        types = [t.strip() for t in re.split(r"\s*(?:,|/|&|\band\b|\bor\b)\s*", m.group(1), flags=re.I) if t.strip()]
        spans.append((m.start(), m.end(), [(t[0].upper() + t[1:].lower()) + " valve" for t in types]))
    for m in _VALVE_ABBR_RX.finditer(u):
        if not any(s <= m.start() < e for s, e, _ in spans):
            spans.append((m.start(), m.end(), [_VALVE_ABBR[m.group(1)]]))
    spans.sort()
    rows = []
    for i, (s, e, names) in enumerate(spans):
        w = _window(u, spans, i)
        size, rating = _size(w), _find_all(_RATING_RX, w)
        if not rating:
            for fam, val in _nums(w):
                if fam == "Pressure" and re.search(r"rat|class|design|pressure", w, re.I):
                    rating = val
                    break
        end, mat, act = _find_all(_END_RX, w), _materials(w), _find_all(_ACT_RX, w)
        svc = _uniq_join(m.group(0).strip().capitalize() for m in list(_SERVICE_RX.finditer(w))[:2])
        qty, _ = _qty_config(w)
        std = _uniq_join(m.group(1) + " " + m.group(2).strip(" .,-:;") for m in STD_RX.finditer(w))
        filled = sum(1 for v in (size, rating, end, mat, act, svc, qty, std) if v)
        for nm in names:
            r = _base_row("Valve List", u, ctx, _conf(filled, bool(_SUPPLY_CUE.search(u))))
            r.update({"Valve type": nm, "Size": size, "Rating / class": rating, "End connection": end,
                      "Body material": mat, "Actuation": act, "Service": svc, "Qty": qty, "Standard": std})
            rows.append(r)
    return rows


def _x_instrument(u: str, ctx: dict) -> list[dict]:
    spans = []
    for m in _INST_RX.finditer(u):
        t = re.sub(r"\s+", " ", m.group(0)).strip()
        words = t.split(" ")
        if not words[-1].isupper():
            words[-1] = _singular(words[-1])
        t = " ".join(words)
        spans.append((m.start(), m.end(), t[0].upper() + t[1:] if t.isupper() else t[0].upper() + t[1:].lower()))
    spans = [sp for i, sp in enumerate(spans)
             if not (sp[2].lower() in ("rtd", "dpt") and i + 1 < len(spans) and spans[i + 1][0] - sp[1] < 14)]
    rows = []
    last_qty = ""
    for i, (s, e, nm) in enumerate(spans):
        w = _window(u, spans, i)
        rng, sig, acc = _find_all(_RANGE_RX, w), _find_all(_SIGNAL_RX, w), _find_all(_ACCURACY_RX, w)
        prot, mat, tag = _find_all(_PROT_RX, w), _materials(w), _tag(w)
        qty, _ = _qty_config(w)
        if not qty and i and re.fullmatch(r"[\s,]*(?:and|&|or)?[\s,]*", u[spans[i - 1][1]:s], re.I):
            qty = last_qty                                    # '1 set of A, B and C' -> same qty for the group
        last_qty = qty
        app = ""
        eq = [(m2.start(), _name(_modifiers(u[:m2.start()]), m2.group(1))) for m2 in _EQ_NOUN.finditer(u)]
        if eq:
            app = min(eq, key=lambda t: abs(t[0] - s))[1]
        filled = sum(1 for v in (rng, sig, acc, prot, mat, tag, qty) if v)
        r = _base_row("Instrument List", u, ctx, _conf(filled, bool(_SUPPLY_CUE.search(u))))
        r.update({"Instrument": nm, "Application": app, "Range": rng, "Output signal": sig, "Accuracy": acc,
                  "Protection": prot, "Material": mat, "Qty": qty, "Tag": tag})
        rows.append(r)
    return rows


def _x_line(u: str, ctx: dict) -> list[dict]:
    spans = [(m.start(), m.end(), m) for m in _PIPE_NOUN.finditer(u)]
    if not spans:
        return []
    size = _size(u)
    ft = _FROM_TO.search(u)
    ln = _LINE_NO_RX.search(u)
    if not (size or ft or (ln and ln.group(1).split("-")[0] not in _NOT_TAG)):
        return []
    # use the first pipe noun that has modifiers (service); otherwise the first one
    pick = None
    for s, e, m in spans:
        mods = _modifiers(u[:s])
        if mods or m.group(1).lower() != "line":
            pick = (s, e, m, mods)
            break
    if pick is None:
        return []
    s, e, m, mods = pick
    noun = m.group(1).lower()
    noun = "pipe" if noun == "pipes" else _singular(noun) if noun not in ("piping",) else "piping"
    name = _name(mods, noun)
    nums = _nums(u)
    dp = ""
    mdp = re.search(r"design pressure\D{0,30}?(\d+(?:\.\d+)?\s*(?:kg/cm2(?:\s*\(g\))?|barg?|MPa|kPa))", u, re.I)
    dp = mdp.group(1) if mdp else ""
    dt = ""
    mdt = re.search(r"design temperature\D{0,30}?(\d+(?:\.\d+)?\s*(?:°\s?C|deg\.? ?C))", u, re.I)
    dt = mdt.group(1) if mdt else ""
    sched = _find_all(_SCHEDULE, u)
    for fam, val in nums:
        if fam == "Thickness":
            sched = _uniq_join([sched, val + " thk"]) if sched else val + " thk"
            break
    ml = _LENGTH_RX.search(u)
    length = (ml.group(1) + " " + ml.group(2) if ml and ml.group(1) else (ml.group(3) + " " + ml.group(4) if ml else ""))
    ins = ""
    mi = _INSULATION_RX.search(u)
    if mi:
        ins = "None" if re.match(r"un|non|without|no ", mi.group(1), re.I) else "Insulated"
    mats = _materials(u)
    lining = _find_all(_LINING_RX, u)
    frm, to = (ft.group(1).strip(), ft.group(2).strip()) if ft else ("", "")
    line_no = ln.group(1) if ln and ln.group(1).split("-")[0] not in _NOT_TAG else ""
    filled = sum(1 for v in (size, mats, sched, lining, dp, dt, frm, length, ins, line_no) if v)
    r = _base_row("Line List", u, ctx, _conf(filled, False))
    r.update({"Line / service": name, "Line no.": line_no, "Size": size, "Material": mats, "Schedule / thickness": sched,
              "Lining / coating": lining, "Design pressure": dp, "Design temperature": dt, "From": frm, "To": to,
              "Length": length, "Insulation": ins})
    return [r]


for _n, _f in (("Equipment List", _x_equipment), ("Motor List", _x_motor), ("Valve List", _x_valve),
               ("Instrument List", _x_instrument), ("Line List", _x_line)):
    LIST_SPECS[_n]["extract"] = _f


# ---- merge + finalize ------------------------------------------------------ #
_RANK = {"Low": 0, "Medium": 1, "High": 2, "AI": 3}


def _merge_into(dst: dict, src: dict, cols: list[str]) -> None:
    for c in cols:
        a, b = str(dst.get(c, "")), str(src.get(c, ""))
        if b and not a:
            dst[c] = b
        elif b and b.lower() not in a.lower() and len(a) < 120:
            dst[c] = a + " / " + b
    pg = set(str(dst["Pages"]).split(", ")) | set(str(src["Pages"]).split(", "))
    dst["Pages"] = ", ".join(sorted((p for p in pg if p), key=lambda x: int(x) if x.isdigit() else 0))
    if src["Source text"] and src["Source text"] not in dst["Source text"] and len(dst["Source text"]) < 500:
        dst["Source text"] += "  ||  " + src["Source text"]
    if _RANK.get(src["Confidence"], 0) > _RANK.get(dst["Confidence"], 0):
        dst["Confidence"] = src["Confidence"]


def finalize_list(name: str, rows: list[dict]) -> list[dict]:
    spec = LIST_SPECS[name]
    cols, key, soft = spec["cols"], spec["key"], spec["soft"]
    merged: dict[tuple, dict] = {}
    for r in rows:
        k = tuple(str(r.get(c, "")).lower() for c in key)
        if k in merged:
            _merge_into(merged[k], r, cols)
        else:
            merged[k] = dict(r)
    for sc in soft:                                           # fold rows with a blank soft-key into the unique match
        hard = [c for c in key if c != sc]
        for k, r in list(merged.items()):
            if k not in merged or str(r.get(sc, "")):
                continue
            cands = [(k2, r2) for k2, r2 in merged.items() if k2 != k and str(r2.get(sc, ""))
                     and all(str(r2.get(c, "")).lower() == str(r.get(c, "")).lower() for c in hard)]
            if len(cands) == 1:
                _merge_into(cands[0][1], r, cols)
                del merged[k]
    out = list(merged.values())
    if name == "Equipment List":
        out = [r for r in out if r["Confidence"] != "Low"]   # bare mentions with no data are noise
        for r in list(out):                                  # 'Pump' -> fold into the one specific '... pump'
            nm = str(r["Equipment"]).lower()
            if " " not in nm:
                cands = [o for o in out if o is not r and str(o["Equipment"]).lower().endswith(" " + nm)]
                if len(cands) == 1:
                    _merge_into(cands[0], r, cols)
                    out.remove(r)
    out.sort(key=lambda r: (str(r.get(key[0], "")).lower(), str(r.get("Pages", "")).zfill(4)))
    for n, r in enumerate(out, 1):
        r["Ref"] = f"{spec['code']}-{n:03d}"
    return out


def extract_lists(pages, doc_name: str, which: list[str] | None = None) -> dict[str, list[dict]]:
    which = which or list(LIST_SPECS)
    raw: dict[str, list[dict]] = {n: [] for n in which}
    for c in iter_clauses(strip_repeating_lines(pages)):
        units = [c["text"]] if (c["bullet"] or c["row"]) else _sentences(c["text"])
        ctx = {"section": " > ".join(c["path"][-2:]), "page": c["page"], "lead": c["lead"]}
        for u in units:
            u = u.strip()
            if len(u) < 8:
                continue
            for n in which:
                raw[n].extend(LIST_SPECS[n]["extract"](u, ctx))
    return {n: finalize_list(n, rows) for n, rows in raw.items()}


def ai_lists(pages, doc_name: str, which: list[str] | None = None, progress=None) -> dict[str, list[dict]]:
    which = which or list(LIST_SPECS)
    chunks, cur = [], ""
    for page, text in pages:
        block = f"\n[[Page {page}]]\n{text}\n"
        if cur and len(cur) + len(block) > AI_CHUNK_CHARS:
            chunks.append(cur)
            cur = ""
        cur += block
    if cur.strip():
        chunks.append(cur)
    out: dict[str, list[dict]] = {n: [] for n in which}
    steps, step = len(chunks) * len(which), 0
    for n in which:
        spec = LIST_SPECS[n]
        for ch in chunks:
            step += 1
            if progress:
                progress(step, steps)
            prompt = (
                f"You are an engineer preparing a {n} from a technical specification (pages marked [[Page N]]).\n"
                f"Extract every item that belongs in this list: {spec['desc']}.\n"
                "Rules:\n- One item per distinct equipment/size/type. Keep exact numbers, units, ratings and standards.\n"
                "- Fill only values stated in the text (leave others empty strings). Do not invent anything.\n"
                "- Do not include commercial terms. 'text' = short source sentence; 'section' = clause/heading if visible.\n"
                f"- Fields: {', '.join(spec['cols'])}\n"
                'Return ONLY JSON: {"items":[{' + ", ".join(f'"{c}":""' for c in spec["cols"])
                + ',"text":"","section":"","page":<int>}]}\n\n'
                f"SPECIFICATION TEXT:\n{ch}")
            for d in _parse_items_generic(_claude_text(prompt), spec["cols"][0]):
                pg = d.get("page")
                row = {c: str(d.get(c, "") or "").strip() for c in spec["cols"]}
                row.update({"Source text": str(d.get("text", "") or "")[:400], "Section": str(d.get("section", "") or ""),
                            "Pages": str(int(pg)) if str(pg).isdigit() else "", "Confidence": "AI"})
                out[n].append(row)
    return {n: finalize_list(n, [r for r in rows if r.get(LIST_SPECS[n]["cols"][0])]) for n, rows in out.items()}


def _parse_items_generic(raw: str, need_key: str) -> list[dict]:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
        data = data.get("items", []) if isinstance(data, dict) else data
        return [d for d in data if isinstance(d, dict) and d.get(need_key)]
    except Exception:
        pass
    items, dec, pos = [], json.JSONDecoder(), raw.find("{")
    while pos != -1:
        try:
            obj, end = dec.raw_decode(raw, pos)
            items.append(obj)
            pos = raw.find("{", end)
        except json.JSONDecodeError:
            pos = raw.find("{", pos + 1)
    return [d for d in items if isinstance(d, dict) and d.get(need_key)]


def run_list_extraction(store: "DocumentStore", doc, which=None, use_ai: bool = False, progress=None):
    pages = get_pages(store, doc["id"])
    if not any(t.strip() for _, t in pages):
        raise RuntimeError("No readable text in this document (scanned PDF? it needs OCR first).")
    return ai_lists(pages, doc["name"], which, progress) if use_ai else extract_lists(pages, doc["name"], which)


# ---- exports --------------------------------------------------------------- #
def lists_csv(name: str, rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list_columns(name), extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def export_lists_xlsx(lists: dict[str, list[dict]], target, title: str) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise RuntimeError(_pip_hint("openpyxl"))
    wb = Workbook()
    head_fill, head_font = PatternFill("solid", fgColor="0F4C81"), Font(bold=True, color="FFFFFF")
    sm = wb.active
    sm.title = "Summary"
    sm["A1"], sm["A1"].font = _xl_safe(title), Font(bold=True, size=14)
    sm["A2"] = f"Generated {datetime.now():%d %b %Y %H:%M}. Check Low / Medium confidence rows against the document."
    sm.append([])
    sm.append(["List", "Items"])
    for cell in sm[4]:
        cell.fill, cell.font = head_fill, head_font
    for name, rows in lists.items():
        sm.append([name, len(rows)])
    sm.column_dimensions["A"].width = 28
    for name, rows in lists.items():
        cols = list_columns(name)
        ws = wb.create_sheet(name[:31])
        ws.append(cols)
        for cell in ws[1]:
            cell.fill, cell.font = head_fill, head_font
        for r in rows:
            ws.append([_xl_safe(r.get(c, "")) for c in cols])
        for i, c in enumerate(cols, 1):
            ws.column_dimensions[get_column_letter(i)].width = 70 if c == "Source text" else 30 if c in ("Section",) else \
                26 if c in ("Driven equipment", "Equipment", "Line / service", "Valve type", "Instrument") else 16
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "C2"
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
    wb.save(target)


def build_lists_pdf(lists: dict[str, list[dict]], target, title: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        raise RuntimeError(_pip_hint("reportlab"))
    base, bold, unicode_ok = _register_pdf_font()
    clean = lambda s: (lambda t: t if unicode_ok else t.encode("latin-1", "replace").decode("latin-1"))(
        re.sub(r"[\x00-\x08\x0b-\x1f]", " ", str(s if s is not None else "")))
    esc = lambda s: html.escape(clean(s), quote=False)
    ss = getSampleStyleSheet()
    grey, blue = colors.HexColor("#6B7280"), colors.HexColor("#0F4C81")
    s_title = ParagraphStyle("T", parent=ss["Title"], fontName=bold, fontSize=20, leading=24, alignment=0,
                             textColor=colors.HexColor("#1F2937"))
    s_meta = ParagraphStyle("M", parent=ss["Normal"], fontName=base, fontSize=8.5, leading=12, textColor=grey)
    s_h = ParagraphStyle("H", parent=ss["Heading2"], fontName=bold, fontSize=13.5, leading=17,
                         spaceBefore=14, spaceAfter=4, textColor=blue, keepWithNext=1)
    s_cell = ParagraphStyle("C", parent=ss["Normal"], fontName=base, fontSize=7.6, leading=9.6)
    s_cellb = ParagraphStyle("CB", parent=s_cell, fontName=bold)
    page = landscape(A4)
    width = page[0] - 30 * mm
    story = [Paragraph(esc(title), s_title),
             Paragraph(esc(f"Generated {datetime.now():%d %b %Y, %H:%M}"), s_meta),
             Paragraph("Auto-extracted - verify against the source document before use.", s_meta), Spacer(1, 6)]
    first = True
    for name, rows in lists.items():
        if not rows:
            continue
        first = False
        story.append(Paragraph(esc(f"{name}  ({len(rows)})"), s_h))
        cols = ["Ref"] + [c for c in LIST_SPECS[name]["cols"] if any(str(r.get(c, "")).strip() for r in rows)] + ["Pages"]
        weights = [0.95 if c in ("Ref", "Pages") else 0.6 if c == "Qty" else 1.6 if c in ("Driven equipment", "Equipment", "Line / service",
                                                                         "Valve type", "Instrument") else 1.0 for c in cols]
        widths = [width * w / sum(weights) for w in weights]
        data = [[Paragraph(esc(c), s_cellb) for c in cols]]
        for r in rows:
            data.append([Paragraph(esc(r.get(c, "")), s_cell) for c in cols])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EDF5")),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        story.append(t)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(base, 8)
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.drawString(15 * mm, 8 * mm, clean(title)[:100])
        canvas.drawRightString(page[0] - 15 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(target, pagesize=page, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                      bottomMargin=16 * mm, title=clean(title)).build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------- #
# 5. Command line interface
# --------------------------------------------------------------------------- #
def cmd_index(a, store: DocumentStore):
    store.add_path(Path(a.path), force=a.force)


def cmd_list(a, store: DocumentStore):
    docs = store.list_docs()
    if not docs:
        print("Nothing indexed yet. Try:  python document_reader.py index ./docs")
        return
    for d in docs:
        print(f"{d['name']}  —  {d['n_pages']} pages, {d['n_chars']:,} chars\n    {d['path']}")


def cmd_search(a, store: DocumentStore):
    hits = store.search(" ".join(a.query), a.n, a.any, a.doc)
    if not hits:
        print("No results.")
        return
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.doc_name}  (page/section {h.page})")
        print(f"   {h.snippet}\n")


def cmd_summarize(a, store: DocumentStore):
    doc = store.find_doc(a.file)
    if not doc:
        sys.exit(f"Document not found in index: {a.file}  (run 'index' first)")
    text = store.doc_text(doc["id"])
    if a.page:
        text = store.page_text(doc["id"], a.page) or sys.exit("No such page.")
    print(f"=== {doc['name']} ===\n")
    if a.ai:
        print(ai_summary(text, a.focus))
    else:
        print(extractive_summary(text, a.sentences, a.focus))
        print("\nKeywords:", ", ".join(keywords(text)))


def cmd_show(a, store: DocumentStore):
    doc = store.find_doc(a.file)
    if not doc:
        sys.exit("Document not found.")
    if a.page:
        print(store.page_text(doc["id"], a.page) or "No such page.")
    else:
        print(store.doc_text(doc["id"]))


def cmd_remove(a, store: DocumentStore):
    print("Removed." if store.remove(a.file) else "Not found.")


def cmd_report(a, store: DocumentStore):
    names = []
    for f in a.files:
        p = Path(f)
        if p.exists():
            print(f"- {p.name}: {store.add_file(p)}")
            names.append(p.name)
        else:
            d = store.find_doc(f)
            if d:
                names.append(d["name"])
            else:
                print(f"- {f}: not found", file=sys.stderr)
    kws = parse_keywords(a.keywords)
    if not kws:
        sys.exit("Give at least one keyword, e.g.  -k \"torque, safety\"")
    matches = find_keyword_matches(store, kws, names or None, a.context, a.max)
    out = a.output or "keyword_report.pdf"
    build_pdf_report(matches, kws, names or [d["name"] for d in store.list_docs()], out, a.title)
    print(f"{len(matches)} match(es) -> {out}")
    if a.csv:
        csv_path = str(Path(out).with_suffix(".csv"))
        Path(csv_path).write_bytes(matches_to_csv(matches))
        print(f"data -> {csv_path}")


def cmd_scope(a, store: DocumentStore):
    p = Path(a.file)
    if p.exists():
        print(f"- {p.name}: {store.add_file(p)}")
        doc = store.find_doc(p.name)
    else:
        doc = store.find_doc(a.file)
    if not doc:
        sys.exit(f"Document not found: {a.file}")
    items, standards = run_scope_extraction(
        store, doc, a.ai, lambda i, n: print(f"  AI pass {i}/{n}...", file=sys.stderr))
    base = a.output or f"{Path(doc['name']).stem}_scope"
    export_scope_xlsx(items, standards, base + ".xlsx", f"Scope Extraction - {doc['name']}", [doc["name"]])
    build_scope_pdf(items, standards, base + ".pdf", f"Scope Extraction - {doc['name']}", [doc["name"]])
    Path(base + ".csv").write_bytes(scope_csv(items))
    for name, lst in group_items(items).items():
        if lst:
            print(f"{len(lst):4d}  {name}")
    print(f"{len(standards):4d}  codes & standards referenced")
    print(f"-> {base}.xlsx / .pdf / .csv")


def cmd_inputs(a, store: DocumentStore):
    p = Path(a.file)
    if p.exists():
        print(f"- {p.name}: {store.add_file(p)}")
        doc = store.find_doc(p.name)
    else:
        doc = store.find_doc(a.file)
    if not doc:
        sys.exit(f"Document not found: {a.file}")
    rows = run_design_extraction(store, doc, a.ai, lambda i, n: print(f"  AI pass {i}/{n}...", file=sys.stderr))
    base = a.output or f"{Path(doc['name']).stem}_design_inputs"
    title = f"Design Inputs - {doc['name']}"
    export_design_xlsx(rows, base + ".xlsx", title)
    build_design_pdf(rows, base + ".pdf", title)
    Path(base + ".csv").write_bytes(design_csv(rows))
    for g in GROUP_ORDER:
        n = sum(1 for r in rows if r["Group"] == g)
        if n:
            print(f"{n:4d}  {g}")
    print(f"-> {base}.xlsx / .pdf / .csv")


def cmd_lists(a, store: DocumentStore):
    p = Path(a.file)
    if p.exists():
        print(f"- {p.name}: {store.add_file(p)}")
        doc = store.find_doc(p.name)
    else:
        doc = store.find_doc(a.file)
    if not doc:
        sys.exit(f"Document not found: {a.file}")
    which = None
    if a.only:
        wanted = [w.strip().lower() for w in a.only.split(",") if w.strip()]
        which = [n for n in LIST_SPECS if any(w in n.lower() for w in wanted)]
        if not which:
            sys.exit("--only must name: equipment, motor, valve, instrument, line")
    lists = run_list_extraction(store, doc, which, a.ai, lambda i, n: print(f"  AI pass {i}/{n}...", file=sys.stderr))
    base = a.output or f"{Path(doc['name']).stem}_lists"
    title = f"Engineering Lists - {doc['name']}"
    export_lists_xlsx(lists, base + ".xlsx", title)
    build_lists_pdf(lists, base + ".pdf", title)
    for name, rows in lists.items():
        Path(f"{base}_{name.split()[0].lower()}.csv").write_bytes(lists_csv(name, rows))
        print(f"{len(rows):4d}  {name}")
    print(f"-> {base}.xlsx / .pdf / _<list>.csv")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Index, search and summarize technical documents.")
    p.add_argument("--db", default=DB_PATH, help=f"database file (default: {DB_PATH})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index", help="index a file or folder")
    s.add_argument("path")
    s.add_argument("--force", action="store_true", help="re-index even if unchanged")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("list", help="list indexed documents")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("search", help="full-text search")
    s.add_argument("query", nargs="+")
    s.add_argument("-n", type=int, default=10, help="max results")
    s.add_argument("--any", action="store_true", help="match ANY word instead of ALL")
    s.add_argument("--doc", help="limit to documents whose name contains this")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("summarize", help="summarize a document")
    s.add_argument("file")
    s.add_argument("-s", "--sentences", type=int, default=7, help="sentences (extractive mode)")
    s.add_argument("--focus", help="topic/question to center the summary on")
    s.add_argument("--page", type=int, help="summarize only this page/section")
    s.add_argument("--ai", action="store_true", help="use Claude for the summary")
    s.set_defaults(fn=cmd_summarize)

    s = sub.add_parser("show", help="print stored text")
    s.add_argument("file")
    s.add_argument("--page", type=int)
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("remove", help="remove a document from the index")
    s.add_argument("file")
    s.set_defaults(fn=cmd_remove)
    s = sub.add_parser("report", help="extract keyword data from documents into a PDF")
    s.add_argument("files", nargs="+", help="files to read (indexed automatically) or indexed names")
    s.add_argument("-k", "--keywords", required=True, help='e.g. "torque, safety, pressure relief valve"')
    s.add_argument("-o", "--output", help="PDF file name (default keyword_report.pdf)")
    s.add_argument("--context", type=int, default=1, help="sentences of context around a match")
    s.add_argument("--max", type=int, default=25, help="max matches per keyword")
    s.add_argument("--title", default="Keyword Extraction Report")
    s.add_argument("--csv", action="store_true", help="also write the data as CSV")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("scope", help="sort a specification into scope lists (engineering, supply, design inputs, ...)")
    s.add_argument("file", help="specification file (indexed automatically) or an indexed name")
    s.add_argument("-o", "--output", help="output file prefix (default: <name>_scope)")
    s.add_argument("--ai", action="store_true", help="use Claude (more accurate; needs ANTHROPIC_API_KEY)")
    s.set_defaults(fn=cmd_scope)

    s = sub.add_parser("inputs", help="extract design inputs (pipe size, pump capacity, head, ...) to a table")
    s.add_argument("file")
    s.add_argument("-o", "--output", help="output file prefix")
    s.add_argument("--ai", action="store_true", help="use Claude (more accurate; needs ANTHROPIC_API_KEY)")
    s.set_defaults(fn=cmd_inputs)

    s = sub.add_parser("lists", help="equipment / motor / valve / instrument / line lists -> xlsx/pdf/csv")
    s.add_argument("file")
    s.add_argument("-o", "--output", help="output file prefix")
    s.add_argument("--only", help="comma list: equipment,motor,valve,instrument,line")
    s.add_argument("--ai", action="store_true", help="use Claude (more accurate; needs ANTHROPIC_API_KEY)")
    s.set_defaults(fn=cmd_lists)

    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = DocumentStore(args.db)
    try:
        args.fn(args, store)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")


# --------------------------------------------------------------------------- #
# 6. Web dashboard (Streamlit)
# --------------------------------------------------------------------------- #
DASH_CSS = """
<style>
:root { --b900:#0A2A66; --b800:#0D47A1; --b600:#1976D2; --b400:#42A5F5; --b100:#E3F0FF; --ink:#0F172A; --mut:#64748B; }
.stApp { background: linear-gradient(180deg,#E8F1FD 0%, #F5F9FF 38%, #FFFFFF 100%); color: var(--ink); }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }

.hero { background: linear-gradient(120deg,#0A2A66 0%,#0D47A1 38%,#1976D2 72%,#42A5F5 100%);
        border-radius: 18px; padding: 30px 36px; color:#fff; margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(13,71,161,.28); position: relative; overflow: hidden; }
.hero:after { content:""; position:absolute; right:-60px; top:-80px; width:300px; height:300px; border-radius:50%;
              background: radial-gradient(circle, rgba(255,255,255,.22), rgba(255,255,255,0) 70%); }
.hero-title { font-size: 2rem; font-weight: 750; letter-spacing:-.4px; line-height:1.15; }
.hero-sub { margin-top:6px; font-size: 1rem; opacity:.92; }
.hero-tags { margin-top: 14px; }
.hero-tags span { display:inline-block; background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
                  padding: 3px 12px; border-radius: 999px; font-size:.78rem; margin-right:8px; backdrop-filter: blur(2px); }

.kpi { background: linear-gradient(#fff,#fff) padding-box, linear-gradient(90deg,#0D47A1,#42A5F5) border-box;
       border: 2px solid transparent; border-radius: 14px; padding: 14px 18px;
       box-shadow: 0 4px 14px rgba(13,71,161,.10); }
.kpi-label { color: var(--mut); font-size:.74rem; text-transform: uppercase; letter-spacing:.9px; font-weight:600; }
.kpi-value { font-size: 1.85rem; font-weight: 750; line-height:1.2;
             background: linear-gradient(90deg,#0A2A66,#1976D2); -webkit-background-clip:text; background-clip:text;
             -webkit-text-fill-color: transparent; }

div[data-baseweb="tab-list"] { gap: 6px; background: #fff; padding: 6px; border-radius: 14px;
                               box-shadow: 0 3px 12px rgba(13,71,161,.10); border: 1px solid #DCE8F8; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display:none !important; }
button[role="tab"] { border-radius: 10px !important; padding: 8px 18px !important; font-weight: 600 !important;
                     color: var(--b800) !important; background: transparent !important; }
button[role="tab"][aria-selected="true"] { background: linear-gradient(120deg,#0D47A1,#1976D2 60%,#42A5F5) !important;
                     color:#fff !important; box-shadow: 0 4px 12px rgba(25,118,210,.35); }
button[role="tab"][aria-selected="true"] p { color:#fff !important; }
button[role="tab"]:hover { background: var(--b100) !important; }
button[role="tab"][aria-selected="true"]:hover { background: linear-gradient(120deg,#0D47A1,#1976D2 60%,#42A5F5) !important; }

button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(120deg,#0D47A1,#1976D2 60%,#42A5F5) !important; color:#fff !important; border:0 !important;
    border-radius: 10px !important; font-weight: 600 !important; box-shadow: 0 4px 12px rgba(25,118,210,.35); }
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); transform: translateY(-1px); }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] { border-radius:10px !important; border:1px solid #BBD3F2 !important;
    color: var(--b800) !important; background:#fff !important; font-weight:600 !important; }

[data-testid="stFileUploader"] section { border: 2px dashed #90B8EA; border-radius: 14px; background: linear-gradient(180deg,#F4F9FF,#EAF3FF); }
div[data-testid="stExpander"] { border: 1px solid #D5E4F8; border-radius: 12px; background:#fff; box-shadow: 0 2px 8px rgba(13,71,161,.06); }
div[data-testid="stExpander"] summary { font-weight: 600; color: var(--b800); }
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] { border-radius: 12px; overflow: hidden; border:1px solid #D5E4F8; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #D5E4F8; border-radius:12px; padding:10px 14px; box-shadow:0 2px 8px rgba(13,71,161,.06); }
div[data-testid="stMetricValue"] { color: var(--b800); }
div[data-baseweb="input"], div[data-baseweb="select"] > div, textarea { border-radius: 10px !important; }
h2, h3 { color: var(--b900); letter-spacing: -.2px; }
.sec-title { font-size:1.15rem; font-weight:700; color: var(--b900); margin: 4px 0 2px; }
.sec-sub { color: var(--mut); font-size:.9rem; margin-bottom: 10px; }
</style>
"""


def run_dashboard() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Document Intelligence", page_icon="📘", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(DASH_CSS, unsafe_allow_html=True)
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_types = [e.lstrip(".") for e in sorted(SUPPORTED_EXTS)]

    # ----------------------------------------------------------------- hero --
    st.markdown(
        '<div class="hero"><div class="hero-title">Document Intelligence</div>'
        '<div class="hero-sub">Read specifications, extract design inputs and scope lists, search and summarise - '
        'then export to Excel and PDF.</div>'
        '<div class="hero-tags"><span>Design inputs</span><span>Scope of work &amp; supply</span>'
        '<span>Equipment, motor, valve &amp; line lists</span><span>Keyword reports</span><span>Full-text search</span><span>AI summaries</span></div></div>',
        unsafe_allow_html=True)

    with st.expander("⚙️ Settings"):
        s1, s2 = st.columns(2)
        db_path = s1.text_input("Database file", DB_PATH)
        key_in = s2.text_input("Anthropic API key (only for Claude AI mode)", type="password",
                               value=os.environ.get("ANTHROPIC_API_KEY", ""),
                               help="Kept in memory while the app runs. Not saved to disk.")
        if key_in:
            os.environ["ANTHROPIC_API_KEY"] = key_in.strip()
    store = DocumentStore(db_path)

    # ------------------------------------------------------------------ KPIs --
    docs = store.list_docs()
    n_chunks = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    kpis = [("Documents", f"{len(docs):,}"), ("Pages / sections", f"{sum(d['n_pages'] for d in docs):,}"),
            ("Characters read", f"{sum(d['n_chars'] for d in docs):,}"), ("Searchable chunks", f"{n_chunks:,}")]
    for col, (label, value) in zip(st.columns(4), kpis):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
                     unsafe_allow_html=True)
    st.write("")

    tab_di, tab_scope, tab_el, tab_rep, tab_search, tab_sum, tab_lib = st.tabs(
        ["Design Inputs", "Scope Lists", "Engineering Lists", "Keyword Report", "Search", "Summarize", "Library"])

    # ------------------------------------------------------- design inputs --
    with tab_di:
        st.markdown("Extracts **design input values** from the whole document - pipe size, pump capacity, head, "
                    "pressure, temperature, velocity, motor rating, distances, site data … - as an editable table.")
        di_up = st.file_uploader("Upload specification", type=file_types, key="di_up")
        di_pick = st.selectbox("…or choose an already-indexed document", ["—"] + [d["name"] for d in docs], key="di_pick")
        di_mode = st.radio("Method", ["Offline rules (free, no internet)", "Claude AI (more accurate, needs API key)"],
                           horizontal=True, key="di_mode")
        di_ai = di_mode.startswith("Claude")
        if di_ai and not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("Add your API key under ⚙️ Settings (top of page), or use the offline method.")

        if st.button("Extract design inputs", type="primary", key="di_go"):
            doc_row = None
            if di_up is not None:
                target = UPLOAD_DIR / di_up.name
                target.write_bytes(di_up.getbuffer())
                try:
                    with st.spinner(f"Reading {di_up.name}…"):
                        store.add_file(target)
                    doc_row = store.find_doc(di_up.name)
                except Exception as e:
                    st.error(f"{di_up.name}: {e}")
            elif di_pick != "—":
                doc_row = store.find_doc(di_pick)
            else:
                st.warning("Upload a file or pick an indexed document.")
            if doc_row is not None:
                try:
                    bar = st.progress(0.0, text="Analysing…")
                    cb = lambda i, n: bar.progress(i / n, text=f"AI pass {i}/{n}…")
                    with st.spinner("Finding design values…"):
                        rows_ = run_design_extraction(store, doc_row, di_ai, cb)
                    bar.empty()
                    st.session_state["di"] = {"name": doc_row["name"], "rows": rows_}
                    st.session_state.pop("di_editor", None)
                except RuntimeError as e:
                    st.error(str(e))

        di = st.session_state.get("di")
        if di:
            st.divider()
            st.subheader(di["name"])
            ddf = pd.DataFrame(di["rows"], columns=DESIGN_COLS)
            if ddf.empty:
                st.warning("No design values found. If this is a scanned PDF it needs OCR first.")
            else:
                gsel = st.multiselect("Filter by group", GROUP_ORDER, key="di_groups")
                st.caption(f"{len(ddf)} value(s). Edit cells, add or delete rows - downloads use your edited table. "
                           "Check Low / Medium confidence rows.")
                edited = st.data_editor(ddf, num_rows="dynamic", use_container_width=True, hide_index=True,
                                        key="di_editor", disabled=["Ref"])
                out_rows = edited.fillna("").astype(str).to_dict("records")
                if gsel:
                    out_rows_dl = [r for r in out_rows if r["Group"] in gsel]
                else:
                    out_rows_dl = out_rows
                ttl = f"Design Inputs - {di['name']}"
                try:
                    xb, pb = io.BytesIO(), io.BytesIO()
                    export_design_xlsx(out_rows_dl, xb, ttl)
                    build_design_pdf(out_rows_dl, pb, ttl)
                    stem = Path(di["name"]).stem
                    e1, e2, e3 = st.columns(3)
                    e1.download_button("⬇️ Excel", xb.getvalue(), file_name=f"{stem}_design_inputs.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       type="primary")
                    e2.download_button("⬇️ PDF", pb.getvalue(), file_name=f"{stem}_design_inputs.pdf",
                                       mime="application/pdf")
                    e3.download_button("⬇️ CSV", design_csv(out_rows_dl), file_name=f"{stem}_design_inputs.csv",
                                       mime="text/csv")
                except RuntimeError as e:
                    st.error(str(e))

    # ------------------------------------------------------ scope extraction --
    with tab_scope:
        st.markdown("Upload a specification and the app reads it **clause by clause**, then sorts it into "
                    "Engineering Scope, Supply Scope, Design Inputs, Performance Requirements, Exclusions, "
                    "Codes & Standards … and exports **Excel + PDF**.")
        sc_up = st.file_uploader("Upload specification", type=file_types, key="scope_up")
        sc_pick = st.selectbox("…or choose an already-indexed document", ["—"] + [d["name"] for d in docs])
        mode = st.radio("Method", ["Offline rules (free, no internet)", "Claude AI (more accurate, needs API key)"],
                        horizontal=True)
        use_ai = mode.startswith("Claude")
        if use_ai and not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("No API key set - add it under ⚙️ Settings (top of page) or use the offline method.")

        if st.button("Extract scope lists", type="primary", key="scope_go"):
            doc_row = None
            if sc_up is not None:
                target = UPLOAD_DIR / sc_up.name
                target.write_bytes(sc_up.getbuffer())
                try:
                    with st.spinner(f"Reading {sc_up.name}…"):
                        store.add_file(target)
                    doc_row = store.find_doc(sc_up.name)
                except Exception as e:
                    st.error(f"{sc_up.name}: {e}")
            elif sc_pick != "—":
                doc_row = store.find_doc(sc_pick)
            else:
                st.warning("Upload a file or pick an indexed document.")
            if doc_row is not None:
                try:
                    bar = st.progress(0.0, text="Analysing…")
                    cb = lambda i, n: bar.progress(i / n, text=f"AI pass {i}/{n}…")
                    with st.spinner("Sorting clauses into scope lists…"):
                        its, stds = run_scope_extraction(store, doc_row, use_ai, cb)
                    bar.empty()
                    ttl = f"Scope Extraction - {doc_row['name']}"
                    xb, pb = io.BytesIO(), io.BytesIO()
                    export_scope_xlsx(its, stds, xb, ttl, [doc_row["name"]])
                    build_scope_pdf(its, stds, pb, ttl, [doc_row["name"]])
                    st.session_state["scope"] = {
                        "name": doc_row["name"], "rows": scope_rows(its), "stds": stds,
                        "xlsx": xb.getvalue(), "pdf": pb.getvalue(), "csv": scope_csv(its)}
                except RuntimeError as e:
                    st.error(str(e))

        sc = st.session_state.get("scope")
        if sc:
            st.divider()
            st.subheader(sc["name"])
            sdf = pd.DataFrame(sc["rows"])
            k1, k2, k3 = st.columns(3)
            k1.metric("Items extracted", len(sdf))
            k2.metric("Categories used", sdf["Category"].nunique() if not sdf.empty else 0)
            k3.metric("Codes & standards", len(sc["stds"]))
            b1, b2, b3 = st.columns(3)
            stem = Path(sc["name"]).stem
            b1.download_button("⬇️ Excel (one sheet per list)", sc["xlsx"], file_name=f"{stem}_scope.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
            b2.download_button("⬇️ PDF report", sc["pdf"], file_name=f"{stem}_scope.pdf", mime="application/pdf")
            b3.download_button("⬇️ CSV (all items)", sc["csv"], file_name=f"{stem}_scope.csv", mime="text/csv")
            if sdf.empty:
                st.warning("Nothing could be extracted. If this is a scanned PDF it needs OCR first.")
            else:
                st.caption("Rule-based items carry a confidence level - check Low / Medium items against the document.")
                counts = sdf.groupby("Category", sort=False).size().reset_index(name="Items")
                order = {c["name"]: n for n, c in enumerate(CATEGORIES)}
                counts = counts.sort_values("Category", key=lambda s: s.map(order))
                st.dataframe(counts, use_container_width=True, hide_index=True)
                for cat in counts["Category"]:
                    sub = sdf[sdf["Category"] == cat]
                    cols = (["Ref", "Parameter", "Value", "Requirement", "Section", "Page", "Confidence"]
                            if cat == "Design Inputs" else ["Ref", "Requirement", "Section", "Page", "Confidence"])
                    with st.expander(f"{cat} — {len(sub)}"):
                        st.dataframe(sub[cols], use_container_width=True, hide_index=True)
            if sc["stds"]:
                with st.expander(f"Codes & standards referenced — {len(sc['stds'])}"):
                    st.dataframe(pd.DataFrame(sc["stds"]), use_container_width=True, hide_index=True)

    # ---------------------------------------------------- engineering lists --
    with tab_el:
        st.markdown('<div class="sec-title">Engineering Lists</div>'
                    '<div class="sec-sub">Builds equipment, motor, valve, instrument and line lists from the document. '
                    'Every table is editable, then export to Excel / PDF / CSV.</div>', unsafe_allow_html=True)
        el_up = st.file_uploader("Upload specification", type=file_types, key="el_up")
        el_pick = st.selectbox("…or choose an already-indexed document", ["—"] + [d["name"] for d in docs], key="el_pick")
        el_sel = st.multiselect("Lists to build", list(LIST_SPECS), default=list(LIST_SPECS), key="el_sel")
        el_mode = st.radio("Method", ["Offline rules (free, no internet)", "Claude AI (more accurate, needs API key)"],
                           horizontal=True, key="el_mode")
        el_ai = el_mode.startswith("Claude")
        if el_ai and not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("No API key set - add it under ⚙️ Settings (top of page) or use the offline method.")

        if st.button("Generate lists", type="primary", key="el_go"):
            doc_row = None
            if el_up is not None:
                target = UPLOAD_DIR / el_up.name
                target.write_bytes(el_up.getbuffer())
                try:
                    with st.spinner(f"Reading {el_up.name}…"):
                        store.add_file(target)
                    doc_row = store.find_doc(el_up.name)
                except Exception as e:
                    st.error(f"{el_up.name}: {e}")
            elif el_pick != "—":
                doc_row = store.find_doc(el_pick)
            else:
                st.warning("Upload a file or pick an indexed document.")
            if doc_row is not None and not el_sel:
                st.warning("Select at least one list.")
            elif doc_row is not None:
                try:
                    bar = st.progress(0.0, text="Analysing…")
                    cb = lambda i, n: bar.progress(i / n, text=f"AI pass {i}/{n}…")
                    with st.spinner("Building lists…"):
                        result = run_list_extraction(store, doc_row, el_sel, el_ai, cb)
                    bar.empty()
                    st.session_state["el"] = {"name": doc_row["name"], "lists": result}
                    for n_ in LIST_SPECS:
                        st.session_state.pop(f"el_ed_{n_}", None)
                except RuntimeError as e:
                    st.error(str(e))

        el = st.session_state.get("el")
        if el:
            st.divider()
            st.subheader(el["name"])
            top = st.container()
            names_ = list(el["lists"])
            edited_lists: dict[str, list[dict]] = {}
            for tab_, name_ in zip(st.tabs([f"{n} ({len(el['lists'][n])})" for n in names_]), names_):
                with tab_:
                    cols_ = list_columns(name_)
                    df_ = pd.DataFrame(el["lists"][name_], columns=cols_)
                    if df_.empty:
                        st.info(f"No {name_.lower()} items found in this document.")
                        edited_lists[name_] = []
                        continue
                    st.caption("Edit cells, add or delete rows. Check Low / Medium confidence rows against the document.")
                    ed_ = st.data_editor(df_, num_rows="dynamic", use_container_width=True, hide_index=True,
                                         key=f"el_ed_{name_}", disabled=["Ref"])
                    edited_lists[name_] = ed_.fillna("").astype(str).to_dict("records")
                    st.download_button(f"⬇️ CSV - {name_}", lists_csv(name_, edited_lists[name_]),
                                       file_name=f"{Path(el['name']).stem}_{name_.split()[0].lower()}_list.csv",
                                       mime="text/csv", key=f"el_csv_{name_}")
            with top:
                try:
                    ttl = f"Engineering Lists - {el['name']}"
                    xb, pb = io.BytesIO(), io.BytesIO()
                    export_lists_xlsx(edited_lists, xb, ttl)
                    build_lists_pdf(edited_lists, pb, ttl)
                    stem = Path(el["name"]).stem
                    c1_, c2_ = st.columns(2)
                    c1_.download_button("⬇️ Excel - all lists (one sheet each)", xb.getvalue(),
                                        file_name=f"{stem}_engineering_lists.xlsx", type="primary", key="el_xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    c2_.download_button("⬇️ PDF - all lists", pb.getvalue(), file_name=f"{stem}_engineering_lists.pdf",
                                        mime="application/pdf", key="el_pdf")
                except RuntimeError as e:
                    st.error(str(e))

    # ------------------------------------------------------ keyword report --
    with tab_rep:
        st.markdown("**1.** Upload documents  **2.** Type the keywords you need  "
                    "**3.** Get the matching data as a PDF (and CSV).")
        ups = st.file_uploader("Upload documents", accept_multiple_files=True,
                               type=file_types, key="rep_up")
        existing = [d["name"] for d in docs]
        pick = st.multiselect("Also use already-indexed documents", existing)
        kw_raw = st.text_area("Keywords (comma or new line separated)", height=100,
                              placeholder="torque, pressure relief valve, safety, 45 Nm")
        o1, o2, o3 = st.columns(3)
        context = o1.slider("Sentences of context around each match", 0, 3, 1)
        max_kw = o2.number_input("Max matches per keyword", 1, 500, 25)
        report_title = o3.text_input("Report title", "Keyword Extraction Report")

        if st.button("Generate report", type="primary"):
            kws = parse_keywords(kw_raw)
            names = list(pick)
            for up in ups or []:
                target = UPLOAD_DIR / up.name
                target.write_bytes(up.getbuffer())
                try:
                    with st.spinner(f"Reading {up.name}…"):
                        store.add_file(target)
                    names.append(up.name)
                except Exception as e:
                    st.error(f"{up.name}: {e}")
            names = list(dict.fromkeys(names))
            if not kws:
                st.warning("Enter at least one keyword.")
            elif not names and not store.list_docs():
                st.warning("Upload at least one document.")
            else:
                searched = names or [d["name"] for d in store.list_docs()]
                found = find_keyword_matches(store, kws, searched, context, int(max_kw))
                try:
                    buf = io.BytesIO()
                    build_pdf_report(found, kws, searched, buf, report_title or "Keyword Extraction Report")
                    st.session_state["report"] = {
                        "keywords": kws, "docs": searched, "rows": matches_to_rows(found),
                        "pdf": buf.getvalue(), "csv": matches_to_csv(found),
                    }
                except RuntimeError as e:
                    st.error(str(e))

        rep = st.session_state.get("report")
        if rep:
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Keywords", len(rep["keywords"]))
            m2.metric("Matches found", len(rep["rows"]))
            m3.metric("Documents searched", len(rep["docs"]))

            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download PDF report", rep["pdf"], file_name="keyword_report.pdf",
                               mime="application/pdf", type="primary")
            d2.download_button("⬇️ Download data (CSV)", rep["csv"], file_name="keyword_data.csv",
                               mime="text/csv")

            df = pd.DataFrame(rep["rows"])
            if df.empty:
                st.warning("No matches found. Try other keywords, or check that the file has "
                           "selectable text (scanned PDFs need OCR).")
            else:
                summary = (df.groupby("Keyword", sort=False)
                             .agg(Matches=("Text", "count"), Documents=("Document", "nunique"))
                             .reset_index())
                st.dataframe(summary, use_container_width=True, hide_index=True)
                for kw in rep["keywords"]:
                    sub = df[df["Keyword"] == kw]
                    with st.expander(f"{kw} — {len(sub)} match(es)",
                                     expanded=len(rep["keywords"]) <= 3):
                        if sub.empty:
                            st.caption("No matches.")
                        for _, r in sub.iterrows():
                            st.markdown(f"**{r['Document']}** · page/section {r['Page']}")
                            st.write(r["Text"])
                            st.divider()

    # -------------------------------------------------------------- search --
    with tab_search:
        if not docs:
            st.info("Index some documents first (Library tab, or upload in the Design Inputs / Scope Lists tabs).")
        else:
            with st.form("search_form"):
                q = st.text_input("Search", placeholder='e.g. pressure relief valve   or   "exact phrase"')
                a, b, c = st.columns([2, 1, 1])
                doc_filter = a.selectbox("Document", ["All documents"] + [d["name"] for d in docs])
                limit = b.number_input("Max results", 1, 100, 15)
                any_word = c.checkbox("Match ANY word")
                go = st.form_submit_button("Search", type="primary")

            if go and q.strip():
                hits = store.search(q, int(limit), any_word,
                                    None if doc_filter == "All documents" else doc_filter)
                st.caption(f"{len(hits)} result(s)")
                if not hits:
                    st.warning("No matches. Try fewer words or 'Match ANY word'.")
                for h in hits:
                    with st.container(border=True):
                        st.markdown(f"**{h.doc_name}** · page/section {h.page}")
                        st.markdown(h.snippet.replace("[[", "**:orange[").replace("]]", "]**"))

    # ----------------------------------------------------------- summarize --
    with tab_sum:
        if not docs:
            st.info("Index some documents first.")
        else:
            left, right = st.columns([1, 2])
            with left:
                choice = st.selectbox("Document", [d["name"] for d in docs])
                doc = store.find_doc(choice)
                page = st.number_input(f"Page/section (0 = whole document, max {doc['n_pages']})",
                                       0, doc["n_pages"], 0)
                focus = st.text_input("Focus on (optional)", placeholder="e.g. safety requirements")
                use_ai = st.toggle("Use Claude (needs ANTHROPIC_API_KEY)")
                n_sent = st.slider("Sentences (offline mode)", 3, 25, 8, disabled=use_ai)
                run = st.button("Generate summary", type="primary")

            with right:
                if run:
                    text = (store.page_text(doc["id"], page) if page else store.doc_text(doc["id"])) or ""
                    if not text.strip():
                        st.warning("No text found for that selection.")
                    else:
                        try:
                            with st.spinner("Summarizing…"):
                                summary = (ai_summary(text, focus or None) if use_ai
                                           else extractive_summary(text, n_sent, focus or None))
                            st.subheader(choice)
                            st.markdown(summary)
                            if not use_ai:
                                st.caption("Keywords: " + ", ".join(keywords(text)))
                            st.download_button("⬇️ Download summary", summary,
                                               file_name=f"{Path(choice).stem}_summary.txt")
                        except RuntimeError as e:
                            st.error(str(e))

    # ------------------------------------------------------------- library --
    with tab_lib:
        st.markdown('<div class="sec-title">Add documents</div>'
                    '<div class="sec-sub">Upload files or index a whole folder. Unchanged files are skipped.</div>',
                    unsafe_allow_html=True)
        for msg in st.session_state.pop("lib_msg", []):
            st.write(msg)
        l1, l2 = st.columns(2)
        with l1:
            lib_up = st.file_uploader("Upload files", accept_multiple_files=True, type=file_types, key="lib_up")
            force = st.checkbox("Re-index even if unchanged", key="lib_force")
            if lib_up and st.button("Index uploaded files", type="primary", key="lib_idx"):
                msgs = []
                for up in lib_up:
                    target = UPLOAD_DIR / up.name
                    target.write_bytes(up.getbuffer())
                    try:
                        msgs.append(f"**{up.name}**: {store.add_file(target, force)}")
                    except Exception as e:
                        msgs.append(f"**{up.name}**: failed - {e}")
                st.session_state["lib_msg"] = msgs
                st.rerun()
        with l2:
            folder = st.text_input("Index a folder", placeholder="C:/docs or ./docs", key="lib_folder")
            if folder and st.button("Index folder", key="lib_fold"):
                p = Path(folder).expanduser()
                if not p.exists():
                    st.error("Folder not found.")
                else:
                    files = store._collect(p)
                    bar = st.progress(0.0)
                    msgs = []
                    for i, f in enumerate(files, 1):
                        try:
                            msgs.append(f"**{f.name}**: {store.add_file(f, force)}")
                        except Exception as e:
                            msgs.append(f"**{f.name}**: failed - {e}")
                        bar.progress(i / max(len(files), 1))
                    st.session_state["lib_msg"] = msgs or ["No supported files found."]
                    st.rerun()

        st.divider()
        st.markdown('<div class="sec-title">Indexed documents</div>', unsafe_allow_html=True)
        if not docs:
            st.info("Nothing indexed yet. Upload a file above, or use any tab - uploads there are indexed automatically.")
        else:
            df = pd.DataFrame([{
                "Name": d["name"], "Pages": d["n_pages"], "Characters": d["n_chars"],
                "Indexed": d["indexed"], "Path": d["path"],
            } for d in docs])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("Name")["Characters"], color="#1976D2")
            r1, r2 = st.columns([3, 1])
            to_remove = r1.selectbox("Remove a document from the index", [""] + list(df["Name"]))
            r2.write("")
            r2.write("")
            if to_remove and r2.button("Remove", key="lib_rm"):
                store.remove(to_remove)
                st.rerun()


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_in_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    main()