"""Corpus loading: fictional bank FAQ (JSONL) + own-words guideline notes (MD).

Documents are the unit of retrieval. FAQ items map 1:1 to a document; the
guideline notes are markdown documents that can either be indexed whole
(baseline) or split at their `## ` section headings (chunked mode), which is
the design change measured by the eval iteration (see DECISIONS.md).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str
    kind: str  # "faq" | "note"
    category: str = ""
    section: str = ""  # filled for chunked notes, e.g. "NOTE-001-A"

    def display_id(self) -> str:
        return self.section if self.section else self.id


@dataclass
class Corpus:
    docs: list[Doc] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.docs)


def load_faq(path) -> list[Doc]:
    docs: list[Doc] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Index the question together with the answer: user questions
            # share vocabulary with the FAQ question field.
            text = f"{item['question']}\n{item['answer']}"
            docs.append(
                Doc(
                    id=item["id"],
                    title=item["product"],
                    text=text,
                    kind="faq",
                    category=item.get("category", ""),
                )
            )
    return docs


def _split_note_sections(md_text: str) -> list[tuple[str, str]]:
    """Split a notes markdown file into (section_title, section_body) pairs."""
    sections: list[tuple[str, str]] = []
    current_title, current_lines = "", []
    for line in md_text.splitlines():
        if line.startswith("## "):
            if current_title:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif current_title:
            current_lines.append(line)
    if current_title:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def load_notes(notes_dir, chunked: bool | None = None) -> list[Doc]:
    """Load guideline summary notes.

    chunked=None  -> read RAG_CHUNK_NOTES env (default "none")
    chunked=False -> whole file = one document (baseline)
    chunked=True  -> each `## ` section = one document (id NOTE-001-A style)
    """
    if chunked is None:
        chunked = os.environ.get("RAG_CHUNK_NOTES", "none") == "sections"
    docs: list[Doc] = []
    for md_path in sorted(notes_dir.glob("*.md")):
        note_id = "NOTE-001" if "governance" in md_path.stem else "NOTE-002"
        raw = md_path.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip() if raw else md_path.stem
        if not chunked:
            # Strip the blockquote disclaimer from retrieval text? No — keep
            # the whole file: the disclaimer is part of the document.
            docs.append(Doc(id=note_id, title=title, text=raw, kind="note"))
            continue
        for sec_title, body in _split_note_sections(raw):
            if not body:
                continue
            # Stable section id: NOTE-001-A (letter code from title prefix
            # like "NOTE-001-A: ..." written in the heading itself).
            sec_id = sec_title.split(":")[0].strip()
            docs.append(
                Doc(
                    id=note_id,
                    title=title,
                    text=f"{sec_title}\n{body}",
                    kind="note",
                    section=sec_id,
                )
            )
    return docs


def load_corpus(faq_path, notes_dir, chunked: bool | None = None) -> Corpus:
    docs = load_faq(faq_path) + load_notes(notes_dir, chunked=chunked)
    return Corpus(docs=docs)
