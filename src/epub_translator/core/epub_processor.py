import logging
import re
import threading
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString

from ..constants import SKIP_TAGS, XHTML_EXTENSIONS
from .model_loader import Translator
from .translator import translate_in_batches

log = logging.getLogger(__name__)
_XML_DECL_RE = re.compile(r"(<\?xml[^?]*\?>)\s*")
_DOCTYPE_RE = re.compile(r"(<!DOCTYPE[^>]*>)\s*")


def _should_skip(element):
    for parent in element.parents:
        if not hasattr(parent, "name"):
            continue
        if parent.name in SKIP_TAGS:
            return True
        classes = parent.get("class") or []
        if any("code" in c.lower() or "source" in c.lower() for c in classes):
            return True
    return False


def _collect_nodes(soup):
    nodes, texts = [], []
    for el in soup.find_all(string=True):
        if isinstance(el, Comment) or _should_skip(el):
            continue
        text = el.strip()
        if len(text) > 2:
            nodes.append(el)
            texts.append(text)
    return nodes, texts


def _split_header(raw):
    header = []
    body = raw
    xml_match = _XML_DECL_RE.match(body)
    if xml_match:
        header.append(xml_match.group(1))
        body = body[xml_match.end():]
    doctype_match = _DOCTYPE_RE.match(body)
    if doctype_match:
        header.append(doctype_match.group(1))
        body = body[doctype_match.end():]
    return header, body


def translate_xhtml(raw, tr: Translator, system_prompt, batch_size, max_new_tokens, num_beams):
    header, body = _split_header(raw)
    soup = BeautifulSoup(body, "html.parser")
    nodes, texts = _collect_nodes(soup)

    if not texts:
        return ("\n".join(header) + "\n\n" + str(soup)) if header else str(soup)

    translations = translate_in_batches(
        tr, texts, system_prompt, batch_size, max_new_tokens, num_beams
    )

    for node, translated in zip(nodes, translations):
        leading = " " if str(node).startswith(" ") else ""
        trailing = " " if str(node).endswith(" ") else ""
        node.replace_with(NavigableString(leading + translated + trailing))

    serialized = str(soup)
    return ("\n".join(header) + "\n\n" + serialized) if header else serialized


def _is_xhtml(name):
    return Path(name).suffix.lower() in XHTML_EXTENSIONS


def translate_epub(
    source: Path,
    tr: Translator,
    system_prompt,
    batch_size,
    max_new_tokens,
    num_beams,
    cancel_event: threading.Event | None = None,
    on_progress=None,
) -> Path | None:
    destination = source.parent / f"{source.stem}_tr.epub"
    cancel = cancel_event or threading.Event()

    with zipfile.ZipFile(source, "r") as zin:
        members = zin.namelist()
        total = sum(1 for n in members if _is_xhtml(n))
        done = 0

        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name in members:
                if cancel.is_set():
                    log.info("Çeviri iptal edildi.")
                    return None

                info = zin.getinfo(name)
                payload = zin.read(name)

                if _is_xhtml(name):
                    done += 1
                    log.info("[%3d/%d] Çevriliyor: %s", done, total, name)
                    translated = translate_xhtml(
                        payload.decode("utf-8"),
                        tr,
                        system_prompt,
                        batch_size,
                        max_new_tokens,
                        num_beams,
                    )
                    payload = translated.encode("utf-8")
                    if on_progress:
                        on_progress(done, total)

                zout.writestr(info, payload)

    return destination
