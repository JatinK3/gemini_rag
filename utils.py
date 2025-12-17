import os
import re
import io
import time
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

# Try import google-genai client (if installed & configured)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

logger = logging.getLogger("echo_rag_chat")

# ---------------------
# Helper functions
# ---------------------
def _store_session_key_for(store):
    return "filestore_rows_for_" + (
        getattr(store, "name", None)
        or getattr(store, "display_name", None)
        or "none"
    )

def _render_with_cursor(text: str) -> str:
    # Blink every ~500ms
    blink_on = int(time.time() * 2) % 2 == 0
    return text + (" " if blink_on else "")

def _looks_like_structured_output(text: str) -> bool:
    """
    Detects JSON or Markdown table intent.
    Streaming tables/JSON should NOT be rendered mid-stream.
    """
    stripped = text.strip()
    return any([
        "\n|" in text,                # Markdown table (mid-text)
        stripped.startswith("|"),     # Markdown table (start)
        stripped.startswith("{"),     # JSON
        stripped.startswith("["),     # JSON array
        "```" in text                 # Code block
    ])

def classify_gemini_error(err: Exception) -> dict:
    """
    Returns structured info about Gemini errors.
    """
    msg = str(err)
    lower = msg.lower()

    result = {
        "type": None,            # quota | rate | auth | unknown
        "tier": None,            # free | paid | unknown
        "retryable": False,
        "model": None,
        "raw": msg,
    }

    # --- RESOURCE_EXHAUSTED / QUOTA ---
    if "resource_exhausted" in lower or "quota exceeded" in lower:
        result["type"] = "quota"

        # Detect free tier explicitly
        if "free_tier" in lower:
            result["tier"] = "free"

        # Extract model if present
        for m in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.0"]:
            if m in lower:
                result["model"] = m

        # Retry info
        result["retryable"] = "retry in" in lower and "limit: 0" not in lower
        return result

    # --- RATE LIMIT (429 but not quota exhausted) ---
    if "429" in lower or "too many requests" in lower:
        result["type"] = "rate"
        result["retryable"] = True
        return result

    # --- AUTH / API KEY ---
    if any(k in lower for k in ["api key", "permission", "unauthorized"]):
        result["type"] = "auth"
        return result

    result["type"] = "unknown"
    return result

def _safe_to_plain(o: Any) -> Any:
    if o is None:
        return None
    if isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, dict):
        return {k: _safe_to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe_to_plain(x) for x in o]

    # Pydantic / dataclasses / SDK objects — prefer model_dump
    for fn in ("model_dump", "to_dict", "as_dict"):
        if hasattr(o, fn):
            try:
                return _safe_to_plain(getattr(o, fn)())
            except Exception:
                pass

    # Fallback to __dict__
    if hasattr(o, "__dict__"):
        try:
            return _safe_to_plain(vars(o))
        except Exception:
            pass

    try:
        return str(o)
    except Exception:
        return repr(o)

def extract_text(resp) -> Optional[str]:
    """Best-effort text extraction from generate_content-like response."""
    try:
        if getattr(resp, "text", None):
            return resp.text
    except Exception:
        pass
    pieces = []
    for attr in ("candidates", "outputs", "output", "generations", "choices"):
        seq = getattr(resp, attr, None)
        if not seq:
            continue
        if not isinstance(seq, (list, tuple)):
            seq = [seq]
        for item in seq:
            item_plain = _safe_to_plain(item)
            if isinstance(item_plain, dict):
                for k in ("text", "content", "display_text"):
                    if k in item_plain and isinstance(item_plain[k], str):
                        pieces.append(item_plain[k])
    raw = getattr(resp, "_raw", None) or getattr(resp, "raw", None)
    if raw:
        raw_plain = _safe_to_plain(raw)
        if isinstance(raw_plain, str):
            pieces.append(raw_plain)
    pieces = [p.strip() for p in pieces if isinstance(p, str) and p.strip()]
    if pieces:
        return "\n\n".join(pieces)
    return None

def _extract_page_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r'---\s*PAGE\s*(\d+)\s*---', text, re.I)
    if m:
        return m.group(1)
    m2 = re.search(r'\bpage[\s:]*([0-9]{1,3})\b', text, re.I)
    if m2:
        return m2.group(1)
    return None

def pretty_doc_name(raw_name: Optional[str]) -> str:
    if not raw_name:
        return "Unknown document"
    name = str(raw_name).strip()
    name = re.sub(r'^\d+_+', '', name)
    name = name.split('/')[-1].split('\\')[-1]
    name = re.sub(r'[_]+', ' ', name).strip()
    if len(name) <= 60:
        name = name.title()
    return name

def _normalize_chunk(ch, resp_for_lookup=None):
    if not ch:
        return {"text": "", "doc": None, "start": None, "end": None, "score": None}
    try:
        if not isinstance(ch, dict):
            ch = _safe_to_plain(ch) or {}
    except Exception:
        ch = _safe_to_plain(ch) or {}

    text = None
    doc = None
    start = None
    end = None
    score = None

    for container_key in ("retrieved_context", "retrieved_contexts", "segment", "retrieval_segment", "rag_chunk", "content", "retrieval"):
        seg = ch.get(container_key) if isinstance(ch, dict) else None
        if isinstance(seg, dict):
            text = text or seg.get("text") or seg.get("content")
            doc = doc or seg.get("document_name") or seg.get("file_name") or seg.get("source")
            start = start or seg.get("start_index") or seg.get("start")
            end = end or seg.get("end_index") or seg.get("end")

    if text is None:
        for k in ("text", "rag_chunk", "chunk_text", "retrieved_text", "content"):
            v = ch.get(k)
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break

    for k in ("document_name", "file_name", "document", "source", "file", "display_name", "title", "name"):
        if doc is None:
            v = ch.get(k)
            if isinstance(v, str) and v.strip():
                doc = v.strip()

    if isinstance(ch.get("score"), (int, float)):
        score = ch.get("score")
    else:
        score = ch.get("score") or ch.get("confidence") or ch.get("confidence_score")

    if start is None:
        start = ch.get("start_index") or ch.get("start")
    if end is None:
        end = ch.get("end_index") or ch.get("end")

    return {"text": (text or "").strip(), "doc": doc, "start": start, "end": end, "score": score}

def get_top_grounding_snippets(resp, top_n=3):
    snippets = []
    try:
        candidates = getattr(resp, "candidates", None)
        if candidates:
            for cand in candidates:
                gm = (getattr(cand, "grounding_metadata", None)
                      or getattr(cand, "grounding", None)
                      or getattr(cand, "grounding_chunks", None)
                      or getattr(cand, "retrievals", None)
                      or getattr(cand, "sources", None))
                gm_plain = _safe_to_plain(gm)
                if isinstance(gm_plain, dict):
                    for key in ("grounding_chunks", "retrieved_chunks", "chunks", "items"):
                        chunks = gm_plain.get(key)
                        if isinstance(chunks, list):
                            for ch in chunks:
                                snippets.append(_normalize_chunk(ch, resp))
                elif isinstance(gm_plain, list):
                    for ch in gm_plain:
                        snippets.append(_normalize_chunk(ch, resp))
    except Exception:
        pass

    if len(snippets) < top_n:
        for attr in ("grounding_metadata", "grounding", "grounding_chunks", "retrievals", "sources", "citation_metadata"):
            val = getattr(resp, attr, None)
            if not val:
                continue
            val_plain = _safe_to_plain(val)
            if isinstance(val_plain, dict):
                for key in ("grounding_chunks", "retrieved_chunks", "chunks", "items"):
                    chunks = val_plain.get(key)
                    if isinstance(chunks, list):
                        for ch in chunks:
                            snippets.append(_normalize_chunk(ch, resp))
            elif isinstance(val_plain, list):
                for ch in val_plain:
                    snippets.append(_normalize_chunk(ch, resp))

    if len(snippets) < top_n:
        raw = getattr(resp, "_raw", None) or getattr(resp, "raw", None)
        raw_plain = _safe_to_plain(raw)
        def scan(obj):
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lk = k.lower()
                    if lk in ("grounding_chunks", "retrieved_chunks", "chunks", "grounding", "retrievals", "sources", "items"):
                        if isinstance(v, list):
                            for el in v:
                                found.append(el)
                        else:
                            found.append(v)
                    else:
                        found.extend(scan(v))
            elif isinstance(obj, list):
                for el in obj:
                    found.extend(scan(el))
            return found
        try:
            if isinstance(raw_plain, dict):
                for ch in scan(raw_plain):
                    snippets.append(_normalize_chunk(ch, resp))
        except Exception:
            pass

    seen = set()
    out = []
    for s in snippets:
        key = (s.get("doc") or "") + "|" + (s.get("text") or "")[:300]
        if not s.get("text") or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= top_n:
            break
    return out

def sanitize_filename(filename: str) -> str:
    base = Path(filename or "").name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe or f"upload_{int(time.time())}"

def safe_write_tmp(uploaded_file, original_name: str) -> str:
    """
    Stream a Streamlit UploadedFile to a temp file without retaining the whole file in memory.
    Returns the temp file path.
    """
    safe_name = sanitize_filename(original_name)
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.abspath(os.path.join(tmp_dir, f"{int(time.time())}_{safe_name}"))

    # Try to stream read in chunks
    try:
        # Ensure file pointer at start
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        with open(tmp_path, "wb") as out_f:
            # Some UploadedFile provide .read into memory; reading in chunks is safer
            while True:
                chunk = uploaded_file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                # chunk may be str (unlikely) or bytes
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                out_f.write(chunk)
                # Optional: flush occasionally for very large files
            out_f.flush()
    except Exception as e:
        # If streaming failed (rare), try a fallback that uses getvalue() for small files
        try:
            data = uploaded_file.getvalue()
            with open(tmp_path, "wb") as out_f:
                if isinstance(data, str):
                    out_f.write(data.encode("utf-8"))
                else:
                    out_f.write(data)
        except Exception as e2:
            raise RuntimeError(f"Failed to save upload: {e} / fallback: {e2}")

    return tmp_path

def robust_upload_to_store(client, tmp_path, store_name, config, sidebar_status):
    """
    Try several upload modes (path string, file handle, bytes-io).
    Returns operation object (or raises). sidebar_status is dict-like with
    'text' and 'progress' callables to update UI in sidebar.
    """
    upload_exceptions = []
    op = None

    # Mode 1: path string
    try:
        sidebar_status["text"]("Starting upload (path mode)...")
        op = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store_name,
            config=config
        )
        sidebar_status["text"]("Upload started (path mode).")
    except Exception as e_path:
        # FAIL FAST: If quota/resource exhausted, don't try other modes
        msg = str(e_path).lower()
        if "resource_exhausted" in msg or "quota" in msg:
            raise RuntimeError("Storage limit exceeded (Resource Exhausted). Please check your Google Cloud/Gemini quota.") from e_path
            
        upload_exceptions.append(("path", e_path))
        try:
            sidebar_status["text"](f"Path upload failed: {e_path}")
        except Exception:
            pass

    # Mode 2: file handle
    if op is None:
        try:
            sidebar_status["text"]("Trying upload (file-handle mode)...")
            with open(tmp_path, "rb") as fh:
                op = client.file_search_stores.upload_to_file_search_store(
                    file=fh,
                    file_search_store_name=store_name,
                    config=config
                )
            sidebar_status["text"]("Upload started (file-handle mode).")
        except Exception as e_fh:
            upload_exceptions.append(("file_handle", e_fh))
            try:
                sidebar_status["text"](f"File-handle upload failed: {e_fh}")
            except Exception:
                pass

    # Mode 3: BytesIO
    if op is None:
        try:
            sidebar_status["text"]("Trying upload (bytes-io mode)...")
            with open(tmp_path, "rb") as f:
                data = f.read()
            bio = io.BytesIO(data)
            bio.name = Path(tmp_path).name
            op = client.file_search_stores.upload_to_file_search_store(
                file=bio,
                file_search_store_name=store_name,
                config=config
            )
            sidebar_status["text"]("Upload started (bytes-io mode).")
        except Exception as e_bio:
            upload_exceptions.append(("bytes_io", e_bio))
            try:
                sidebar_status["text"](f"BytesIO upload failed: {e_bio}")
            except Exception:
                pass

    if op is None:
        msg_lines = ["Upload failed in all modes:"]
        for mode, ex in upload_exceptions:
            msg_lines.append(f"- {mode}: {ex}")
        raise RuntimeError("\n".join(msg_lines))

    return op

def poll_operation(client, op, sidebar_status, timeout=300, interval=2):
    waited = 0
    sidebar_status["text"]("Polling import status...")
    max_steps = max(1, int(timeout // max(1, interval)))
    step = 0
    last_pct = -1

    # attempt to get a stable operation name/identifier if present
    op_name = None
    try:
        op_name = getattr(op, "name", None) or getattr(op, "operation_name", None)
        if op_name is None and isinstance(op, dict):
            op_name = op.get("name") or op.get("id")
    except Exception:
        op_name = None

    # quick check: if op already done, return immediately
    if getattr(op, "done", False) is True:
        sidebar_status["text"]("Import operation completed.")
        return op

    while waited < timeout:
        # compute progress percent (heuristic)
        pct = min(100, int((step / max_steps) * 100))
        if pct != last_pct:
            try:
                sidebar_status["progress"](pct)
            except Exception:
                pass
            last_pct = pct

        # if op indicates done, break
        if getattr(op, "done", None) is True:
            sidebar_status["text"]("Import operation completed.")
            break

        # attempt a single refresh per loop only if operations.get is available and we have an op_name
        try:
            if op_name and getattr(client, "operations", None):
                refreshed = client.operations.get(op_name)
                if refreshed is not None:
                    op = refreshed
        except Exception:
            # ignore refresh errors
            pass

        if getattr(op, "done", False) is True:
            sidebar_status["text"]("Import operation completed (after refresh).")
            break

        time.sleep(interval)
        waited += interval
        step += 1

    if getattr(op, "done", False) is not True:
        sidebar_status["text"]("Import timed out or still running.")
    else:
        # final progress: set 100 if done
        try:
            sidebar_status
        except Exception:
            pass

    return op

def list_filestore_documents_via_rest(parent_resource_name: str, api_key: Optional[str] = None) -> List[Dict]:
    """
    REST fallback to list documents for a File Search store resource path.
    parent_resource_name should be like "projects/PROJECT_ID/locations/global/fileSearchStores/STORE_ID"
    Requires an API key / bearer token in env GEMINI_API_KEY or passed in api_key.
    Returns the raw list of document dicts from the REST response.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("REST fallback requires GEMINI_API_KEY in environment or passed as api_key.")
    url = f"https://generativelanguage.googleapis.com/v1beta/{parent_resource_name}/documents"
    headers = {"Authorization": f"Bearer {key}"}
    params = {"pageSize": 200}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        items = j.get("documents") or j.get("items") or j.get("files") or []
        return items
    except Exception as e:
        raise RuntimeError(f"REST documents list failed: {e}")

def list_documents_in_store(client, store):
    """
    Return list of plain dicts for documents in the given store object or store name.
    Tries SDK first and falls back to REST if the SDK call fails.
    """
    docs = []
    if client is None or store is None:
        return docs

    parent = store
    if hasattr(store, "name"):
        parent = getattr(store, "name")

    # First try the SDK
    try:
        items = list(client.file_search_stores.documents.list(parent=parent))
    except Exception as e:
        logger.warning("SDK list failed for parent=%s: %s. Attempting REST fallback.", parent, e)
        try:
            raw_items = list_filestore_documents_via_rest(parent, api_key=os.environ.get("GEMINI_API_KEY"))
        except Exception as e_rest:
            logger.warning("REST fallback failed for parent=%s: %s", parent, e_rest)
            return []
        # Convert REST items to a uniform structure similar to SDK plain dicts
        items = raw_items

    for it in items:
        plain = _safe_to_plain(it) if not isinstance(it, dict) else it
        title = (
            plain.get("display_name")
            or plain.get("title")
            or plain.get("document_name")
            or plain.get("file_name")
            or plain.get("name")
            or plain.get("id")
            or "(unnamed)"
        )
        doc_res_name = plain.get("name") or plain.get("id") or title
        docs.append({"title": title, "name": doc_res_name, "raw": plain})
    return docs

def delete_file_from_store(client, file_resource_name: str):
    """
    Deletes a file from the corpus/store.
    file_resource_name is the full name, e.g. 'corpora/.../documents/...'
    """
    try:
        # The file resource is typically 'corpora/.../documents/...' or 'projects/.../files/...'
        # Depending on the API, we might need client.files.delete(name=...) 
        # BUT for File Search, files are documents in a store.
        # Actually in Google Gen AI SDK for Python (v0.x/beta), deletion is usually:
        # client.file_search_stores.documents.delete(name=..., force=True) to allow cascading
        # We'll try that first.
        
        # Try with simple force=True first (standard for some resources)
        try:
            client.file_search_stores.documents.delete(name=file_resource_name, force=True)
            return True
        except (TypeError, Exception):
            pass

        # Use user-suggested pattern: config={'force': True}
        # This is common in some Google SDKs for "request options" or "config"
        try:
            client.file_search_stores.documents.delete(name=file_resource_name, config={'force': True})
            return True
        except (TypeError, Exception):
            pass
            
        # Fallback to no args (will fail if non-empty, but we tried our best)
        client.file_search_stores.documents.delete(name=file_resource_name)
        return True
    except Exception as e:
        # Raise so UI can show the specific error
        raise RuntimeError(f"Deletion failed: {e}")

def list_stores_via_sdk(client):
    """Return list of store SDK objects (best-effort)."""
    try:
        return list(client.file_search_stores.list())
    except Exception as e:
        logger.warning("Could not list file search stores: %s", e)
        return []
