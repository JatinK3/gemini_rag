"""
Echo-style chat UI that uses your helper functions and calls google.genai when available.
Run:
    streamlit run app.py
"""

import os
import re
import io
import time
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# REST fallback needs requests
import requests

# Try import google-genai client (if installed & configured)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# Load .env if present (optional)
load_dotenv(override=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echo_rag_chat")

# ---------------------
# Minimal helper functions (copied/adapted)
# ---------------------
def _store_session_key_for(store):
    return "filestore_rows_for_" + (
        getattr(store, "name", None)
        or getattr(store, "display_name", None)
        or "none"
    )


# def _safe_to_plain(o: Any) -> Any:
#     if o is None:
#         return None
#     if isinstance(o, (str, int, float, bool)):
#         return o
#     if isinstance(o, dict):
#         return {k: _safe_to_plain(v) for k, v in o.items()}
#     if isinstance(o, (list, tuple)):
#         return [_safe_to_plain(x) for x in o]
#     for fn in ("model_dump", "to_dict", "as_dict"):
#         if hasattr(o, fn):
#             try:
#                 return _safe_to_plain(getattr(o, fn)())
#             except Exception:
#                 continue
#     if hasattr(o, "__dict__"):
#         try:
#             return _safe_to_plain(vars(o))
#         except Exception:
#             pass
#     try:
#         return str(o)
#     except Exception:
#         return repr(o)
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


# --- File helper functions (must be defined BEFORE sidebar / upload usage) ---
def sanitize_filename(filename: str) -> str:
    base = Path(filename or "").name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe or f"upload_{int(time.time())}"

# def safe_write_tmp(uploaded_file, original_name: str) -> str:
#     """
#     Save a Streamlit UploadedFile to a temp file as quickly as possible.
#     Uses getvalue() when available (fast) but checks size to avoid memory blow.
#     Returns the temp file path.
#     """
#     safe_name = sanitize_filename(original_name)
#     tmp_dir = tempfile.gettempdir()
#     tmp_path = os.path.abspath(os.path.join(tmp_dir, f"{int(time.time())}_{safe_name}"))

#     # Try the fast path: UploadedFile.getvalue() but only for reasonably small files
#     try:
#         data = None
#         if hasattr(uploaded_file, "getvalue"):
#             try:
#                 maybe = uploaded_file.getvalue()
#                 if maybe is not None and (isinstance(maybe, (bytes, bytearray)) and len(maybe) <= 5 * 1024 * 1024):
#                     # use fast path for <=5MB
#                     data = maybe
#                 elif maybe is not None and isinstance(maybe, str) and len(maybe.encode("utf-8")) <= 5 * 1024 * 1024:
#                     data = maybe
#                 else:
#                     data = None
#             except Exception:
#                 data = None

#         if not data:
#             # sometimes .read() is the only option
#             try:
#                 uploaded_file.seek(0)
#             except Exception:
#                 pass
#             try:
#                 data = uploaded_file.read()
#             except Exception:
#                 data = None

#         if not data:
#             raise RuntimeError("Uploaded file appears empty or unreadable.")
#         # write once
#         with open(tmp_path, "wb") as out:
#             if isinstance(data, str):
#                 out.write(data.encode("utf-8"))
#             else:
#                 out.write(data)
#     except Exception as e:
#         # fallback: streaming write (memory safe)
#         try:
#             tf = tempfile.NamedTemporaryFile(prefix=f"{int(time.time())}_", suffix=f"_{safe_name}", delete=False)
#             path = tf.name
#             try:
#                 try:
#                     uploaded_file.seek(0)
#                 except Exception:
#                     pass
#                 while True:
#                     chunk = uploaded_file.read(1024 * 1024)
#                     if not chunk:
#                         break
#                     if isinstance(chunk, str):
#                         chunk = chunk.encode("utf-8")
#                     tf.write(chunk)
#             finally:
#                 tf.flush()
#                 tf.close()
#             tmp_path = path
#         except Exception as e2:
#             raise RuntimeError(f"Failed to save upload: {e} / fallback: {e2}")

#     return tmp_path

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



# --- FileSearch listing helpers (with REST fallback) ---
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


def list_stores_via_sdk(client):
    """Return list of store SDK objects (best-effort)."""
    try:
        return list(client.file_search_stores.list())
    except Exception as e:
        logger.warning("Could not list file search stores: %s", e)
        return []


# ---------------------
# App: Echo style UI + genai integration
# ---------------------

st.set_page_config(page_title="FILE SEARCH RAG", layout="wide")
st.title("FILE SEARCH RAG")

# initialize client if possible
client = None
if GENAI_AVAILABLE:
    try:
        client = genai.Client()
    except Exception as e:
        client = None
        st.sidebar.error(f"genai init failed: {e}")
else:
    st.sidebar.info("google-genai client not installed — running in echo/fallback mode.")

# Sidebar: minimal model / api controls (optional) + upload UI moved here


with st.sidebar:
    st.header("Model / API")
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.0"], index=0)
    max_output_tokens = st.number_input("Max output tokens", min_value=64, max_value=2000, value=1000, step=64)
    api_key_input = st.text_input("GEMINI_API_KEY (or set env var)", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input.strip()

    st.markdown("---")
    st.subheader("File store / Upload")

    if client is None:
        st.info("genai client not initialized — file upload disabled.")
    else:
        # Create new store
        new_store_name = st.text_input("New store display name (optional)", key="sidebar_new_store_name")
        if st.button("Create store", key="sidebar_create_store_btn"):
            try:
                created = client.file_search_stores.create(config={"display_name": new_store_name or f"store-{int(time.time())}"})
                st.success(f"Created store: {getattr(created,'name',str(created))}")
                st.session_state.pop("filestores_cached", None)
            except Exception as e:
                st.error(f"Create failed: {e}")

        st.markdown("---")
        st.caption("Upload & import file into a File Search store")
        wait_for_import = st.checkbox("Wait for import to finish (may take time)", value=False, key="wait_for_import")
        # optional: allow max polling timeout when waiting
        if wait_for_import:
            import_timeout = st.number_input("Import wait timeout (seconds)", min_value=30, max_value=3600, value=300, step=30, key="import_wait_timeout")
        else:
            import_timeout = 0

        # Fetch stores (cached)
        stores = st.session_state.get("filestores_cached")
        if stores is None:
            with st.spinner("Loading stores..."):
                stores = list_stores_via_sdk(client)
            st.session_state["filestores_cached"] = stores

        # if stores:
        #     store_labels = [ getattr(s, "display_name", None) or getattr(s, "name", None) or str(s) for s in stores ]
        #     sel_store_idx = st.selectbox("Select target File Search store", options=list(range(len(store_labels))),
        #                                  format_func=lambda i: store_labels[i], key="sidebar_files_select_store_idx")
        #     target_store = stores[sel_store_idx]
        #     st.markdown(f"**Target store:**  `{getattr(target_store,'name',str(target_store))}`")
        # else:
        #     target_store = None
        #     st.info("No File Search stores found. Create one above or use the Files tab to create.")

        if stores:
            # Build display labels + keep stable index in session
            store_options = [(getattr(s, "display_name", None) or getattr(s, "name", None) or str(s), s) for s in stores]
            store_labels = [lab for lab, _ in store_options]

            # Single selectbox used everywhere (upload + listing + RAG)
            sel_idx = st.selectbox(
                "Select File Search store (upload and queries will use this)",
                options=list(range(len(store_labels))),
                format_func=lambda i: store_labels[i],
                key="selected_filestore_index"  # single stable key
            )

            # selected_store / target_store are the same object for the whole app
            target_store = stores[sel_idx]
            selected_store = target_store  # ensure the same variable is used by the rest of the app
            st.markdown(f"**Selected store:**  `{getattr(target_store,'name',str(target_store))}`")
            # try:
            #     docs_in_store = list_documents_in_store(client, selected_store)
            # except Exception as e:
            #     logger.warning("Could not list docs for store %s: %s", getattr(selected_store, "name", str(selected_store)), e)
            #     docs_in_store = []
            # Prefer cached listing per-store to avoid network calls on every rerun
            store_key = _store_session_key_for(selected_store)
            docs_in_store = st.session_state.get(store_key)
            if docs_in_store is None:
                try:
                    docs_in_store = list_documents_in_store(client, selected_store)
                except Exception as e:
                    logger.warning("Could not list docs for store %s: %s", getattr(selected_store, "name", str(selected_store)), e)
                    docs_in_store = []
                st.session_state[store_key] = docs_in_store

            if docs_in_store:
                st.markdown("**Documents in this store:**")
                display_rows = []
                for d in docs_in_store:
                    title = d.get("title") or "(no title)"
                    resname = d.get("name") or ""
                    raw = d.get("raw") or {}
                    pages = ""
                    for k in ("page_count", "pages", "num_pages"):
                        try:
                            if isinstance(raw, dict) and k in raw:
                                pages = raw.get(k)
                                break
                        except Exception:
                            pass
                    display_rows.append({"title": title, "resource_name": resname, "pages": pages or ""})
                st.table(display_rows)
            else:
                st.info("No documents found in this store.")
        else:
            target_store = None
            selected_store = None
            st.info("No File Search stores found. Create one above or import files in Files tab.")

        if selected_store is not None:
            refreshing = st.session_state.get("_refreshing_docs", False)
            if st.button("Refresh documents", key="sidebar_refresh_docs_btn", disabled=refreshing):
                # invalidate cached doc list for this store
                store_key = "filestore_rows_for_" + (getattr(selected_store, "name", "none"))
                st.session_state.pop(store_key, None)
                # also optionally refresh store list cache to pick up new stores
                st.session_state.pop("filestores_cached", None)

                # fetch new docs immediately and store them in session (non-blocking, short)
                try:
                    with st.spinner("Refreshing documents..."):
                        rows = list_documents_in_store(client, selected_store)
                    st.session_state[store_key] = rows
                    st.success("Document list refreshed.")
                except Exception as e:
                    st.error(f"Failed to refresh documents: {e}")

        upload_file = st.file_uploader("Choose a file (PDF, DOCX, TXT, etc.)", accept_multiple_files=False, key="sidebar_files_uploader")
        display_name = st.text_input("Display name for the file (optional)", key="sidebar_files_display_name")
        chunk_tokens = st.number_input("Max tokens per chunk (0 to disable)", min_value=0, max_value=2000, value=0, step=50, key="sidebar_chunk_tokens")
        chunk_overlap = st.number_input("Max overlap tokens", min_value=0, max_value=1000, value=0, step=10, key="sidebar_chunk_overlap")

        if st.button("Import file into selected store", key="sidebar_files_import_btn"):
            if upload_file is None:
                st.error("Please choose a file to upload.")
            else:
                # ensure target_store exists ; create ephemeral store if not selected
                if target_store is None:
                    try:
                        target_store = client.file_search_stores.create(config={"display_name": f"streamlit-store-{int(time.time())}"})
                        st.success(f"Created store {getattr(target_store,'name',str(target_store))}")
                        st.session_state.pop("filestores_cached", None)
                    except Exception as e:
                        st.error(f"Failed to create store: {e}")
                        target_store = None

                if target_store is None:
                    st.error("No target store available.")
                else:
                    tmp_path = None
                    try:
                        tmp_path = safe_write_tmp(upload_file, upload_file.name)
                    except Exception as e:
                        st.error(f"Failed to save uploaded file locally: {e}")
                        tmp_path = None

                    if tmp_path:
                        cfg = {}
                        if int(chunk_tokens) > 0:
                            cfg["chunking_config"] = {
                                "white_space_config": {
                                    "max_tokens_per_chunk": int(chunk_tokens),
                                    "max_overlap_tokens": int(chunk_overlap or 0),
                                }
                            }
                        if display_name:
                            cfg["display_name"] = display_name

                        # in-sidebar status
                        sidebar_status = st.empty()
                        sidebar_progress = st.progress(0)
                        status_obj = {"text": lambda m: sidebar_status.info(m), "progress": lambda p: sidebar_progress.progress(p)}

                        try:
                            status_obj["text"]("Starting upload...")
                            op = robust_upload_to_store(client, tmp_path, getattr(target_store,"name", str(target_store)), cfg, status_obj)
                        except Exception as e:
                            status_obj["text"](f"Upload failed: {e}")
                            st.error(f"Upload failed: {e}")
                        else:
                            # If user does NOT want to wait for import, return early and show the operation id
                            if not wait_for_import:
                                # operation may be a dict-like or object; try to get a name/id
                                op_name = getattr(op, "name", None) or getattr(op, "operation_name", None)
                                if op_name is None and isinstance(op, dict):
                                    op_name = op.get("name") or op.get("id")
                                # st.success("Upload complete — import has been queued.")
                                # if op_name:
                                #     st.info(f"Import operation id: `{op_name}`")
                                #     # Invalidate caches
                                #     store_key = _store_session_key_for(target_store)
                                #     st.session_state.pop(store_key, None)
                                #     st.session_state.pop("filestores_cached", None)
                                # # invalidate cached stores/doc listings as before (they will show after indexing completes)
                                # st.session_state.pop("filestore_rows_for_" + (getattr(target_store, "name", "none")), None)
                                # st.session_state.pop("filestores_cached", None)
                                # try:
                                #     st.session_state["_refreshing_docs"] = True
                                #     with st.spinner("Fetching updated document list..."):
                                #         new_rows = list_documents_in_store(client, target_store)
                                #     st.session_state[store_key] = new_rows
                                #     st.success("Document list updated after upload.")
                                # except Exception as e_fetch:
                                #     logger.debug("Post-upload doc fetch failed or indexing not ready: %s", e_fetch)
                                #     st.info("Upload queued; document may take a moment to appear. Use 'Refresh documents' to check.")
                                # finally:
                                #     st.session_state["_refreshing_docs"] = False

                                st.success("Upload complete — import has been queued.")
                                if op_name:
                                    st.info(f"Import operation id: `{op_name}`")

                                # Invalidate caches (single place)
                                store_key = _store_session_key_for(target_store)
                                st.session_state.pop(store_key, None)
                                st.session_state.pop("filestores_cached", None)

                                # Try one immediate refresh (may fail if indexing not finished)
                                try:
                                    if st.session_state.get("_refreshing_docs", False):
                                        docs_in_store = st.session_state.get(store_key, []) or []
                                    else:
                                        st.session_state["_refreshing_docs"] = True
                                        with st.spinner("Fetching updated document list..."):
                                            new_rows = list_documents_in_store(client, target_store)
                                        st.session_state[store_key] = new_rows
                                    st.success("Document list updated after upload.")
                                    
                                except Exception as e_fetch:
                                    logger.debug("Post-upload doc fetch failed or indexing not ready: %s", e_fetch)
                                    st.info("Upload queued; document may take a moment to appear. Use 'Refresh documents' to check.")
                                finally:
                                    st.session_state["_refreshing_docs"] = False
                            else:
                                # Wait for the import to finish (user opted in)
                                try:
                                    # pass configured timeout and a slightly shorter interval
                                    # final_op = poll_operation(client, op, status_obj, timeout=int(import_timeout or 300), interval=1)
                                    final_op = poll_operation(client, op, status_obj, timeout=int(import_timeout or 300), interval=2)
                                    status_obj["text"]("Import finished (or queued).")
                                    st.success("Import finished (or queued). File has been indexed into the selected File Search store.")
                                    # Invalidate caches and refresh documents after indexing
                                    store_key = _store_session_key_for(target_store)
                                    st.session_state.pop(store_key, None)
                                    st.session_state.pop("filestores_cached", None)

                                    try:
                                        st.session_state["_refreshing_docs"] = True
                                        with st.spinner("Refreshing document list after indexing..."):
                                            new_rows = list_documents_in_store(client, target_store)
                                        st.session_state[store_key] = new_rows
                                        st.success("Document list refreshed (indexing complete).")
                                    except Exception as e_fetch:
                                        logger.debug("Post-import doc fetch failed: %s", e_fetch)
                                        st.info("Import finished but the document may not yet be listed. Use 'Refresh documents' later.")
                                    finally:
                                        st.session_state["_refreshing_docs"] = False
                                    st.session_state.pop("filestore_rows_for_" + (getattr(target_store, "name", "none")), None)
                                    st.session_state.pop("filestores_cached", None)
                                except Exception as e:
                                    st.error(f"Error while waiting for import: {e}")
                        finally:
                            try:
                                if tmp_path and os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                            except Exception:
                                pass


# initialize session-state message history (only once)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render message history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Sidebar: select a File Search store (shows stores only if client available)
# if client is None:
#     st.sidebar.info("genai client not available — filestore selector disabled.")
#     selected_store = None
# else:
#     # cache stores in session to avoid repeated network calls
#     if "filestores_cached" not in st.session_state:
#         st.session_state["filestores_cached"] = None
#     if st.sidebar.button("Refresh file stores"):
#         st.session_state["filestores_cached"] = None

#     stores = st.session_state.get("filestores_cached")
#     if stores is None:
#         sidebar_status = st.sidebar.empty()
#         try:
#             sidebar_status.info("Loading File Search stores...")
#             with st.spinner("Loading File Search stores..."):
#                 stores = list_stores_via_sdk(client)
#             st.session_state["filestores_cached"] = stores
#         finally:
#             try:
#                 sidebar_status.empty()
#             except Exception:
#                 pass

#     # Build mapping for display
#     if stores:
#         store_options = []
#         for s in stores:
#             display = getattr(s, "display_name", None) or getattr(s, "name", None) or str(s)
#             store_options.append((display, s))
#         store_labels = [lab for lab, _ in store_options]

#         sel_store_idx = st.sidebar.selectbox(
#             "Select File Search store",
#             options=list(range(len(store_labels))),
#             format_func=lambda i: store_labels[i],
#             key="selected_filestore_index"
#         )

#         selected_store = store_options[sel_store_idx][1]
    #     st.sidebar.markdown(f"**Selected store:**  \n`{getattr(selected_store,'name',str(selected_store))}`")

    #     # Try to list documents for the selected store and show them in a compact table
    #     try:
    #         docs_in_store = list_documents_in_store(client, selected_store)
    #     except Exception as e:
    #         logger.warning("Could not list docs for store %s: %s", getattr(selected_store, "name", str(selected_store)), e)
    #         docs_in_store = []

    #     if docs_in_store:
    #         st.sidebar.markdown("**Documents in this store:**")
    #         display_rows = []
    #         for d in docs_in_store:
    #             title = d.get("title") or "(no title)"
    #             resname = d.get("name") or ""
    #             raw = d.get("raw") or {}
    #             pages = ""
    #             for k in ("page_count", "pages", "num_pages"):
    #                 try:
    #                     if isinstance(raw, dict) and k in raw:
    #                         pages = raw.get(k)
    #                         break
    #                 except Exception:
    #                     pass
    #             display_rows.append({"title": title, "resource_name": resname, "pages": pages or ""})
    #         st.sidebar.table(display_rows)
    #     else:
    #         st.sidebar.info("No documents found in this store.")
    # else:
    #     selected_store = None
    #     st.sidebar.info("No File Search stores found (create/import in Files tab).")

# Main chat input (single, well-indented block)
prompt = st.chat_input("What is up?")
if prompt:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # If no client available, fallback to echo (helps development)
    if client is None:
        response_text = f"Echo: {prompt}"
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
    else:
        # Base system prompt
        system_prompt = ("""SYSTEM INSTRUCTIONS:
                You are a grounded assistant that MUST prioritize and use only content from the documents returned by the File Search retrieval tool when answering user questions.

                PRIMARY RULES (read carefully):
                1. Retrieval-first:
                - First, retrieve relevant document passages. Only use passages returned by the File Search tool or the explicit "RETRIEVED_SNIPPETS" context provided to you.
                - Do NOT invent sources. If a requested fact is not in the retrieved passages, say you cannot find it in the retrieved documents (but may answer from general knowledge only if explicitly allowed below).

                2. Answering & citations:
                - If the retrieved passages contain the answer, produce a final answer that uses only those passages (you may paraphrase).
                - When quoting a passage verbatim, enclose it in quotes and include the same inline citation immediately after the quote.

                3. Output structure (STRICT):
                - Provide a short summary (2–3 sentences) first.
                - Then provide step-by-step explanation or runnable code if relevant.
                - Use code fences and specify language for code blocks.

                4. Fallback rule:
                - If the documents do NOT contain the needed information, and you are NOT allowed to use general knowledge, respond exactly with: "I cannot answer from the provided documents."
                - If general knowledge is allowed (the UI will indicate this), you MAY supplement with general knowledge but MUST mark general-knowledge segments with the prefix [GENERAL].

                5. Safety & honesty:
                - Never fabricate page numbers, document names, or direct quotes.

                6. ANSWER STYLE:
                - Provide a short summary (2–3 sentences) first.
                - Then provide step-by-step explanation or runnable code if relevant.
                - Use code fences and specify language for code blocks.
                - Answer must be a minimum of 100–150 words.
                If (and only if) the user explicitly requests a **comparison** or **difference** or **contrast** or **briefly** (e.g., “compare”, “difference between”, “advantages vs disadvantages”),
                then present the comparison in a **Markdown table**.
                - The table must include only information grounded in the retrieved document passages. Do NOT invent rows, columns, or attributes.
                - After the table, include a short narrative explanation summarizing the key differences.
            """)


        # If a store is selected, let the model know which store it can consult (for clarity)
        if selected_store is not None:
            try:
                store_name_display = getattr(selected_store, "name", None) or getattr(selected_store, "display_name", None) or str(selected_store)
                system_prompt = system_prompt + f"\n\nNOTE: You may consult documents in the File Search store: {store_name_display}\n"
            except Exception:
                pass

        # Compose contents in the same order your original used: system prompt then user prompt
        contents = [system_prompt, prompt]

        try:
            # Build tools list: if a store is selected and SDK is available, include the FileSearch tool so the model can ask for retrievals.
            tools_list = None
            if GENAI_AVAILABLE and selected_store is not None:
                try:
                    store_res_name = getattr(selected_store, "name", None) or getattr(selected_store, "display_name", None)
                    if store_res_name:
                        fs = types.FileSearch(file_search_store_names=[store_res_name])
                        fs_tool = types.Tool(file_search=fs)
                        tools_list = [fs_tool]
                except Exception as e_tool:
                    logger.warning("Failed to build FileSearch tool: %s", e_tool)
                    tools_list = None

            # Call model with tools if present
            if GENAI_AVAILABLE:
                # include tools only if built
                if tools_list:
                    cfg = types.GenerateContentConfig(max_output_tokens=int(max_output_tokens), tools=tools_list)
                else:
                    cfg = types.GenerateContentConfig(max_output_tokens=int(max_output_tokens))
                resp = client.models.generate_content(model=model_choice, contents=contents, config=cfg)
            else:
                # Fallback: echoing to simulate a response object shape as minimally as possible
                class DummyResp:
                    text = f"Echo (no genai): {prompt}"
                resp = DummyResp()

            # extract answer text using provided helper
            answer_text = extract_text(resp) or "(no answer returned)"

            # show assistant message
            with st.chat_message("assistant"):
                st.write(answer_text)
                # print(answer_text)
            st.session_state["messages"].append({"role": "assistant", "content": answer_text})

            # For debugging: store the raw response (plainified) for inspection
            try:
                raw_inspect = getattr(resp, "_raw", None) or getattr(resp, "raw", None)
                st.session_state["_last_resp_raw"] = _safe_to_plain(raw_inspect)
            except Exception:
                st.session_state["_last_resp_raw"] = None

            # try to extract grounding snippets (may be empty)
            snips = get_top_grounding_snippets(resp, top_n=5)
            if snips:
                sources_summary = "Sources: " + ", ".join(pretty_doc_name(s.get("doc") or "Unknown") for s in snips)
            else:
                # If no grounding snippets and you passed a store, it's useful to inform the user
                if selected_store is not None:
                        st.info("_No grounding snippets were found in the model response. The model may have used general knowledge or the store contained no relevant content._")

        except Exception as e:
            # show error to user
            with st.chat_message("assistant"):
                st.markdown(f"Request failed: {e}")
            st.session_state["messages"].append({"role": "assistant", "content": f"Request failed: {e}"})

