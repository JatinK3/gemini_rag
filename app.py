# app.py
import os
import time
from typing import Optional
import os
import tempfile
from pathlib import Path
import streamlit as st
import re
# google-genai official client
from google import genai
from google.genai import types

# Optional: nicer mime detection if you want to show file type
try:
    import magic
except Exception:
    magic = None

st.set_page_config(page_title="Gemini File Search RAG demo", layout="wide")

st.title("RAG with Gemini File Search — Streamlit demo")

# ---- Auth / Init ----
st.sidebar.header("Configuration")
# Sidebar config (use unique keys)
api_key = st.sidebar.text_input(
    "GEMINI API KEY (or set GEMINI_API_KEY env var)",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password",
    key="gemini_api_key_input"
)

use_vertex = st.sidebar.checkbox(
    "Use VertexAI client (vertexai=True)",
    value=False,
    key="use_vertex_checkbox"
)

project = st.sidebar.text_input(
    "GCP project (optional, only for VertexAI)",
    value=os.environ.get("GOOGLE_CLOUD_PROJECT", "") or "",
    key="gcp_project_input"
)

location = st.sidebar.text_input(
    "Location (optional)",
    value=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    key="gcp_location_input"
)


# Initialize client
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# if you prefer service account auth for Vertex AI, set GOOGLE_APPLICATION_CREDENTIALS env var outside this app
try:
    if use_vertex:
        client = genai.Client(vertexai=True, project=project or None, location=location or None)
    else:
        client = genai.Client()
except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    st.stop()

st.sidebar.markdown(
    """
    **How to authenticate**
    * Easiest: set `GEMINI_API_KEY` env var (developer API).
    * Vertex: use `GOOGLE_APPLICATION_CREDENTIALS` (service account) and set `vertexai=True`.
    """
)

# ---- File Search store management ----
st.header("File Search stores")

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Create a store")
    new_store_name = st.text_input("Display name for new File Search store")
    if st.button("Create store"):
        try:
            created = client.file_search_stores.create(config={"display_name": new_store_name or f"store-{int(time.time())}"})
            st.success(f"Created: {created.name}")
        except Exception as e:
            st.error(f"Create failed: {e}")

with col2:
    st.subheader("Your stores")
    stores = []
    try:
        stores = list(client.file_search_stores.list())
    except Exception as e:
        st.warning(f"Could not list stores: {e}")
    store_map = {s.display_name or s.name: s for s in stores}
    if store_map:
        selected_store_label = st.selectbox("Select File Search store", options=list(store_map.keys()))
        selected_store = store_map[selected_store_label]
        st.write("Store name (use this in code):", selected_store.name)
    else:
        st.info("No File Search stores found — create one above or upload a file to create+import automatically.")
        selected_store = None

# ---- Upload / import files ----
st.header("Upload & import files into File Search store")
upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    upload_file = st.file_uploader("Choose a file to upload (PDF, TXT, DOCX, etc.)", accept_multiple_files=False)
    display_name = st.text_input("File display name (optional)")

with upload_col2:
    chunk_max_tokens = st.number_input("Max tokens per chunk (optional)", min_value=0, max_value=2000, value=0, step=50)
    max_overlap = st.number_input("Max overlap tokens (optional)", min_value=0, max_value=1000, value=0, step=10)
def sanitize_filename(filename: str) -> str:
    base = Path(filename).name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe or f"upload_{int(time.time())}"

if st.button("Upload & import to store", key="upload_and_import_btn"):
    if upload_file is None and selected_store is None:
        st.error("Upload a file or select/create a store first.")
    else:
        # --- create/select store ---
        target_store = selected_store
        if target_store is None:
            st.info("No store selected — creating one for you...")
            try:
                target_store = client.file_search_stores.create(
                    config={"display_name": f"streamlit-store-{int(time.time())}"}
                )
                st.success(f"Created store {target_store.name}")
            except Exception as e:
                st.error(f"Failed to create store: {e}")
                st.stop()

        if upload_file is None:
            st.info("No file uploaded — select a file to import.")
            st.stop()

        # ---------- Write uploaded file to a safe temp path ----------
        safe_name = sanitize_filename(upload_file.name)
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.abspath(os.path.join(tmp_dir, f"{int(time.time())}_{safe_name}"))

        try:
            # Reset pointer then read bytes (support different Streamlit versions)
            try:
                upload_file.seek(0)
            except Exception:
                pass

            try:
                data = upload_file.getvalue()
            except Exception:
                data = upload_file.read()

            if not data:
                st.error("Uploaded file appears empty.")
                st.stop()

            with open(tmp_path, "wb") as out:
                out.write(data)
        except Exception as e:
            st.error(f"Failed to save uploaded file to {tmp_path}: {e}")
            st.stop()

        st.success(f"Saved upload to {tmp_path}")
        st.write("Temp path exists:", os.path.exists(tmp_path))
        st.write("Abs path:", tmp_path)

        # ---------- Build the chunking/display_name config ONCE ----------
        config = {}
        if chunk_max_tokens and int(chunk_max_tokens) > 0:
            config["chunking_config"] = {
                "white_space_config": {
                    "max_tokens_per_chunk": int(chunk_max_tokens),
                    "max_overlap_tokens": int(max_overlap or 0),
                }
            }
        if display_name:
            config["display_name"] = display_name

        # ---------- Try upload in several modes (path -> file-handle -> BytesIO) ----------
        op = None
        upload_exceptions = []

        # Mode 1: path string
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file=tmp_path,
                file_search_store_name=target_store.name,
                config=config
            )
            st.info("Upload started (path mode).")
        except Exception as e_path:
            upload_exceptions.append(("path", e_path))
            st.warning(f"Path upload failed: {e_path}")

        # Mode 2: file handle
        if op is None:
            try:
                with open(tmp_path, "rb") as fh:
                    op = client.file_search_stores.upload_to_file_search_store(
                        file=fh,
                        file_search_store_name=target_store.name,
                        config=config
                    )
                st.info("Upload started (file-handle mode).")
            except Exception as e_fh:
                upload_exceptions.append(("file_handle", e_fh))
                st.warning(f"File-handle upload failed: {e_fh}")

        # Mode 3: in-memory BytesIO (give it a .name in case SDK checks)
        if op is None:
            try:
                import io
                bio = io.BytesIO(data)
                bio.name = safe_name
                op = client.file_search_stores.upload_to_file_search_store(
                    file=bio,
                    file_search_store_name=target_store.name,
                    config=config
                )
                st.info("Upload started (bytes-io mode).")
            except Exception as e_bio:
                upload_exceptions.append(("bytes_io", e_bio))
                st.error("All upload modes failed. See diagnostics below.")
                for mode, ex in upload_exceptions:
                    st.write(f"Mode: {mode} -> Exception: {ex}")
                # show helpful FS checks
                st.write("Temp file exists:", os.path.exists(tmp_path))
                st.write("Temp file size:", os.path.getsize(tmp_path) if os.path.exists(tmp_path) else "N/A")
                st.stop()

        # ---------- Poll the operation defensively ----------
        try:
            st.info("Started import operation — polling status...")
            max_wait = 300
            waited = 0
            interval = 2
            while True:
                # Some SDKs provide op.done attribute
                done = getattr(op, "done", None)
                st.write(f"op.done = {done}")
                if done is True:
                    st.success("Import operation reports done=True")
                    break

                # Some SDKs require refreshing with client.operations.get(op)
                try:
                    op = client.operations.get(op)
                except Exception:
                    # refresh may not be supported; continue to wait a bit
                    pass

                if getattr(op, "done", False) is True:
                    st.success("Import operation reports done=True after refresh")
                    break

                if waited >= max_wait:
                    st.warning("Import is taking longer than expected; it may finish in background.")
                    break

                time.sleep(interval)
                waited += interval

            st.success("Import finished (or queued). File has been indexed into the selected File Search store.")
            st.write("Store:", target_store.name)

        except Exception as e:
            st.error(f"Error while waiting for import: {e}")

        finally:
            # CLEANUP temp file AFTER upload/poll
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

# ---- Ask questions (RAG) ----
st.header("Ask questions (RAG)")

question = st.text_area("Enter your question", height=120)
model = st.selectbox("Model", options=["gemini-2.5-flash", "gemini-1.0"], index=0)
max_tokens = st.number_input("Max output tokens", min_value=64, max_value=2000, value=512, step=64)

if st.button("Ask", key="ask_button"):
    if not question.strip():
        st.error("Type a question.")
    elif selected_store is None:
        st.error("Select or create a File Search store that contains your docs.")
    else:
        st.info("Querying Gemini with File Search as a tool...")
        try:
            # Build FileSearch tool config and attach it to GenerateContentConfig.tools
            fs_tool = types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[selected_store.name]
                )
            )

            cfg = types.GenerateContentConfig(
                max_output_tokens=int(max_tokens),
                tools=[fs_tool],  # <<< important: attach the File Search tool here
            )

            # Call generate_content (note: pass model name directly)
            response = client.models.generate_content(
                model=model,
                contents=question,
                config=cfg,
            )

            # --- Extract answer text robustly across SDK versions ---
            def extract_text(resp):
                # 1) common: response.text
                txt = getattr(resp, "text", None)
                if txt:
                    return txt

                # 2) candidates / outputs / generations / choices (iterate and concat)
                for attr in ("candidates", "outputs", "output", "generations", "choices"):
                    val = getattr(resp, attr, None)
                    if val:
                        # if it's a list-like, gather text-like fields
                        pieces = []
                        try:
                            iterable = val if isinstance(val, (list, tuple)) else [val]
                            for item in iterable:
                                for a in ("text", "content", "display_text"):
                                    if hasattr(item, a):
                                        pieces.append(getattr(item, a))
                                # nested message/completion/result objects
                                for sub in ("message", "completion", "result"):
                                    if hasattr(item, sub):
                                        subobj = getattr(item, sub)
                                        for a in ("text", "content"):
                                            if hasattr(subobj, a):
                                                pieces.append(getattr(subobj, a))
                                # fallback to str(item)
                                pieces.append(str(item))
                        except Exception:
                            pass
                        # return first non-empty concatenated piece
                        joined = "\n\n".join([p for p in pieces if p])
                        if joined:
                            return joined

                # 3) raw dict search
                raw = getattr(resp, "_raw", None) or getattr(resp, "raw", None)
                if isinstance(raw, dict):
                    # shallow search for short text fields
                    def find_first_text(d):
                        if isinstance(d, str):
                            return d
                        if isinstance(d, dict):
                            for k, v in d.items():
                                if isinstance(v, str) and len(v) < 10000:
                                    return v
                                res = find_first_text(v)
                                if res:
                                    return res
                        if isinstance(d, list):
                            for el in d:
                                res = find_first_text(el)
                                if res:
                                    return res
                        return None
                    candidate = find_first_text(raw)
                    if candidate:
                        return candidate

                return None
            
            def extract_text(resp):
                # 1) Direct text
                if getattr(resp, "text", None):
                    return resp.text

                pieces = []

                # 2) Look into candidates / outputs / generations
                for attr in ("candidates", "outputs", "output", "generations", "choices"):
                    seq = getattr(resp, attr, None)
                    if not seq:
                        continue

                    if not isinstance(seq, (list, tuple)):
                        seq = [seq]

                    for item in seq:
                        # Try common text fields
                        for f in ("text", "content", "display_text"):
                            if hasattr(item, f):
                                val = getattr(item, f)
                                if isinstance(val, str):
                                    pieces.append(val)
                                else:
                                    pieces.append(str(val))

                        # Nested fields (message/result/completion)
                        for sub in ("message", "completion", "result"):
                            if hasattr(item, sub):
                                subobj = getattr(item, sub)
                                for f in ("text", "content"):
                                    if hasattr(subobj, f):
                                        val = getattr(subobj, f)
                                        if isinstance(val, str):
                                            pieces.append(val)
                                        else:
                                            pieces.append(str(val))

                        # Final fallback for item
                        pieces.append(str(item))

                # 3) Raw fallback
                raw = getattr(resp, "_raw", None) or getattr(resp, "raw", None)
                if isinstance(raw, dict):
                    def scan(obj):
                        if isinstance(obj, str):
                            return [obj]
                        out = []
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                out += scan(v)
                        elif isinstance(obj, list):
                            for el in obj:
                                out += scan(el)
                        return out

                    raw_text = scan(raw)
                    for t in raw_text:
                        if t and isinstance(t, str):
                            pieces.append(t)

                # Clean + join collected pieces
                pieces = [str(p).strip() for p in pieces if p and str(p).strip()]

                if pieces:
                    return "\n\n".join(pieces)

                return None


            answer_text = extract_text(response)

            st.subheader("Answer")
            if answer_text:
                st.write(answer_text)
            else:
                st.info("No plain text answer found in response. See raw fields below.")
                # Show raw response to help debugging
                raw = getattr(response, "_raw", None) or getattr(response, "raw", None)
                if raw:
                    try:
                        st.json(raw)
                    except Exception:
                        st.write(str(raw))
                else:
                    # fallback: show repr/dir
                    st.write("repr(response):", repr(response)[:1000])
                    st.write("attributes:", [a for a in dir(response) if not a.startswith("_")])

            # --- Grounding / citations extraction (best-effort) ---
            st.subheader("Grounding / citations (if provided)")
            grounding = None
            for attr in ("grounding_metadata", "grounding", "groundingMetadata", "grounding_chunks", "retrievals", "sources"):
                grounding = getattr(response, attr, None)
                if grounding:
                    break

            if grounding:
                try:
                    st.json(grounding.__dict__ if hasattr(grounding, "__dict__") else grounding)
                except Exception:
                    st.write(grounding)
            else:
                # try to find citations in raw
                raw = getattr(response, "_raw", None) or getattr(response, "raw", None)
                if raw:
                    # show keys likely to contain citations
                    def find_keys(obj, keys):
                        found = {}
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if k.lower() in keys:
                                    found[k] = v
                                else:
                                    res = find_keys(v, keys)
                                    if res:
                                        found[k] = res
                        elif isinstance(obj, list):
                            for el in obj:
                                res = find_keys(el, keys)
                                if res:
                                    found.setdefault("list_matches", []).append(res)
                        return found
                    matches = find_keys(raw, {"grounding", "retrieval", "sources", "citations"})
                    if matches:
                        st.write("Found candidate grounding keys in raw:")
                        st.json(matches)
                    else:
                        st.info("No explicit grounding metadata found in raw response.")
                else:
                    st.info("No explicit grounding metadata returned by SDK for this response.")
        except Exception as e:
            st.error(f"Request failed: {e}")
