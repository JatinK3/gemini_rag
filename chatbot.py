import os
import time
import logging
from itertools import chain
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

# Import helpers from utils
from utils import (
    _store_session_key_for,
    _render_with_cursor,
    _looks_like_structured_output,
    classify_gemini_error,
    _safe_to_plain,
    extract_text,
    _extract_page_from_text,
    pretty_doc_name,
    _normalize_chunk,
    get_top_grounding_snippets,
    sanitize_filename,
    safe_write_tmp,
    robust_upload_to_store,
    poll_operation,
    list_documents_in_store,
    list_stores_via_sdk,
    delete_file_from_store
)

# Load .env if present (optional)
load_dotenv(override=False)
if "gemini_api_key" not in st.session_state:
    st.session_state["gemini_api_key"] = os.environ.get("GEMINI_API_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echo_rag_chat")

# ---------------------
# UI Helpers
# ---------------------
def handle_streaming_error(e, model_choice):
    info = classify_gemini_error(e)
    st.session_state["_last_gemini_error_handled"] = True

    if info["type"] == "quota":
        st.error(
            "🚫 **Gemini free-tier quota reached**\n\n"
            f"- Model: `{info['model'] or model_choice}`\n"
            "Please wait for quota reset or upgrade your plan."
        )
    elif info["type"] == "rate":
        st.warning("⚠️ **Rate limit hit** — please retry shortly.")
    elif info["type"] == "auth":
        st.error("🔑 **API key issue** — check your key.")
    else:
        st.error(f"Request failed: {e}")

    st.stop()

# ---------------------
# App: Echo style UI + genai integration
# ---------------------

st.set_page_config(page_title="FILE SEARCH RAG", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa; 
    }
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>FILE SEARCH RAG</h1>", unsafe_allow_html=True)
st.markdown("---")

# initialize client if possible
client = None

if GENAI_AVAILABLE and st.session_state.get("gemini_api_key"):
    try:
        # Recreate client only if key changed or client missing
        if st.session_state.get("_client_needs_refresh") or "genai_client" not in st.session_state:
            st.session_state["genai_client"] = genai.Client()
            st.session_state["_client_needs_refresh"] = False

        client = st.session_state["genai_client"]

    except Exception as e:
        client = None
        st.sidebar.error(f"Gemini client initialization failed: {e}")
else:
    st.sidebar.info("Set GEMINI_API_KEY to enable Gemini features.")
if st.session_state["gemini_api_key"]:
    st.sidebar.success("Gemini API key loaded")
else:
    st.sidebar.warning("No Gemini API key set")

# Sidebar: minimal model / api controls (optional) + upload UI moved here

with st.sidebar:
    st.markdown("### Usage / Quota")
    if st.session_state.get("_quota_warning"):
        st.warning(st.session_state["_quota_warning"])
    else:
        st.caption(
            "Free Gemini tier has strict request limits. "
            "If you hit limits, wait a bit or upgrade your API plan."
            )
    st.markdown("---")
    st.header("Model / API")
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.0"], index=0)
    max_output_tokens = st.number_input("Max output tokens", min_value=64, max_value=2000, value=1000, step=64)
    api_key_input = st.text_input(
            "GEMINI_API_KEY",
            value=st.session_state["gemini_api_key"],
            type="password",
            help="Loaded from environment by default. You can override it here."
        )
    apply_key = st.button("Apply API Key", type="primary")
    if apply_key or api_key_input != st.session_state["gemini_api_key"]:
        new_key = api_key_input.strip()

        if new_key:
            # Persist key
            st.session_state["gemini_api_key"] = new_key
            os.environ["GEMINI_API_KEY"] = new_key

            # Force client + store refresh
            st.session_state["_client_needs_refresh"] = True
            st.session_state.pop("genai_client", None)
            st.session_state.pop("filestores_cached", None)

            # Clear selected store (API-key scoped)
            st.session_state.pop("selected_filestore_index", None)

            st.success("✅ API key applied. Reloading stores…")
            st.rerun()

    if api_key_input != st.session_state["gemini_api_key"]:
        st.session_state["gemini_api_key"] = api_key_input.strip()
        os.environ["GEMINI_API_KEY"] = api_key_input.strip()
        st.session_state["_client_needs_refresh"] = True

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
                with st.expander(f"Documents in this store ({len(docs_in_store)})", expanded=False):
                    for i, d in enumerate(docs_in_store):
                        title = d.get("title") or "(no title)"
                        st.markdown(f"**{i+1}.** 📄 {title}")
            else:
                st.info("No documents found in this store.")
            
            # --- FILE DELETION SECTION ---
            if docs_in_store:
                st.markdown("---")
                st.caption("🗑️ **Delete Files**")
                
                # Multi-select for deletion
                del_title_options = [d.get("title") for d in docs_in_store]
                selected_del = st.multiselect("Select files to delete", del_title_options, key="sidebar_delete_multi")
                
                col_d1, col_d2 = st.columns(2)
                
                # Button 1: Delete Selected
                if col_d1.button("Delete Selected", type="secondary", disabled=not selected_del):
                    success_count = 0
                    fail_count = 0
                    
                    with st.status("Deleting files...", expanded=True) as status:
                        for del_title in selected_del:
                            # find doc
                            doc = next((d for d in docs_in_store if d.get("title") == del_title), None)
                            if doc:
                                res_name = doc.get("name")
                                try:
                                    status.write(f"Deleting {del_title}...")
                                    delete_file_from_store(client, res_name)
                                    status.write(f"✅ Deleted {del_title}")
                                    success_count += 1
                                except Exception as e:
                                    status.write(f"❌ Failed {del_title}: {e}")
                                    fail_count += 1
                    
                    if success_count > 0:
                        st.success(f"Deleted {success_count} files.")
                        # Refresh logic
                        store_key = _store_session_key_for(target_store)
                        st.session_state.pop(store_key, None)
                        st.session_state.pop("filestores_cached", None)
                        time.sleep(1) # Brief pause to let API propagate
                        st.rerun()
                
                # Button 2: Delete ALL
                if col_d2.button("Delete ALL Files", type="primary"):
                    # We can't easily do a confirmation pop-up in sidebar without rerun, 
                    # so we will make this a direct action with a status indicator for safety/feedback.
                    # Or simpler: trust the user (as requested "clear whole store")
                    
                    with st.status("Emptying store...", expanded=True) as status:
                        for doc in docs_in_store:
                            title = doc.get("title")
                            res_name = doc.get("name")
                            try:
                                status.write(f"Deleting {title}...")
                                delete_file_from_store(client, res_name)
                            except Exception as e:
                                status.write(f"Failed {title}: {e}")
                        status.update(label="Store emptied!", state="complete", expanded=False)
                    
                    st.success("All files deleted.")
                    # Refresh logic
                    store_key = _store_session_key_for(target_store)
                    st.session_state.pop(store_key, None)
                    st.session_state.pop("filestores_cached", None)
                    time.sleep(1)
                    st.rerun()
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

        uploaded_files = st.file_uploader("Choose files (PDF, DOCX, TXT, etc.)", accept_multiple_files=True, key="sidebar_files_uploader")
        
        # Display name logic is tricky for multiple files. We'll disable it or use it as a common prefix if single?
        # Simpler: If multiple files, ignore specific display name or use filename.
        # Let's keep it simple: Optional common prefix/tag? Or just per-file renaming is too complex for this UI.
        # We will hide the display name input if multiple files are selected to avoid confusion, 
        # or just show it but clarify it applies to the single file if only one.
        
        display_name_input = ""
        if uploaded_files and len(uploaded_files) == 1:
            display_name_input = st.text_input("Display name for the file (optional)", key="sidebar_files_display_name")
        elif uploaded_files and len(uploaded_files) > 1:
            st.caption(f"Selected {len(uploaded_files)} files. They will be imported with their filenames.")

        chunk_tokens = 0
        chunk_overlap = 0

        if st.button("Import file(s) into selected store", key="sidebar_files_import_btn"):
            if not uploaded_files:
                st.error("Please choose file(s) to upload.")
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
                    # Sidebar status container for batch progress
                    sidebar_status = st.empty()
                    
                    success_count = 0
                    files_processed = 0
                    total_files = len(uploaded_files)

                    for upload_file in uploaded_files:
                        files_processed += 1
                        current_status_obj = {"text": lambda m: sidebar_status.info(f"[{files_processed}/{total_files}] {upload_file.name}: {m}")}
                        
                        tmp_path = None
                        try:
                            tmp_path = safe_write_tmp(upload_file, upload_file.name)
                        except Exception as e:
                            st.error(f"Failed to save locally '{upload_file.name}': {e}")
                            tmp_path = None

                        if tmp_path:
                            # Config per file
                            cfg = {}
                            if int(chunk_tokens) > 0:
                                cfg["chunking_config"] = {
                                    "white_space_config": {
                                        "max_tokens_per_chunk": int(chunk_tokens),
                                        "max_overlap_tokens": int(chunk_overlap or 0),
                                    }
                                }
                            
                            # Use display name only if single file
                            final_display_name = None
                            if len(uploaded_files) == 1 and display_name_input:
                                final_display_name = display_name_input
                            
                            if final_display_name:
                                cfg["display_name"] = final_display_name

                            try:
                                current_status_obj["text"]("Starting upload...")
                                op = robust_upload_to_store(client, tmp_path, getattr(target_store,"name", str(target_store)), cfg, current_status_obj)
                            except Exception as e:
                                msg = str(e)
                                if "Storage limit exceeded" in msg:
                                    st.error("🚫 **Storage Limit Exceeded**\n\nYou have exhausted your storage quota. Please upgrade your plan or delete old files.\nPlease refer to https://ai.google.dev/gemini-api/docs/file-search#rate-limits for more information on limits.", icon="🚫")
                                    current_status_obj["text"]("Upload failed: Storage limit exceeded.")
                                    # Stop batch on quota error
                                    break 
                                else:
                                    current_status_obj["text"](f"Upload failed: {e}")
                                    st.error(f"Upload failed for {upload_file.name}: {e}")
                                op = None

                            if op:
                                # Wait for import (defaulting to wait for robustness)
                                if not wait_for_import:
                                    st.success(f"'{upload_file.name}' upload queued.")
                                    success_count += 1
                                else:
                                    try:
                                        poll_operation(client, op, current_status_obj, timeout=int(import_timeout or 300), interval=2)
                                        current_status_obj["text"]("Import finished.")
                                        st.success(f"Imported '{upload_file.name}'.")
                                        success_count += 1
                                    except Exception as e:
                                        st.error(f"Error waiting for '{upload_file.name}': {e}")
                            
                        # Cleanup temp
                        try:
                            if tmp_path and os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass

                    # End of batch loop
                    if success_count > 0:
                        st.success(f"Batch import complete: {success_count}/{total_files} files.")
                        
                        # Invalidate caches
                        store_key = _store_session_key_for(target_store)
                        st.session_state.pop(store_key, None)
                        st.session_state.pop("filestores_cached", None)

                        # Auto-refresh
                        try:
                            # Do a quick updated fetch
                            st.session_state["_refreshing_docs"] = True
                            with st.spinner("Refreshing document list..."):
                                new_rows = list_documents_in_store(client, target_store)
                            st.session_state[store_key] = new_rows
                            st.session_state["_refreshing_docs"] = False
                            
                            time.sleep(1)
                            st.rerun()
                        except Exception:
                            st.warning("Could not refresh list automatically, please click 'Refresh documents'.")


# initialize session-state message history (only once)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Main interface Import operation id
chat_container = st.container()

# Placeholder for the grid so we can wipe it immediately when interaction starts
grid_placeholder = st.empty()

# Instructional Grid (Only if no messages)
if not st.session_state["messages"]:
    with grid_placeholder.container():
        st.markdown(" ") # spacer
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(128,128,128,0.05); border-radius: 10px;">
                <div style="font-size: 2rem;">📂</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">Upload</div>
                <div style="font-size: 0.9rem; color: #888;">Upload your PDFs/Docs via the sidebar.</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(128,128,128,0.05); border-radius: 10px;">
                <div style="font-size: 2rem;">🔑</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">Connect</div>
                <div style="font-size: 0.9rem; color: #888;">Enter your Gemini API Key to start.</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(128,128,128,0.05); border-radius: 10px;">
                <div style="font-size: 2rem;">💬</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">Chat</div>
                <div style="font-size: 0.9rem; color: #888;">Ask questions and get cited answers.</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

# Render message history
with chat_container:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Main chat input (single, well-indented block)
prompt = st.chat_input("What is up?")
if prompt:
    # Clear grid immediately
    grid_placeholder.empty()

    # Append to history immediately to avoid UI state mismatch
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # Display user message in container
    with chat_container:
        st.chat_message("user").markdown(prompt)
    
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
                    - If producing JSON or a table, generate the full structure before emitting content.

                    4. Fallback rule:
                    - If the documents do NOT contain the needed information, and you are NOT allowed to use general knowledge, respond exactly with: "I cannot answer from the provided documents."
                    - If general knowledge is allowed (the UI will indicate this), you MAY supplement with general knowledge but MUST mark general-knowledge segments with the prefix [GENERAL].

                    5. Safety & honesty:
                    - Never fabricate page numbers, document names, or direct quotes.

                    6. ANSWER STYLE:
                    - Output must be a minimum of 100–150 words.
                    
                    *** CRITICAL INSTRUCTION FOR COMPARISONS ***
                    If (and only if) the user requests a **comparison**, **difference**, **contrast**, **vs**, **versus**, **distinguish**, **how they are different** or **differentiation**:
                    1. You MUST generate the **Markdown Table FIRST**, before any text summary.
                    2. The table must be the very first thing in your response.
                    3. **Do NOT use bolding or markdown formatting (like **) inside the table cells or headers.** Keep the text clean.
                    4. Follow the table with a short narrative explanation.
                    ********************************************

                    For normal questions (non-comparisons):
                    - Provide a short summary (2–3 sentences) first.
                    - Then provide step-by-step explanation or runnable code if relevant.
                    - Use code fences and specify language for code blocks.
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
                    try:
                        stream = client.models.generate_content_stream(
                            model=model_choice,
                            contents=contents,
                            config=cfg
                        )

                    except Exception as e:
                        handle_streaming_error(e, model_choice)
                else:
                    # Fallback: echoing to simulate a response object shape as minimally as possible
                    class DummyResp:
                        text = f"Echo (no genai): {prompt}"
                    resp = DummyResp()
                    stream = [resp]

                # --- STREAMING RESPONSE HANDLING ---
                full_text = ""
                final_chunk = None
                TYPE_DELAY_SEC = 0.02

                def response_generator(stream, delay=TYPE_DELAY_SEC):
                    for chunk in stream:
                        delta = getattr(chunk, "text", None)
                        if not delta:
                            continue
                        for word in delta.split(" "):
                            yield word + " "
                            time.sleep(delay)

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    uses_file_search = bool(tools_list)

                    thinking_label = "Retrieving documents" if uses_file_search else "Thinking"
                    think_start = time.time()
                    placeholder.markdown(f"_{thinking_label}…_")

                    full_text = ""
                    buffered_chunks = []
                    structured = False

                    try:
                        # Peek
                        for chunk in stream:
                            buffered_chunks.append(chunk)
                            delta = getattr(chunk, "text", None)
                            if delta:
                                full_text += delta
                            if len(full_text) > 120:  # peeking threshold
                                break

                        structured = _looks_like_structured_output(full_text)

                        # PLAIN TEXT
                        if not structured:
                            placeholder.empty()
                            response_text = st.write_stream(
                                response_generator(
                                    chain(buffered_chunks, stream)
                                )
                            )
                            full_text = response_text

                        # STRUCTURED
                        else:
                            placeholder.empty()
                            placeholder.markdown("_Formatting structured output…_")
                            full_text = ""

                            for chunk in chain(buffered_chunks, stream):
                                delta = getattr(chunk, "text", None)
                                if delta:
                                    full_text += delta
                                    placeholder.markdown(full_text)

                            placeholder.markdown(full_text)

                        thinking_time = time.time() - think_start
                        st.caption(f"💭 Thought for {thinking_time:.1f}s")

                    except Exception as e:
                        handle_streaming_error(e, model_choice)

                # persist message
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": full_text or "(no answer returned)"
                })

                # store raw response for debugging (best-effort)
                try:
                    # final_chunk might technically be elusive if we iterated everything,
                    # but we usually don't need it if we have full text.
                    # If needed, we'd have to capture the last chunk in the loop.
                    st.session_state["_last_resp_raw"] = None
                except Exception:
                    pass

                # extract grounding snippets AFTER streaming completes
                snips = []
                try:
                    if hasattr(stream, "response"):
                        snips = get_top_grounding_snippets(stream.response, top_n=5)
                except Exception:
                    pass
                if not snips and selected_store is not None:
                    pass

            except Exception as e:
                # Do NOT append Gemini quota/rate errors to chat
                if st.session_state.pop("_last_gemini_error_handled", False):
                    st.stop()

                with st.chat_message("assistant"):
                    st.markdown(f"Request failed: {e}")

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"Request failed: {e}"
                })
    
    # Rerun to update state and remove grid on next load
    st.rerun()

# Footer
st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; color: #888; font-size: 0.8rem; padding: 10px; pointer-events: none; z-index: 10001;">
        AI can be inaccurate/make mistakes. Please be cautious.
    </div>
""", unsafe_allow_html=True)