# fiken_app.py — Voice Agent + Auto-launch relay + Fiken company picker + Health panel + Optional Weaviate reindex
import os
import sys
import time
import socket
import subprocess
from pathlib import Path
from textwrap import dedent
from datetime import date, timedelta, datetime
from io import BytesIO

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Optional deps (indexing like in your backup) ----------
_HAS_WEAVIATE = True
_HAS_EMB = True
try:
    import weaviate
except Exception:
    _HAS_WEAVIATE = False
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain.embeddings import HuggingFaceEmbeddings
except Exception:
    _HAS_EMB = False

# ---------- Optional TTS ----------
try:
    from gtts import gTTS
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False

load_dotenv()

st.set_page_config(page_title="Fiken Voice Agent", page_icon="🎙️", layout="centered")
st.title("🎙️ Fiken Voice Agent")
st.caption("Hands-free voice with on-screen transcript. The agent can call your invoice search tool mid-conversation.")

# ----------------------------- Helpers -----------------------------
def _abs(u: str) -> str:
    """Resolve :PORT → http://localhost:PORT and keep absolute URLs intact."""
    if not isinstance(u, str):
        return ""
    u = u.strip()
    if u.startswith(("http://", "https://")):
        return u
    if u.startswith(":"):
        return f"http://localhost{u}"
    return u

def tts_bytes(text: str, voice="alloy") -> bytes:
    text = (text or "").strip()
    if not text or not _HAS_GTTS:
        return b""
    try:
        client = OpenAI()
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            format="mp3",
        )
        content = getattr(resp, "content", None)
        if content is None and hasattr(resp, "read"):
            content = resp.read()
        return content or b""
    except Exception:
        return b""

# ---------- Auto-launch local relay (uvicorn realtime_server:app) ----------
RELAY_HOST = os.getenv("RELAY_HOST", "127.0.0.1")
RELAY_PORT = int(os.getenv("RELAY_PORT", "5050"))
RELAY_BASE = f"http://{RELAY_HOST}:{RELAY_PORT}"
RELAY_MODULE = os.getenv("RELAY_MODULE", "realtime_server:app")  # module:app

def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _relay_health_ok() -> bool:
    try:
        r = requests.get(f"{RELAY_BASE}/health", timeout=1.2)
        return r.ok
    except Exception:
        return False

def _start_relay_subprocess() -> subprocess.Popen:
    """
    Launch uvicorn as a child process:
    python -m uvicorn realtime_server:app --host 0.0.0.0 --port 5050
    """
    here = Path(__file__).resolve().parent
    env = os.environ.copy()

    cmd = [
        sys.executable, "-m", "uvicorn", RELAY_MODULE,
        "--host", "0.0.0.0", "--port", str(RELAY_PORT)
    ]
    # For dev hot-reload, add: cmd += ["--reload"]

    # On Windows, avoid spawning a new console window
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
    return subprocess.Popen(
        cmd,
        cwd=str(here),
        env=env,
        stdout=subprocess.DEVNULL,   # set to None to see logs in Streamlit console
        stderr=subprocess.DEVNULL,
        creationflags=creation,
    )

def ensure_relay_running(show_sidebar: bool = True, wait_seconds: float = 6.0) -> str:
    """
    Ensure the local relay is up. Returns a short status string.
    Stores the Popen handle in st.session_state['_relay_proc'].
    """
    # Already healthy?
    if _relay_health_ok():
        status = f"✅ Relay running at {RELAY_BASE}"
    else:
        # Drop dead proc handle
        proc = st.session_state.get("_relay_proc")
        if isinstance(proc, subprocess.Popen) and proc.poll() is not None:
            st.session_state["_relay_proc"] = None

        # Start if needed
        if not _port_open(RELAY_HOST, RELAY_PORT):
            st.session_state["_relay_proc"] = _start_relay_subprocess()

        # Wait a bit
        t0 = time.time()
        while time.time() - t0 < wait_seconds:
            if _relay_health_ok():
                break
            time.sleep(0.35)

        status = "✅ Relay started" if _relay_health_ok() else "⚠️ Relay not reachable"

    if show_sidebar:
        with st.sidebar.expander("Relay (local)"):
            st.write(status)
            st.caption(f"Endpoint base: {RELAY_BASE}")
            col1, col2 = st.columns(2)
            if col1.button("Check health"):
                ok = _relay_health_ok()
                st.success("Healthy") if ok else st.error("Not responding")
            if col2.button("Restart relay"):
                # Try to terminate old proc (if any)
                proc = st.session_state.get("_relay_proc")
                try:
                    if isinstance(proc, subprocess.Popen) and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except Exception:
                            proc.kill()
                except Exception:
                    pass
                st.session_state["_relay_proc"] = _start_relay_subprocess()
                time.sleep(1.0)
                st.rerun()

    return status

# ensure the relay is alive (auto-start if needed)
_ = ensure_relay_running(show_sidebar=True)

# ---- Fiken helpers ----
FIKEN_BASE = "https://api.fiken.no/api/v2"

def _fiken_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

def get_companies_fiken(token: str):
    r = requests.get(f"{FIKEN_BASE}/companies", headers=_fiken_headers(token), timeout=30)
    r.raise_for_status()
    return r.json()

def get_invoices(token: str, slug: str, issued_after: str = None, issued_before: str = None):
    params = {}
    if issued_after:  params["issuedAfter"] = issued_after
    if issued_before: params["issuedBefore"] = issued_before
    r = requests.get(
        f"{FIKEN_BASE}/companies/{slug}/invoices",
        headers=_fiken_headers(token),
        params=params,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

def explain_http_error(e: requests.HTTPError) -> str:
    resp = getattr(e, "response", None)
    if resp is None:
        return f"HTTP error: {e}"
    try:
        j = resp.json()
    except Exception:
        j = {"raw": resp.text}
    if resp.status_code == 403:
        return ("🔒 403 Forbidden — enable *Foretak → Tilleggstjenester → API/Netshop* "
                "or use a Test-company. Details: " + str(j))
    if resp.status_code == 401:
        return "🔑 401 Unauthorized — token expired/invalid."
    return f"HTTP {resp.status_code}: {j}"

# ---- Optional Weaviate helpers (for indexing like the backup) ----
def connect_weaviate_v3():
    url = os.environ.get("WEAVIATE_URL", "").strip().rstrip("/")
    key = os.environ.get("WEAVIATE_API_KEY", "")
    if not url or not key or not _HAS_WEAVIATE:
        raise RuntimeError("Weaviate not configured/installed.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return weaviate.Client(url=url, auth_client_secret=weaviate.AuthApiKey(api_key=key))

def ensure_schema_invoicechunk(client):
    class_name = "InvoiceChunk"
    if client.schema.exists(class_name):
        return
    schema = {
        "classes": [{
            "class": class_name,
            "vectorizer": "none",
            "properties": [
                {"name": "company_slug", "dataType": ["text"]},
                {"name": "doc_type",     "dataType": ["text"]},
                {"name": "invoice_no",   "dataType": ["text"]},
                {"name": "issue_date",   "dataType": ["text"]},
                {"name": "supplier",     "dataType": ["text"]},
                {"name": "currency",     "dataType": ["text"]},
                {"name": "amount_total", "dataType": ["number"]},
                {"name": "vat_total",    "dataType": ["number"]},
                {"name": "text",         "dataType": ["text"]},
                {"name": "pdf_url",      "dataType": ["text"]},
            ],
        }]
    }
    client.schema.create(schema)

def _first(*vals, default=None):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return default

def _parse_date(s):
    if not s:
        return ""
    try:
        return datetime.fromisoformat(str(s)[:10]).date().isoformat()
    except Exception:
        return str(s)

def build_rows_from_invoices(invoices, company_slug: str):
    rows = []
    for inv in invoices:
        invoice_no = str(_first(inv.get("invoiceNumber"), inv.get("number"), inv.get("invoiceId"), default=""))
        issue_date = _parse_date(_first(inv.get("issueDate"), inv.get("issuedDate"), default=""))
        supplier = None
        if isinstance(inv.get("supplier"), dict):
            supplier = inv["supplier"].get("name")
        if not supplier and isinstance(inv.get("customer"), dict):
            supplier = inv["customer"].get("name")
        if not supplier and isinstance(inv.get("sale"), dict):
            c = inv["sale"].get("customer")
            if isinstance(c, dict):
                supplier = c.get("name")
        supplier = _first(supplier, inv.get("yourReference"), inv.get("ourReference"), "Unknown")
        amount = _first(inv.get("grossInNok"), inv.get("gross"), inv.get("netInNok"), inv.get("net"))
        try:
            amount = float(amount) if amount is not None else None
        except Exception:
            amount = None
        currency = _first(inv.get("currency"), "NOK")
        vat_total = _first(inv.get("vatTotal"), inv.get("vat"))
        note = _first(inv.get("invoiceText"), inv.get("comment"), inv.get("description"), default="")
        if inv.get("orderReference"):
            note = (note + f"\nOrderRef: {inv['orderReference']}").strip()
        line_bits = []
        for li in inv.get("lines", []) or []:
            q = li.get("quantity")
            up = li.get("unitPrice")
            pn = li.get("productName") or ""
            desc = li.get("description") or ""
            piece = f"{pn}".strip()
            if desc and desc != pn:
                piece += f" — {desc}"
            if q is not None and up is not None:
                piece += f" (qty {q}, unit {up})"
            if piece:
                line_bits.append(piece)
        lines_text = "\n".join(f"- {b}" for b in line_bits)
        pdf_url = inv.get("invoicePdf", {}).get("downloadUrl") if isinstance(inv.get("invoicePdf"), dict) else None
        summary = [f"Invoice {invoice_no} dated {issue_date} for {supplier}"]
        if amount is not None: summary.append(f"Total {amount:.2f} {currency}")
        if note:              summary.append(f"Note: {note}")
        if lines_text:        summary.append("Line items:\n" + lines_text)
        rows.append({
            "company_slug": company_slug,
            "doc_type": "invoice",
            "invoice_no": invoice_no,
            "issue_date": issue_date,
            "supplier": supplier,
            "currency": currency,
            "amount_total": amount,
            "vat_total": float(vat_total) if isinstance(vat_total, (int, float, str)) and str(vat_total) not in ("", "None") else None,
            "text": "\n".join(summary).strip(),
            "pdf_url": pdf_url,
        })
    return rows

def _embedder():
    model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if model.startswith("text-embedding"):
        return OpenAIEmbeddings(model=model)
    return HuggingFaceEmbeddings(model_name=model)

def batch_upsert_invoices(invoices, company_slug: str) -> int:
    client = connect_weaviate_v3()
    ensure_schema_invoicechunk(client)
    rows = build_rows_from_invoices(invoices, company_slug)
    if not rows:
        return 0
    emb = _embedder()
    inserted = 0
    with client.batch as batch:
        batch.batch_size = 64
        for r in rows:
            vec = emb.embed_query(r["text"] or "invoice")
            client.batch.add_data_object(
                data_object=r,
                class_name="InvoiceChunk",
                vector=vec,
            )
            inserted += 1
    return inserted

def weaviate_count_objects():
    client = connect_weaviate_v3()
    res = client.query.raw("{ Aggregate { InvoiceChunk { meta { count } } } }")
    return int(res["data"]["Aggregate"]["InvoiceChunk"][0]["meta"]["count"])

def weaviate_sample_rows(limit=5):
    client = connect_weaviate_v3()
    res = (
        client.query
        .get("InvoiceChunk", ["invoice_no", "issue_date", "supplier", "amount_total", "currency", "text", "pdf_url"])
        .with_limit(int(limit))
        .do()
    )
    return res.get("data", {}).get("Get", {}).get("InvoiceChunk", [])

# ----------------------------- Defaults -----------------------------
default_session = _abs(os.getenv("REALTIME_SESSION_URL", f"{RELAY_BASE}/session"))
default_tool    = _abs(os.getenv("REALTIME_TOOL_URL",    f"{RELAY_BASE}/tool/invoice_search"))
default_model   = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")

# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Voice Agent Settings")

# Fiken token + company picker
fiken_token = st.sidebar.text_input(
    "Fiken access token",
    value=st.secrets.get("FIKEN_ACCESS_TOKEN", os.getenv("FIKEN_ACCESS_TOKEN", "")),
    type="password"
)

company_slug = ""
company_label = ""
if fiken_token.strip():
    try:
        companies = get_companies_fiken(fiken_token.strip())
        if not companies:
            st.sidebar.info("No companies returned for this token.")
        else:
            label_to_slug = {f"{c.get('name')} ({c.get('slug')})": c.get("slug") for c in companies}
            company_label = st.sidebar.selectbox("Company", sorted(label_to_slug.keys()), index=0)
            company_slug = label_to_slug[company_label]
    except requests.HTTPError as e:
        st.sidebar.error(explain_http_error(e))
    except Exception as e:
        st.sidebar.error(f"Fiken companies error: {e}")
else:
    st.sidebar.info("Enter Fiken token to load companies (test companies included).")

rt_models = ["gpt-4o-realtime-preview-2024-12-17"]
rt_index = rt_models.index(default_model) if default_model in rt_models else 0
realtime_model = st.sidebar.selectbox("Realtime model", rt_models, index=rt_index)

voice = st.sidebar.selectbox("Voice", ["alloy", "verse", "coral", "breeze"], index=0)
session_url = st.sidebar.text_input("Session URL", default_session)
tool_url    = st.sidebar.text_input("Tool URL", default_tool)

send_greeting = st.sidebar.checkbox("Play short greeting on connect", value=True)
greeting_text = st.sidebar.text_input("Greeting text", "Hei! Jeg er klar. Hva vil du vite om fakturaene dine?")

tool_body_wrapped = st.sidebar.checkbox(
    'Send tool body as {"args": {...}} (leave on if your API expects it)', value=True
)

st.sidebar.markdown("---")
st.sidebar.header("Text Chat Settings")
chat_models = ["gpt-4o-mini", "gpt-4.1-mini"]
chat_model = st.sidebar.selectbox("Text model", chat_models, index=0)

default_chat_system = "You are a concise assistant for invoices. Answer in Norwegian when possible."
if company_slug:
    default_chat_system += f" The active Fiken company is '{company_label}' (slug: {company_slug})."
chat_system = st.sidebar.text_area("System prompt", default_chat_system, height=100)

speak_chat = st.sidebar.checkbox("Speak chat replies (TTS)", value=False)
chat_voice = st.sidebar.selectbox("Chat TTS voice", ["alloy", "verse", "coral", "breeze"], index=0)

# ---------- Health & Debug ----------
with st.sidebar.expander("Health & Debug"):
    colA, colB = st.columns(2)
    with colA:
        if st.button("Test Session URL"):
            try:
                r = requests.post(session_url, timeout=15)
                r.raise_for_status()
                st.success("Session OK")
                st.code(r.json())
            except Exception as e:
                st.error(f"Session fetch failed: {e}")
                st.caption("Check HTTPS vs HTTP, CORS, and that your relay returns JSON including client_secret.value")
    with colB:
        if st.button("Test Tool URL"):
            try:
                payload = {"query":"ping"} if not tool_body_wrapped else {"args":{"query":"ping"}}
                r = requests.post(tool_url, json=payload, timeout=15)
                st.write("Status:", r.status_code)
                st.code(r.text[:1000] + ("..." if len(r.text) > 1000 else ""))
                if not (200 <= r.status_code < 300):
                    st.warning("Tool responded but not 2xx — the model may get an error JSON.")
            except Exception as e:
                st.error(f"Tool fetch failed: {e}")
                st.caption("If this fails with CORS in the browser, allow Access-Control-Allow-Origin on your tool server.")

# ----------------------------- Voice Agent UI -----------------------------
st.subheader("🗣️ Live Voice (with transcript)")

wrapped_flag_js = "true" if tool_body_wrapped else "false"
escaped_greeting = greeting_text.replace('"', '\\"')
selected_company_slug_js = (company_slug or "").replace('"', '\\"')
selected_company_label_js = (company_label or "").replace('"', '\\"')

st.components.v1.html(dedent(f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    .tray {{ display:flex; gap:.5rem; align-items:center; margin-bottom:.5rem; flex-wrap:wrap; }}
    .status {{ opacity:.8; }}
    .panel {{ border:1px solid #ddd; border-radius:8px; padding:.75rem; max-height:360px; overflow:auto; }}
    .line {{ margin:.35rem 0; }}
    .role-a {{ font-weight:600; color:#0a5; }}
    .role-t {{ color:#222; }}
    .tool-card {{ background:#fafafa; border:1px solid #eee; border-radius:8px; padding:.5rem .75rem; margin:.4rem 0; font-size:.95rem; }}
    .tool-item {{ margin:.15rem 0; }}
    .muted {{ opacity:.65; }}
    .pill {{ font-size:.85rem; padding:.2rem .5rem; border:1px solid #ddd; border-radius:999px; background:#f7f7f7; }}
    button {{ padding:.4rem .75rem; }}
  </style>
</head>
<body style="font-family:system-ui,-apple-system,Segoe UI,Roboto;line-height:1.45;">
  <div class="tray">
    <button id="startBtn">Start voice</button>
    <button id="stopBtn" disabled>Stop</button>
    <span id="status" class="status"></span>
    <span class="pill" id="companyPill" title="Active company"></span>
  </div>

  <div id="transcript" class="panel" aria-live="polite" aria-label="Transcript"></div>
  <audio id="remoteAudio" autoplay controls playsinline style="display:block;width:100%;margin-top:.5rem;"></audio>

<script>
  let pc = null;
  let dc = null;
  let localStream = null;

  const startBtn = document.getElementById('startBtn');
  const stopBtn  = document.getElementById('stopBtn');
  const remoteAudio = document.getElementById('remoteAudio');
  const transcript = document.getElementById('transcript');
  const companyPill = document.getElementById('companyPill');

  const selectedCompanySlug = "{selected_company_slug_js}";
  const selectedCompanyLabel = "{selected_company_label_js}";
  companyPill.textContent = selectedCompanySlug ? ("Company: " + selectedCompanyLabel) : "Company: (none)";

  remoteAudio.muted = false; remoteAudio.volume = 1.0;

  function setStatus(msg) {{
    const el = document.getElementById('status');
    if (el) el.textContent = msg || '';
    try {{ console.log('[status]', msg); }} catch(_e) {{}}
  }}

  function addAssistantContainer() {{
    const wrap = document.createElement('div');
    wrap.className = 'line';
    const role = document.createElement('span');
    role.className = 'role-a';
    role.textContent = 'Assistant: ';
    const text = document.createElement('span');
    text.className = 'role-t';
    text.textContent = '';
    wrap.appendChild(role); wrap.appendChild(text);
    transcript.appendChild(wrap);
    transcript.scrollTop = transcript.scrollHeight;
    return text;
  }}

  function addInfo(msg) {{
    const line = document.createElement('div');
    line.className = 'line muted';
    line.textContent = msg;
    transcript.appendChild(line);
    transcript.scrollTop = transcript.scrollHeight;
  }}

  function addToolResults(toolOut) {{
    const card = document.createElement('div');
    card.className = 'tool-card';
    const title = document.createElement('div');
    title.innerHTML = '<b>Tool:</b> invoice_search ' + (selectedCompanySlug ? ' <span class="pill">company: ' + selectedCompanyLabel + '</span>' : '');
    card.appendChild(title);

    try {{
      if (toolOut && toolOut.results && Array.isArray(toolOut.results) && toolOut.results.length) {{
        toolOut.results.slice(0, 8).forEach((r) => {{
          const row = document.createElement('div');
          row.className = 'tool-item';
          const inv = r.invoice_no || 'N/A';
          const dt  = r.issue_date || 'N/A';
          const sup = r.supplier || 'N/A';
          const amt = (r.amount_total !== undefined && r.amount_total !== null) ? r.amount_total : 'N/A';
          const cur = r.currency || '';
          const pdf = r.pdf_url ? ` — <a href="${{r.pdf_url}}" target="_blank" rel="noopener">PDF</a>` : '';
          row.innerHTML = `• <b>${{inv}}</b> — ${{dt}} — ${{sup}} — <i>${{amt}} ${{cur}}</i>${{pdf}}`;
          card.appendChild(row);
        }});
      }} else if (toolOut && toolOut.error) {{
        const row = document.createElement('div');
        row.className = 'tool-item';
        row.textContent = 'Error: ' + toolOut.error;
        card.appendChild(row);
      }} else {{
        const row = document.createElement('div');
        row.className = 'tool-item';
        row.textContent = 'No results.';
        card.appendChild(row);
      }}
    }} catch (e) {{
      const row = document.createElement('div');
      row.className = 'tool-item';
      row.textContent = 'Could not render tool output.';
      card.appendChild(row);
    }}

    transcript.appendChild(card);
    transcript.scrollTop = transcript.scrollHeight;
  }}

  const toolSchemas = [{{
    "type": "function",
    "name": "invoice_search",
    "description": "Search indexed invoices (Weaviate) by natural language query.",
    "parameters": {{
      "type": "object",
      "properties": {{
        "query": {{ "type": "string", "description": "Supplier/date/amount/invoice number." }},
        "top_k": {{ "type": "integer", "minimum": 1, "maximum": 20, "default": 5 }},
        "company_slug": {{ "type": "string", "description": "Fiken company slug (scope)." }}
      }},
      "required": ["query"]
    }}
  }}];

  async function start() {{
    try {{
      setStatus("Requesting microphone…");
      localStream = await navigator.mediaDevices.getUserMedia({{ audio: {{
        echoCancellation: true, noiseSuppression: true, autoGainControl: true
      }}}});

      setStatus("Creating session…");
      const sessRes = await fetch("{session_url}", {{ method: "POST" }});
      if (!sessRes.ok) {{
        const t = await sessRes.text().catch(()=>"(no body)");
        throw new Error("Session failed: " + sessRes.status + " " + t);
      }}
      const sess = await sessRes.json();
      const EPHEMERAL_KEY = (sess.client_secret && sess.client_secret.value) || null;
      if (!EPHEMERAL_KEY) throw new Error("Bad session JSON: " + JSON.stringify(sess));

      pc = new RTCPeerConnection();
      pc.onconnectionstatechange = () => setStatus("Connection: " + pc.connectionState);
      pc.oniceconnectionstatechange = () => console.log("ICE:", pc.iceConnectionState);
      pc.ontrack = (evt) => {{
        remoteAudio.srcObject = evt.streams[0];
        remoteAudio.play().catch(()=>{{}});
      }};

      dc = pc.createDataChannel("oai-events");
      let currentTextNode = null;

      dc.onopen = () => {{
        const sessionUpdate = {{
          "type": "session.update",
          "session": {{
            "tools": toolSchemas,
            "voice": "{voice}",
            "instructions": "Use invoice_search for invoice questions and answer concisely." + (selectedCompanySlug ? (" Active company is '" + selectedCompanyLabel + "' (slug: " + selectedCompanySlug + ").") : ""),
            "turn_detection": {{
              "type": "server_vad",
              "threshold": 0.5,
              "prefix_padding_ms": 200,
              "silence_duration_ms": 500
            }},
            "input_audio_transcription": {{
              "enabled": true,
              "model": "whisper-1"
            }},
            "metadata": {{
              "company_slug": selectedCompanySlug
            }}
          }}
        }};
        dc.send(JSON.stringify(sessionUpdate));

        if ({str(send_greeting).lower()}) {{
          currentTextNode = addAssistantContainer();
          dc.send(JSON.stringify({{
            "type": "response.create",
            "response": {{ "instructions": "{escaped_greeting}" }}
          }}));
        }}
      }};

      const callArgs = Object.create(null);
      const callName = Object.create(null);
      const bodyWrapped = {wrapped_flag_js};

      dc.onmessage = async (e) => {{
        try {{
          const msg = JSON.parse(e.data);

          if (msg.type === "response.output_text.delta") {{
            if (!currentTextNode) currentTextNode = addAssistantContainer();
            currentTextNode.textContent += (msg.delta || "");
            transcript.scrollTop = transcript.scrollHeight;
          }}
          if (msg.type === "response.completed" || msg.type === "response.output_text.done") {{
            currentTextNode = null;
          }}

          if (msg.type === "response.function_call.created" && msg.call_id) {{
            callName[msg.call_id] = msg.name || callName[msg.call_id] || "";
            addInfo("Calling tool: " + callName[msg.call_id] + (selectedCompanySlug ? (" (company: " + selectedCompanyLabel + ")") : ""));
          }}

          if (msg.type === "response.function_call_arguments.delta") {{
            const id = msg.call_id;
            callArgs[id] = (callArgs[id] || "") + (msg.delta || "");
            if (!callName[id] && msg.name) callName[id] = msg.name;
          }}

          if (msg.type === "response.function_call_arguments.done") {{
            const id = msg.call_id;
            const name = msg.name || callName[id];
            const argsText = callArgs[id] || "{{}}";
            let args; try {{ args = JSON.parse(argsText); }} catch(_) {{ args = {{}}; }}

            if (name === "invoice_search") {{
              let toolOut = {{"error":"no response"}};
              try {{
                if (selectedCompanySlug && !args.company_slug) args.company_slug = selectedCompanySlug;

                const baseUrl = "{tool_url}";
                const urlWithCompany = selectedCompanySlug ? (baseUrl + (baseUrl.includes("?") ? "&" : "?") + "company_slug=" + encodeURIComponent(selectedCompanySlug)) : baseUrl;

                const body = bodyWrapped ? JSON.stringify({{ args }}) : JSON.stringify(args);
                const res = await fetch(urlWithCompany, {{
                  method: "POST",
                  headers: {{
                    "Content-Type": "application/json",
                    ...(selectedCompanySlug ? {{ "X-Fiken-Company": selectedCompanySlug }} : {{}})
                  }},
                  body
                }});
                toolOut = await res.json();
              }} catch(err) {{
                toolOut = {{"error": String(err)}};
              }}

              addToolResults(toolOut);

              dc.send(JSON.stringify({{
                "type": "conversation.item.create",
                "item": {{
                  "type": "function_call_output",
                  "call_id": id,
                  "output": JSON.stringify(toolOut)
                }}
              }}));
              dc.send(JSON.stringify({{ "type": "response.create", "response": {{}} }}));
            }}
          }}
        }} catch(err) {{
          console.error("dc message error:", err);
        }}
      }};

      localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
      pc.addTransceiver('audio', {{ direction: 'recvonly' }});

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const sdpResponse = await fetch("https://api.openai.com/v1/realtime?model=" +
        encodeURIComponent("{realtime_model}"), {{
        method: "POST",
        headers: {{
          "Authorization": "Bearer " + EPHEMERAL_KEY,
          "Content-Type": "application/sdp",
          "OpenAI-Beta": "realtime=v1"
        }},
        body: offer.sdp
      }});
      if (!sdpResponse.ok) {{
        const t = await sdpResponse.text().catch(()=>"(no body)");
        throw new Error("Realtime SDP failed: " + sdpResponse.status + " " + t);
      }}
      const answerSdp = await sdpResponse.text();
      await pc.setRemoteDescription({{ type: "answer", sdp: answerSdp }});

      startBtn.disabled = true;
      stopBtn.disabled  = false;
      setStatus("Connected. Speak when ready.");
    }} catch (err) {{
      console.error(err);
      setStatus("Error: " + (err.message || err));
      cleanup();
    }}
  }}

  function cleanup() {{
    try {{ if (pc) pc.close(); }} catch (_e) {{}}
    pc = null; dc = null;
    if (localStream) {{ localStream.getTracks().forEach(t => t.stop()); }}
    localStream = null;
    startBtn.disabled = false;
    stopBtn.disabled  = true;
  }}

  startBtn.onclick = start;
  stopBtn.onclick  = () => {{ setStatus("Stopped"); cleanup(); }};
</script>
</body>
</html>
"""), height=560)

# ----------------------------- OPTIONAL: Indexing & Status (Weaviate) -----------------------------
st.subheader("🧰 Optional: Indexing & Status (Weaviate)")
need_envs = not os.getenv("WEAVIATE_URL") or not os.getenv("WEAVIATE_API_KEY")
if not _HAS_WEAVIATE or not _HAS_EMB or need_envs:
    st.info("To enable reindex here, install `weaviate-client<4`, `langchain-openai`, `sentence-transformers` "
            "(or set an OpenAI embedding model) and set `WEAVIATE_URL` + `WEAVIATE_API_KEY`. "
            "Otherwise use your existing backup app.")
else:
    if not fiken_token or not company_slug:
        st.warning("Enter Fiken token and pick a company to sync.")
    else:
        with st.expander("🔎 Preview Fiken API"):
            start_dt = st.date_input("Issued after", date.today() - timedelta(days=365))
            end_dt   = st.date_input("Issued before", date.today() + timedelta(days=1))
            if st.button("Fetch (preview)"):
                try:
                    invs = get_invoices(fiken_token, company_slug, start_dt.isoformat(), end_dt.isoformat())
                    st.success(f"API returned {len(invs)} invoices.")
                    if invs: st.json(invs[0])
                except requests.HTTPError as e:
                    st.error(explain_http_error(e))
                except Exception as e:
                    st.exception(e)

        with st.expander("🔄 Sync to Weaviate"):
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Sync last 3 years → Index"):
                    try:
                        invs = get_invoices(
                            fiken_token, company_slug,
                            (date.today()-timedelta(days=365*3)).isoformat(),
                            (date.today()+timedelta(days=1)).isoformat(),
                        )
                        st.write(f"Fetched from API: **{len(invs)}**")
                        before = weaviate_count_objects()
                        st.write(f"Vector DB count (before): **{before}**")
                        n = batch_upsert_invoices(invs, company_slug)
                        after = weaviate_count_objects()
                        st.success(f"Upserted {n}. Vector DB count (after): **{after}**")
                    except Exception as e:
                        st.exception(e)
            with c2:
                if st.button("⚠️ Wipe & Reindex (90 days)"):
                    try:
                        client = connect_weaviate_v3()
                        if client.schema.exists("InvoiceChunk"):
                            client.schema.delete_class("InvoiceChunk")
                        ensure_schema_invoicechunk(client)
                        invs = get_invoices(
                            fiken_token, company_slug,
                            (date.today()-timedelta(days=90)).isoformat(),
                            (date.today()+timedelta(days=1)).isoformat(),
                        )
                        n = batch_upsert_invoices(invs, company_slug)
                        st.success(f"Rebuilt index and inserted {n} docs.")
                    except Exception as e:
                        st.exception(e)

        with st.expander("🧪 Vector DB status"):
            try:
                cnt = weaviate_count_objects()
                st.write(f"Objects in `InvoiceChunk`: **{cnt}**")
                rows = weaviate_sample_rows(limit=5)
                if rows:
                    st.write("Sample objects:")
                    for r in rows:
                        st.code(
                            f"invoice_no={r.get('invoice_no')} | date={r.get('issue_date')} | "
                            f"supplier={r.get('supplier')} | total={r.get('amount_total')} {r.get('currency')}\n"
                            f"text={r.get('text')}\n"
                            f"pdf={r.get('pdf_url')}"
                        )
                else:
                    st.info("No sample rows returned.")
            except Exception as e:
                st.error(f"Status check failed: {e}")

# ----------------------------- Text Chat -----------------------------
st.subheader("💬 Inkludo Text Chat")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Replay chat
for role, content in st.session_state.chat:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

user_q = st.chat_input("Ask anything about your invoices (or general) …")
if user_q:
    st.session_state.chat.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        answer = ""
        try:
            client = OpenAI()
            msgs = [{"role": "system", "content": chat_system}]
            if company_slug:
                msgs.append({"role": "system", "content": f"Active Fiken company is '{company_label}' (slug: {company_slug})."})
            for r, c in st.session_state.chat:
                msgs.append({"role": r, "content": c})

            try:
                resp = client.responses.create(model=chat_model, input=msgs, temperature=0.2)
                answer = (getattr(resp, "output_text", None) or "").strip()
            except Exception:
                resp = client.chat.completions.create(model=chat_model, messages=msgs, temperature=0.2)
                answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"(Chat failed: {e})"

        st.markdown(answer)
        if speak_chat and answer and not answer.startswith("(Chat failed"):
            audio = tts_bytes(answer, voice=chat_voice)
            if audio:
                st.audio(audio, format="audio/mp3")

    st.session_state.chat.append(("assistant", answer))
