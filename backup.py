# fiken_app.py — Voice Agent + Sidebar controls + Text Chat (optional TTS)
import os
from textwrap import dedent
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="Fiken Voice Agent", page_icon="🎙️", layout="centered")
st.title("🎙️ Fiken Voice Agent")
st.caption("Talk hands-free, or use text chat. The agent can call your invoice search tool mid-conversation.")

# ----------------------------- Helpers -----------------------------
def _abs(u: str) -> str:
    """Resolve :PORT → http://localhost:PORT and keep absolute URLs intact."""
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if u.startswith(":"):
        return f"http://localhost{u}"
    return u

def tts_bytes(text: str, voice="alloy") -> bytes:
    """Small helper for speaking text chat replies (OpenAI TTS)."""
    if not text.strip():
        return b""
    try:
        client = OpenAI()
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            format="mp3",
        )
        # SDK returns a binary-like object; get raw bytes
        content = getattr(resp, "content", None)
        if content is None and hasattr(resp, "read"):
            content = resp.read()
        return content or b""
    except Exception:
        return b""

# Default URLs from env (can be edited in Sidebar)
default_session = _abs(os.getenv("REALTIME_SESSION_URL", "http://localhost:5050/session"))
default_tool    = _abs(os.getenv("REALTIME_TOOL_URL",    "http://localhost:5050/tool/invoice_search"))
default_model   = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")

# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Voice Agent Settings")
realtime_model = st.sidebar.selectbox(
    "Realtime model",
    [
        "gpt-4o-realtime-preview-2024-12-17",
        # add other preview names here if you have access
    ],
    index=max(0, ["gpt-4o-realtime-preview-2024-12-17"].index(default_model)
              if default_model in ["gpt-4o-realtime-preview-2024-12-17"] else 0),
)
voice = st.sidebar.selectbox("Voice", ["alloy", "verse", "coral", "breeze"], index=0)

session_url = st.sidebar.text_input("Session URL", default_session)
tool_url    = st.sidebar.text_input("Tool URL", default_tool)

send_greeting = st.sidebar.checkbox("Play short greeting on connect", value=True)
greeting_text = st.sidebar.text_input("Greeting text", "Hei! Jeg er klar. Hva vil du vite om fakturaene dine?")

st.sidebar.markdown("---")
st.sidebar.header("Text Chat Settings")
chat_model = st.sidebar.selectbox("Text model", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)
chat_system = st.sidebar.text_area(
    "System prompt",
    "You are a concise assistant for invoices. Answer in Norwegian when possible.",
    height=80,
)
speak_chat = st.sidebar.checkbox("Speak chat replies (TTS)", value=False)
chat_voice = st.sidebar.selectbox("Chat TTS voice", ["alloy", "verse", "coral", "breeze"], index=0)

# ----------------------------- Voice Agent UI -----------------------------
st.subheader("🗣️ Live Voice")
st.components.v1.html(dedent(f"""
<!DOCTYPE html>
<html>
<body style="font-family:system-ui,-apple-system,Segoe UI,Roboto;line-height:1.45;">
  <div style="margin-bottom:8px">
    <button id="startBtn">Start voice</button>
    <button id="stopBtn" disabled>Stop</button>
    <span id="status" style="margin-left:10px;opacity:.85;"></span>
  </div>
  <audio id="remoteAudio" autoplay controls playsinline style="width:100%;"></audio>

<script>
  let pc = null;
  let dc = null;
  let localStream = null;

  const startBtn = document.getElementById('startBtn');
  const stopBtn  = document.getElementById('stopBtn');
  const remoteAudio = document.getElementById('remoteAudio');
  remoteAudio.muted = false; remoteAudio.volume = 1.0;

  function setStatus(msg) {{
    const el = document.getElementById('status');
    if (el) el.textContent = msg || '';
    try {{ console.log('[status]', msg); }} catch(_e) {{}}
  }}

  const toolSchemas = [{{
    "type": "function",
    "name": "invoice_search",
    "description": "Search indexed invoices (Weaviate) by natural language query.",
    "parameters": {{
      "type": "object",
      "properties": {{
        "query": {{ "type": "string", "description": "Supplier/date/amount/invoice number." }},
        "top_k": {{ "type": "integer", "minimum": 1, "maximum": 20, "default": 5 }}
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

      // --- WebRTC ---
      pc = new RTCPeerConnection();
      pc.onconnectionstatechange = () => setStatus("Connection: " + pc.connectionState);
      pc.oniceconnectionstatechange = () => console.log("ICE:", pc.iceConnectionState);
      pc.ontrack = (evt) => {{
        remoteAudio.srcObject = evt.streams[0];
        remoteAudio.play().catch(()=>{{}});
      }};

      // Data channel (events + tool calls)
      dc = pc.createDataChannel("oai-events");
      dc.onopen = () => {{
        // Advertise tools and preferred voice each time we connect
        dc.send(JSON.stringify({{
          "type": "session.update",
          "session": {{
            "tools": toolSchemas,
            "voice": "{voice}",
            "instructions": "Use invoice_search for invoice questions and answer concisely."
          }}
        }}));
        // Optional: greeting so we hear audio immediately
        if ({str(send_greeting).lower()}) {{
          dc.send(JSON.stringify({{
            "type": "response.create",
            "response": {{ "instructions": "{greeting_text.replace('"','\\\\"')}" }}
          }}));
        }}
      }};

      // Tool-call handling
      const callArgs = Object.create(null);
      const callName = Object.create(null);

      dc.onmessage = async (e) => {{
        try {{
          const msg = JSON.parse(e.data);

          if (msg.type === "response.function_call.created" && msg.call_id) {{
            callName[msg.call_id] = msg.name || callName[msg.call_id] || "";
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
                const res = await fetch("{tool_url}", {{
                  method: "POST",
                  headers: {{ "Content-Type": "application/json" }},
                  body: JSON.stringify({{ args }})
                }});
                toolOut = await res.json();
              }} catch(err) {{
                toolOut = {{"error": String(err)}};
              }}

              // Return tool result and ask model to continue speaking
              dc.send(JSON.stringify({{
                "type": "conversation.item.create",
                "item": {{ "type": "function_call_output", "call_id": id, "output": JSON.stringify(toolOut) }}
              }}));
              dc.send(JSON.stringify({{ "type": "response.create", "response": {{}} }}));
            }}
          }}
        }} catch (err) {{
          console.error("dc message error:", err);
        }}
      }};

      // Send mic; receive model audio
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
"""), height=260)

# ----------------------------- Text Chat -----------------------------
st.subheader("💬 Text Chat")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

user_q = st.chat_input("Ask anything about your invoices (or general) …")
if user_q:
    st.session_state.chat.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        try:
            client = OpenAI()
            messages = [{"role": "system", "content": chat_system}]
            for r, c in st.session_state.chat:
                messages.append({"role": r, "content": c})

            # Simple text completion
            resp = client.chat.completions.create(
                model=chat_model,
                messages=messages,
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"(Chat failed: {e})"

        st.markdown(answer)
        if speak_chat and answer and answer != "(Chat failed:":
            audio = tts_bytes(answer, voice=chat_voice)
            if audio:
                st.audio(audio, format="audio/mp3")

    st.session_state.chat.append(("assistant", answer))
