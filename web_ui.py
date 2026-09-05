# web_ui.py
"""
SDSIE Web Interface
A lightweight, dependency-free local Web UI connecting directly to SDSIE's OpenAI-compatible server.
"""

import os
import json
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- CONFIGURATION ---
SDSIE_API_URL = "http://localhost:8000/v1/chat/completions"
SERVER_PORT = 5000
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SDSIE // Inference UI</title>
<style>
  :root {
    --bg-dark: #090c09;
    --panel-bg: #101510;
    --border-color: #1a261a;
    --green-primary: #0e6b0e;
    --green-hover: #158a15;
    --green-glow: #16db65;
    --text-main: #e2e8e2;
    --text-muted: #758a75;
    --font-mono: 'Consolas', 'Fira Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg-dark); color: var(--text-main); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 24px; display: flex; justify-content: center; }
  .container { width: 100%; max-width: 1000px; display: flex; flex-direction: column; gap: 16px; }
  header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #233d23; padding-bottom: 12px; }
  h1 { font-size: 1.3rem; color: #fff; letter-spacing: 0.5px; font-weight: 700; }
  .badge { font-family: var(--font-mono); font-size: 0.75rem; padding: 4px 8px; border-radius: 4px; background: #142014; border: 1px solid var(--green-primary); color: var(--green-glow); }
  .panel { background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 6px; padding: 16px; }
  .controls-bar { display: flex; justify-content: space-between; align-items: center; }
  textarea { width: 100%; background: #060806; border: 1px solid #1c2b1c; border-radius: 4px; color: #fff; font-size: 0.95rem; padding: 12px; resize: vertical; font-family: inherit; }
  textarea:focus { outline: none; border-color: var(--green-glow); }
  .btn-group { display: flex; gap: 10px; }
  button { background: var(--green-primary); color: #fff; border: 1px solid #233d23; padding: 10px 22px; font-size: 0.9rem; font-weight: 600; border-radius: 4px; cursor: pointer; transition: background 0.15s ease; }
  button:hover { background: var(--green-hover); }
  button.secondary-btn { background: #182218; border-color: #273827; color: #9bb09b; }
  button.secondary-btn:hover { background: #223022; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .label-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 8px; display: flex; justify-content: space-between; }
  #outputBox { background: #060806; color: #f0f0f0; font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; min-height: 220px; padding: 16px; border: 1px solid var(--border-color); }
  .status-footer { font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono); }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>SDSIE // SOVEREIGN ENGINE</h1>
    <span class="badge">Reference UI (backend capability varies - see server logs)</span>
  </header>

  <div class="panel controls-bar">
    <span class="status-footer" id="statusIndicator">Connected to http://localhost:8000</span>
    <div class="btn-group">
      <button class="secondary-btn" onclick="clearSession()">Clear History</button>
      <button id="genBtn" onclick="runGenerate()">Generate</button>
    </div>
  </div>

  <div class="panel">
    <div class="label-title">Prompt Input</div>
    <textarea id="promptInput" rows="4" placeholder="Enter instructions or query for SDSIE..."></textarea>
  </div>

  <div class="panel">
    <div class="label-title"><span>Generated Response</span><span id="genStatus">Idle</span></div>
    <div id="outputBox">Standing by...</div>
  </div>
</div>

<script>
let conversationHistory = [];

async function runGenerate() {
  const promptInput = document.getElementById('promptInput');
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  const genBtn = document.getElementById('genBtn');
  const outputBox = document.getElementById('outputBox');
  const genStatus = document.getElementById('genStatus');

  genBtn.disabled = true;
  genBtn.innerText = 'Generating...';
  genStatus.innerText = 'Running Triton kernels...';
  outputBox.innerText = 'Inferencing...';

  conversationHistory.push({ role: "user", content: prompt });

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ messages: conversationHistory })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    outputBox.innerText = data.output;
    conversationHistory.push({ role: "assistant", content: data.output });
    genStatus.innerText = 'Done (' + data.usage.completion_tokens + ' tokens)';
    promptInput.value = '';
  } catch (err) {
    outputBox.innerText = 'Error: ' + err.message;
    genStatus.innerText = 'Failed';
  } finally {
    genBtn.disabled = false;
    genBtn.innerText = 'Generate';
  }
}

function clearSession() {
  conversationHistory = [];
  document.getElementById('outputBox').innerText = 'Session history cleared.';
  document.getElementById('genStatus').innerText = 'Idle';
}
</script>
</body>
</html>
"""

class WebUIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            payload = {
                "model": MODEL_ID,
                "messages": data.get("messages", []),
                "max_tokens": 1024,
                "temperature": 0.6
            }

            try:
                resp = requests.post(SDSIE_API_URL, json=payload, timeout=60)
                if resp.status_code == 200:
                    res_json = resp.json()
                    output_text = res_json["choices"][0]["message"]["content"]
                    usage = res_json.get("usage", {})
                    self._send_json(200, {"output": output_text, "usage": usage})
                else:
                    self._send_json(resp.status_code, {"error": f"Server Error: {resp.text}"})
            except Exception as e:
                self._send_json(500, {"error": f"Failed to reach SDSIE server on {SDSIE_API_URL}: {str(e)}"})
        else:
            self.send_error(404, "Unknown Route")

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    print("=" * 60)
    print(f"🚀 SDSIE Web UI running on http://localhost:{SERVER_PORT}")
    print(f"📡 Forwarding to SDSIE Engine on {SDSIE_API_URL}")
    print("=" * 60)
    server = HTTPServer(('0.0.0.0', SERVER_PORT), WebUIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web UI...")
        server.server_close()