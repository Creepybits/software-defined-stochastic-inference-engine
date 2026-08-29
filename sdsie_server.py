# sdsie_server.py
"""
SDSIE Open-Source Reference Server
High-efficiency autoregressive inference with real-time Schmitt-trigger entropy telemetry
and automated JSON session logging.

WARNING (added 2026-08-29, not yet fixed): this is the confirmed source of the paper's
Fig. 1 telemetry (telemetry_trace schema: step/mode/k_draft/entropy_bits matches exactly).
The entropy values and clutch decisions (k_draft) ARE real - computed correctly from real
model logits via the verified SDSIESpeculativeController. However, the generation loop
below never branches on k_proposal - every step is one plain cached forward pass + argmax,
regardless of what the clutch decided. No scout model, no draft/verify cycle. So:
  - entropy_bits in the saved trace: genuine, trustworthy.
  - k_draft / mode in the saved trace: genuine clutch *decisions*, but they never affected
    generation - the run was ordinary single-model decoding throughout.
  - throughput_tok_s in the saved trace: reflects plain generation speed, NOT accelerated
    speculative decoding, despite variable names implying otherwise.
Same logged-but-not-acted-on pattern as sweep_real_model.py, harness_telemetry.py, and
cognitive_benchmark.py - see SDSIE_project_status.md, 'compute a signal, log it, never act
on it'. Do not cite tok/s or energy figures from this script as speculative-decoding
performance. The entropy trace itself is usable for a corrected, honestly-captioned Fig. 1.
"""

import os
import time
import json
import argparse
import datetime
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm_sdsie.spec_decode.sdsie_speculator import SDSIESpeculativeController


def parse_args():
    parser = argparse.ArgumentParser(description="SDSIE Open Reference Server")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model ID or path")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--theta-low", type=float, default=0.55, help="Schmitt-trigger lower re-engagement threshold")
    parser.add_argument("--theta-high", type=float, default=1.25, help="Schmitt-trigger upper disengagement threshold")
    parser.add_argument("--draft-k", type=int, default=5, help="Default speculative draft window size (k)")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory to save telemetry logs (default: tools/telemetry/sessions next to this file)")
    return parser.parse_args()


args = parse_args()
if args.log_dir is None:
    args.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "telemetry", "sessions")
os.makedirs(args.log_dir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 68)
print(f"⚡ SDSIE REFERENCE INFERENCE ENGINE (STOCHASTIC SPECULATION)")
print("=" * 68)
print(f"[*] Target Model   : {args.model}")
print(f"[*] Compute Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"[*] Entropy Bounds : θ_low = {args.theta_low:.2f} bits | θ_high = {args.theta_high:.2f} bits")
print(f"[*] Speculation (k): {args.draft_k} draft tokens")
print(f"[*] Telemetry Logs : {args.log_dir}")
print("=" * 68)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()

controller = SDSIESpeculativeController(
    default_k=args.draft_k,
    theta_low=args.theta_low,
    theta_high=args.theta_high
)


class SDSIEOpenAIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req = json.loads(body.decode("utf-8"))

            messages = req.get("messages", [])
            max_tokens = req.get("max_tokens", 512)

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]

            print(f"\n[SDSIE Request Inbound] Prompt Tokens: {prompt_len}")

            generated_tokens = []
            telemetry_trace = []
            past_key_values = None
            current_input_ids = inputs.input_ids
            
            start_time = time.perf_counter()
            tokens_generated = 0

            with torch.no_grad():
                for step in range(max_tokens):
                    outputs = model(
                        input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]

                    # Entropy clutch decision
                    k_proposal = controller.plan_speculation_step(next_token_logits)
                    entropy = controller.clutch.running_entropy
                    mode_str = "SPECULATIVE" if k_proposal > 0 else "SINGLE_STEP_FALLBACK"

                    # Log step telemetry
                    telemetry_trace.append({
                        "step": tokens_generated + 1,
                        "mode": mode_str,
                        "k_draft": k_proposal,
                        "entropy_bits": round(entropy, 4)
                    })

                    # Greedy token selection
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    token_id = next_token.item()
                    generated_tokens.append(token_id)
                    tokens_generated += 1

                    # Real-time console telemetry every 10 steps
                    if tokens_generated % 10 == 0 or token_id == tokenizer.eos_token_id:
                        icon = "🚀 SPECULATIVE (k=5)" if k_proposal > 0 else "🛡️ SINGLE-STEP FALLBACK"
                        print(f"  [Step {tokens_generated:03d}] {icon} | Entropy: {entropy:.3f} bits")

                    if token_id == tokenizer.eos_token_id:
                        break

                    current_input_ids = next_token

            elapsed = time.perf_counter() - start_time
            tps = tokens_generated / elapsed if elapsed > 0 else 0
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"[SDSIE Complete] Generated {tokens_generated} tokens in {elapsed:.2f}s ({tps:.2f} tok/s)\n")

            # --- AUTOMATED TELEMETRY SAVING ---
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": args.model,
                "prompt_tokens": prompt_len,
                "completion_tokens": tokens_generated,
                "total_tokens": prompt_len + tokens_generated,
                "elapsed_seconds": round(elapsed, 3),
                "throughput_tok_s": round(tps, 2),
                "messages": messages,
                "output_text": output_text,
                "telemetry_trace": telemetry_trace
            }
            log_filename = os.path.join(args.log_dir, f"sdsie_trace_{timestamp_str}.json")
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(session_record, f, indent=2, ensure_ascii=False)
            print(f"[Telemetry Saved] -> {log_filename}")

            response_payload = {
                "id": f"sdsie-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": args.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": tokens_generated,
                    "total_tokens": prompt_len + tokens_generated
                }
            }
            self._send_json(200, response_payload)
        else:
            self._send_json(404, {"error": "Endpoint not found"})

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    print(f"📡 SDSIE API Endpoint: http://localhost:{args.port}/v1/chat/completions\n")
    server = HTTPServer(("0.0.0.0", args.port), SDSIEOpenAIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down SDSIE Server...")
        server.server_close()