#!/usr/bin/env python3
"""Small deterministic backend fixture; never performs inference."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from socketserver import TCPServer
import time


parser = argparse.ArgumentParser()
parser.add_argument("--host")
parser.add_argument("--port", type=int)
parser.add_argument("--api-key")
parser.add_argument("--model")
parser.add_argument("--n-gpu-layers")
parser.add_argument("--fit-target")
parser.add_argument("--fit", nargs="?", const="enabled")
parser.add_argument("--fit-margin")
parser.add_argument("--no-mmap", action="store_true")
parser.add_argument("--threads")
parser.add_argument("--threads-batch")
parser.add_argument("--load-mode", default="auto")
parser.add_argument("--moe-cache-mib", default="0")
parser.add_argument("--cpu-moe", action="store_true")
parser.add_argument("--poll")
parser.add_argument("--no-op-offload", action="store_true")
parser.add_argument("--no-repack", action="store_true")
parser.add_argument("--no-host", action="store_true")
parser.add_argument("--ctx-size")
parser.add_argument("--model-draft")
parser.add_argument("--spec-type")
parser.add_argument("--spec-draft-n-max")
parser.add_argument("--gpu-layers-draft")
parser.add_argument("--cache-type-k")
parser.add_argument("--cache-type-v")
parser.add_argument("--flash-attn")
parser.add_argument("--log-verbosity")
parser.add_argument("--jinja", action="store_true")
args, _ = parser.parse_known_args()
stats = {"tokens": 0, "decoded": 0}
if Path(args.model).stem == "fail":
    print("Unsupported model architecture (test fixture).", flush=True)
    sys.exit(7)
if (Path(args.model).stem == "tuning-cache-limit" and os.getenv("VRAMBRIDGE_HYBRID_PLAN") and
        int(args.moe_cache_mib) > 4096):
    print("CUDA cache allocation failed: out of memory (test fixture).", flush=True)
    sys.exit(7)
if Path(args.model).stem == "slow":
    time.sleep(5)
else:
    time.sleep(0.15)
if Path(args.model).stem == "gpu-fail":
    print("ggml_cuda_init: failed to initialize CUDA: test driver unavailable", flush=True)
print(json.dumps({
    "pid": os.getpid(),
    "gpu_layers": args.n_gpu_layers,
    "fit_target": args.fit_target,
    "fit": args.fit,
    "fit_margin": args.fit_margin,
    "no_mmap": args.no_mmap,
    "threads": args.threads,
    "batch_threads": args.threads_batch,
    "load_mode": args.load_mode,
    "expert_cache_mib": args.moe_cache_mib,
    "cpu_moe": args.cpu_moe,
    "poll": args.poll,
    "no_op_offload": args.no_op_offload,
    "op_offload_min_batch": os.getenv("GGML_OP_OFFLOAD_MIN_BATCH"),
    "draft_model": args.model_draft,
    "spec_type": args.spec_type,
    "draft_tokens": args.spec_draft_n_max,
    "draft_gpu_layers": args.gpu_layers_draft,
    "inherited_override": "LLAMA_ARG_MODEL" in os.environ,
    "managed_memory": "GGML_CUDA_ENABLE_UNIFIED_MEMORY" in os.environ,
    "hybrid_plan": os.getenv("VRAMBRIDGE_HYBRID_PLAN"),
    "hybrid_adapt": os.getenv("VRAMBRIDGE_HYBRID_ADAPT"),
    "hybrid_async": os.getenv("VRAMBRIDGE_HYBRID_ASYNC"),
    "hybrid_direct": os.getenv("VRAMBRIDGE_HYBRID_DIRECT"),
    "hybrid_batch": os.getenv("VRAMBRIDGE_HYBRID_BATCH"),
    "hybrid_mtp": os.getenv("VRAMBRIDGE_HYBRID_MTP"),
    "hybrid_repack": os.getenv("VRAMBRIDGE_HYBRID_REPACK"),
    "hybrid_stage_input": os.getenv("VRAMBRIDGE_HYBRID_STAGE_INPUT"),
    "hybrid_overlap": os.getenv("VRAMBRIDGE_HYBRID_OVERLAP"),
    "hybrid_boundary": os.getenv("VRAMBRIDGE_HYBRID_BOUNDARY"),
    "hybrid_fuse_gate_up": os.getenv("VRAMBRIDGE_HYBRID_FUSE_GATE_UP"),
    "hybrid_async_ids": os.getenv("VRAMBRIDGE_HYBRID_ASYNC_IDS"),
    "hybrid_hosted": os.getenv("VRAMBRIDGE_HYBRID_HOSTED"),
    "hybrid_policy": os.getenv("VRAMBRIDGE_HYBRID_POLICY"),
    "hybrid_lfu": [os.getenv("VRAMBRIDGE_HYBRID_LFU_FILL"), os.getenv("VRAMBRIDGE_HYBRID_LFU_ADMIT")],
    "host_mailbox": os.getenv("GGML_CUDA_HOST_MAILBOX"),
    "register_host": os.getenv("GGML_CUDA_REGISTER_HOST"),
    "cuda_plugin": os.getenv("GGML_BACKEND_PATH"),
    "no_repack": args.no_repack,
    "no_host": args.no_host,
    "cache_type_k": args.cache_type_k,
    "cache_type_v": args.cache_type_v,
    "flash_attn": args.flash_attn,
    "jinja": args.jinja,
}), flush=True)
cache = "f16" if Path(args.model).stem == "q8-cache-fallback" else args.cache_type_k or "f16"
if args.log_verbosity == "4":
    print(f"llama_kv_cache_unified: size = 1 MiB, K ({cache}): 1 MiB, V ({cache}): 1 MiB", flush=True)
    print(f"llama_context: flash_attn = {'enabled' if args.flash_attn == 'on' else 'disabled'}", flush=True)
    print("api_keys: [fixture marker]", flush=True)
if os.getenv("VRAMBRIDGE_HYBRID_PLAN") and Path(args.model).stem != "hybrid-no-markers":
    plan = os.environ["VRAMBRIDGE_HYBRID_PLAN"]
    selected = 0 if plan == "auto" else sum(bool(line) and not line.startswith("#")
                                          for line in Path(plan).read_text().splitlines())
    print(f"impl: EXPERIMENTAL HYBRID enabled: {selected} selected experts; adaptive admissions", flush=True)
    if os.getenv("VRAMBRIDGE_HYBRID_DIRECT") == "1":
        print("pin: registered 61073604608 immutable source bytes in 73 ranges; cap 64 GiB, headroom 24 GiB", flush=True)
        print("impl: direct asynchronous admissions; immutable registered sources, no staging copies", flush=True)
    else:
        print("impl: adaptive admissions enabled; 1024-byte bounded staging, asynchronous uploads", flush=True)
    coverage = "1/2" if Path(args.model).stem == "hybrid-partial-coverage" else "2/2"
    if os.getenv("VRAMBRIDGE_HYBRID_ASYNC_IDS") == "1" and Path(args.model).stem != "hybrid-no-async-ids":
        print("impl: asynchronous GPU ID handoff: 1024-byte per-layer pinned staging; protected reuse", flush=True)
    print(f"impl: tensor-derived hybrid coverage={coverage} routed layers; other layers remain native", flush=True)
    if os.getenv("VRAMBRIDGE_HYBRID_MTP") == "1" and Path(args.model).stem != "hybrid-mtp-no-batch":
        print(f"impl: hybrid batch limit={os.environ['VRAMBRIDGE_HYBRID_BATCH']}; multi-position dispatch freezes residency", flush=True)
    hosted = os.getenv("VRAMBRIDGE_HYBRID_HOSTED") == "1"
    if os.getenv("VRAMBRIDGE_HYBRID_POLICY") == "lfu":
        print("impl: admission policy=lfu; recent token window=64", flush=True)
    if hosted and Path(args.model).stem != "hybrid-hosted-missing":
        print("init_hosted: VRAMBridge hosted MoE enabled: 2 layers; device slot maps, in-graph host expert calls, "
              "deferred admissions", flush=True)
    if hosted:
        if os.getenv("VRAMBRIDGE_HYBRID_FUSE_GATE_UP") == "1":
            print("VRAMBridge resident gate/up projection fusion active", flush=True)
    elif os.getenv("VRAMBRIDGE_HYBRID_OVERLAP") == "1" and Path(args.model).stem != "hybrid-no-pipeline":
        print("VRAMBridge decode pipeline configured: staged input, guarded overlap, GPU boundary, gate/up fusion", flush=True)
        if Path(args.model).stem != "hybrid-pipeline-pending":
            print("VRAMBridge guarded expert overlap active", flush=True)
            print("VRAMBridge resident gate/up projection fusion active", flush=True)
    context = "8192" if Path(args.model).stem == "hybrid-wrong-context" else args.ctx_size
    print(f"srv: initializing, n_slots = 1, n_ctx_slot = {context}, kv_unified = 'false'", flush=True)
elif Path(args.model).stem == "context-cap":
    print("srv: initializing, n_slots = 1, n_ctx_slot = 2048, kv_unified = 'false'", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *values):
        pass

    def send_json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        if self.headers.get("Authorization") != f"Bearer {args.api_key}":
            self.send_json(401, {"error": {"message": "Invalid backend key."}})
            return False
        return True

    def do_GET(self):
        if self.authorized():
            if self.path == "/slots":
                self.send_json(200, [{"id": 0, "n_ctx": int(args.ctx_size), "n_prompt_tokens": stats["tokens"],
                                     "speculative": bool(args.model_draft and args.spec_type) and Path(args.model).stem != "mtp-fallback",
                                     "next_token": [{"n_decoded": stats["decoded"]}]}])
            else:
                self.send_json(200, {"object": "list", "data": [{"id": "local-model"}]})

    def do_POST(self):
        if not self.authorized():
            return
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self.send_json(200, {"tokens": list(data["content"].encode())})
            return
        if self.path == "/detokenize":
            self.send_json(200, {"content": bytes(data["tokens"]).decode()})
            return
        prompt = data["messages"][-1]["content"]
        if prompt.startswith("SYNAPSE_TUNING_CHECK"):
            content = json.dumps({"start": "cedar4821", "middle": "violet7319", "end": "quartz9067"})
            if Path(args.model).stem == "tuning-wrong-answer":
                content = '{"start":"wrong"}'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {"choices": [{"delta": {"content": content}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": len(prompt.encode()), "completion_tokens": 43},
                 "timings": {"prompt_ms": 10, "predicted_ms": 10}},
            ]
            for event in events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        stats.update(tokens=18, decoded=0)
        if prompt == "backend-error":
            self.send_json(400, {"error": {"message": "Context limit exceeded."}})
            return
        if data.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                def send_event(value):
                    self.wfile.write(f"data: {json.dumps(value)}\n\n".encode())
                    self.wfile.flush()

                if prompt in ("reasoning", "reasoning-limit"):
                    send_event({"choices": [{"delta": {"reasoning_content": "Working on the answer."}}],
                                "timings": {"predicted_n": 100, "predicted_per_second": 125.5}})
                    time.sleep(0.1)
                texts = () if prompt == "reasoning-limit" else ("Hello", " ", "<script>not executable</script>", " \u2713")
                for text in texts:
                    if prompt == "wait":
                        time.sleep(0.5)
                    send_event({"choices": [{"delta": {"content": text}}]})
                    stats["tokens"] += 1
                    stats["decoded"] += 1
                if prompt != "incomplete":
                    send_event({
                        "choices": [{"delta": {}, "finish_reason": "length" if prompt == "reasoning-limit" else "stop"}],
                        "timings": {"predicted_n": 730, "predicted_per_second": 123.5},
                    })
                    if data.get("stream_options", {}).get("include_usage"):
                        send_event({"choices": [], "usage": {"completion_tokens": 730, "prompt_tokens": 18}})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_json(200, {
                "model": data["model"],
                "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            })


class FixtureServer(ThreadingHTTPServer):
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


FixtureServer((args.host, args.port), Handler).serve_forever()
