"""Python driver for the steerability family survey (robust under nohup).
Waits for the GPU to free, then subprocess-runs deception_screen.py per model."""
import os, subprocess, time, json, glob

R = "/workspace/turnstile"
MODELS = os.environ.get("MODELS", "").split() or [
    "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-2b-it", "google/gemma-2-9b-it",
    "google/gemma-2-27b-it", "microsoft/Phi-3.5-mini-instruct",
]
env = dict(os.environ, PYTHONPATH=R, HF_HUB_ENABLE_HF_TRANSFER="0", R=R,
           HF_TOKEN=open("/root/.hf_token").read().strip(),
           HUGGING_FACE_HUB_TOKEN=open("/root/.hf_token").read().strip(),
           LAYER_FRAC="0.4", N_DIR="100", N_TEST="30")

def gpu_busy():
    out = subprocess.run("ps aux | grep -E '[d]eception_(qwen|mask|screen\\.py)'", shell=True,
                         capture_output=True, text=True).stdout
    return bool(out.strip())

while gpu_busy():
    print("waiting for GPU...", flush=True); time.sleep(30)

for m in MODELS:
    tag = m.split("/")[-1].replace(".", "").lower()
    if os.path.exists(f"{R}/experiments/deception_qwen/screen_{tag}.json"):
        print(f"SKIP {tag}", flush=True); continue
    print(f"=== {time.strftime('%H:%M')} SCREEN {m} ===", flush=True)
    log = f"/workspace/screen_{tag}.log"
    with open(log, "w") as lf:
        subprocess.run(["python3", "-u", "scripts/deception_screen.py"],
                       env=dict(env, SCREEN_MODEL=m), cwd=R, stdout=lf, stderr=subprocess.STDOUT, timeout=2400)
    res = subprocess.run(f"grep -E 'SCREEN_RESULT|Error|OutOfMemory|gated|401|403|Traceback' {log} | tail -3",
                         shell=True, capture_output=True, text=True).stdout
    print(res.strip() or "(no result line)", flush=True)
print("SURVEY_DONE", flush=True)
