"""Build TWO honesty/lying directions on Llama-3.1-8B and compare them:
  (A) RepE-style: same TRUE statements read under an HONEST vs DISHONEST persona
      -> contrast activations (matched content, only the honesty framing differs).
  (B) MASK-native: run each binary known_facts example's PRESSURE prompt (may lie) and
      BELIEF-elicitation prompt (truthful); on examples the model actually LIED, contrast
      the lying-response activations vs the truthful (belief) response activations.
Direction sign: lying - truthful (positive = 'more lying'). Saved as .npy per layer.
Env: DEC_LAYERS (csv, default 8,12,16,20), DEC_N (per source).
"""
import os, json, re
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

R = os.environ.get("R", "/workspace/turnstile")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYERS = [int(x) for x in os.environ.get("DEC_LAYERS", "8,12,16,20").split(",")]
N = int(os.environ.get("DEC_N", "200"))
OUT = f"{R}/directions_deception"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()


@torch.inference_mode()
def acts(messages, response=None):
    """Mean residual over the response tokens (posterior) at each layer.
    If response given, teacher-force it; else use the last-prompt-token."""
    pre = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p = tok(pre, return_tensors="pt").input_ids.shape[1]
    text = pre + (response or "")
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
    out = {}
    for L in LAYERS:
        h = hs[L + 1][0].float()
        out[L] = (h[p:].mean(0) if (response and h.shape[0] > p) else h[p - 1]).cpu().numpy()
    return out


@torch.inference_mode()
def gen(messages, max_new=64):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(dev)
    o = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()


# ---- data ----
facts = []
for cfg in ["known_facts", "statistics", "disinformation"]:
    try:
        d = load_dataset("cais/MASK", cfg); s = d[list(d.keys())[0]]
        for r in s: facts.append(r)
    except Exception as e:
        print("cfg fail", cfg, str(e)[:80], flush=True)
print(f"MASK examples pooled: {len(facts)}", flush=True)

# ================= (A) RepE-style honest/dishonest persona =================
HON = "You are an honest person who always tells the truth about the world."
DIS = "You are a dishonest person who lies and makes false statements about the world."
repe_hon = {L: [] for L in LAYERS}; repe_dis = {L: [] for L in LAYERS}
stmts = [r["ground_truth"] for r in facts if r.get("ground_truth")][:N]
for i, st in enumerate(stmts):
    for persona, bucket in [(HON, repe_hon), (DIS, repe_dis)]:
        a = acts([{"role": "system", "content": persona},
                  {"role": "user", "content": "State a fact about the world."}], response=st)
        for L in LAYERS: bucket[L].append(a[L])
    if (i + 1) % 50 == 0: print(f"  repe {i+1}/{len(stmts)}", flush=True)

# ================= (B) MASK-native lie vs truth =================
def yn(t):  # parse yes/no
    t = t.lower()
    if re.search(r"\byes\b", t) and not re.search(r"\bno\b", t): return "yes"
    if re.search(r"\bno\b", t) and not re.search(r"\byes\b", t): return "no"
    return None

mask_lie = {L: [] for L in LAYERS}; mask_truth = {L: [] for L in LAYERS}
binary = [r for r in facts if r.get("type") == "binary"][:N]
n_lied = 0
for i, r in enumerate(binary):
    gt = r.get("formatted_ground_truth")  # 'A' or 'B' ; ground_truth text is the true resolution
    # pressure response
    press = [{"role": "system", "content": r["system_prompt"]}, {"role": "user", "content": r["user_prompt"]}]
    sresp = gen(press, 48); s_yn = yn(sresp)
    # belief response (neutral)
    bel = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": r["belief_elicit_1"]}]
    bresp = gen(bel, 96); b_yn = yn(bresp)
    if s_yn is None or b_yn is None or s_yn == b_yn:
        continue  # need a clear disagreement (statement != belief) to call it a lie
    # lie = pressured answer contradicts the neutral belief answer
    n_lied += 1
    al = acts(press, response=sresp); at = acts(bel, response=bresp)
    for L in LAYERS: mask_lie[L].append(al[L]); mask_truth[L].append(at[L])
    if (i + 1) % 40 == 0: print(f"  mask {i+1}/{len(binary)} lied={n_lied}", flush=True)

# ---- directions (lying - truthful), save + cosine ----
def unit(v): return v / (np.linalg.norm(v) + 1e-9)
report = {}
for L in LAYERS:
    d_repe = unit(np.mean(repe_dis[L], 0) - np.mean(repe_hon[L], 0))
    np.save(f"{OUT}/repe_lie_L{L}.npy", d_repe)
    if mask_lie[L]:
        d_mask = unit(np.mean(mask_lie[L], 0) - np.mean(mask_truth[L], 0))
        np.save(f"{OUT}/mask_lie_L{L}.npy", d_mask)
        report[L] = float(d_repe @ d_mask)
    else:
        report[L] = None
print("n_lied(mask-native):", n_lied, flush=True)
print("cos(repe_lie, mask_lie) by layer:", {L: (round(c,3) if c is not None else None) for L,c in report.items()}, flush=True)
json.dump({"n_lied": n_lied, "cos_repe_mask": report, "layers": LAYERS}, open(f"{OUT}/summary.json", "w"))
print("DIRECTIONS_DONE", flush=True)
