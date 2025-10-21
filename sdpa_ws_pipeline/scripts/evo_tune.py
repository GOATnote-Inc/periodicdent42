#!/usr/bin/env python3
import os, json, csv, itertools, time, argparse, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "tune"; ART.mkdir(parents=True, exist_ok=True)
BENCH = ROOT / "artifacts" / "bench"

# Search dimensions (expand/contract as needed)
SPACE = {
    "BLOCK_M":    [64, 128],
    "BLOCK_N":    [64, 128],
    "BLOCK_K":    [32, 64],
    "NUM_WARPS":  [2, 4, 8],
    "NUM_STAGES": [2, 3, 4],
    "PREFETCH":   [1, 2],
    "SMEM_LAYOUT":["row","col"],
    "EPILOGUE":   ["none","scale","scale_bias"],
    "MASK_PATH":  ["causal","none"],
    "DTYPE":      ["fp16","bf16"],
    "QKT_CHUNK":  [1, 2, 4],
    "KV_LAYOUT":  ["interleaved","separate"],
    "USE_WS":     [0, 1],     # warp specialization toggle
    "PROD_WARPS": [1, 2],     # when USE_WS=1
}

def env_for(cfg):
    e = os.environ.copy()
    # Map to your build toggles
    e["USE_CP_ASYNC"]="1"
    e["USE_WMMA_PV"]="1"
    e["USE_WARP_SPECIALIZATION"]=str(int(cfg["USE_WS"]))
    e["NUM_PRODUCER_WARPS"]=str(cfg["PROD_WARPS"])
    e["USE_PERSISTENT_CTA"]="0"
    e["USE_FAST_EXP"]="0"
    # optional: pass tile params for Triton/cuda extension build
    e["BLOCK_M"]=str(cfg["BLOCK_M"]); e["BLOCK_N"]=str(cfg["BLOCK_N"]); e["BLOCK_K"]=str(cfg["BLOCK_K"])
    e["NUM_WARPS"]=str(cfg["NUM_WARPS"]); e["NUM_STAGES"]=str(cfg["NUM_STAGES"])
    e["PREFETCH"]=str(cfg["PREFETCH"]); e["SMEM_LAYOUT"]=cfg["SMEM_LAYOUT"]; e["EPILOGUE"]=cfg["EPILOGUE"]
    e["MASK_PATH"]=cfg["MASK_PATH"]; e["DTYPE"]=cfg["DTYPE"]; e["QKT_CHUNK"]=str(cfg["QKT_CHUNK"]); e["KV_LAYOUT"]=cfg["KV_LAYOUT"]
    return e

def try_eval(cfg, shape, warmup, iters):
    env = env_for(cfg)
    # run the Stage-5 WS candidate by default when USE_WS=1, else Stage-2 control
    variants = "candidate_cuda_stub" if cfg["USE_WS"] else "candidate_triton_flashlike"
    cmd = ["python","scripts/kbench.py","--shape",",".join(map(str,shape)),"--warmup",str(warmup),"--iters",str(iters),"--variants",variants]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, cwd=ROOT)
    elapsed = time.time()-t0
    # read the just-written candidate file
    out_name = "candidate_ws_p1.json" if cfg["USE_WS"] else "candidate_stage2.json"
    out_file = BENCH / out_name
    score = None
    ok = False
    if out_file.exists():
        j = json.loads(out_file.read_text())
        ok = bool(j.get("correctness_pass", True))
        score = j["p50_us"]
    return score, ok, elapsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="2,8,512,64")
    ap.add_argument("--warmup", type=int, default=7)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--elite_k", type=int, default=6)
    args = ap.parse_args()
    shape = tuple(map(int,args.shape.split(",")))

    # EvoEngineer-Full: maintain elites + insights
    log_csv = ART / "tune_log.csv"
    with log_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["ts","config","p50_us","ok","elapsed_s"])

    # simple sampling over SPACE with elite preservation
    keys = list(SPACE.keys())
    # generate stream of configs
    def cfg_iter():
        rng = __import__("random").Random(42)
        choices = [SPACE[k] for k in keys]
        for _ in range(args.budget):
            c = {k:rng.choice(choices[i]) for i,k in enumerate(keys)}
            if c["USE_WS"]==0: c["PROD_WARPS"]=1
            yield c

    elites = []  # (score,cfg)
    for cfg in cfg_iter():
        s, ok, el = try_eval(cfg, shape, args.warmup, args.iters)
        if s is None: continue
        row = [int(time.time()), json.dumps(cfg), s, int(ok), round(el,3)]
        with log_csv.open("a", newline="") as f: csv.writer(f).writerow(row)
        if ok:
            elites.append((s,cfg))
            elites = sorted(elites, key=lambda x: x[0])[:args.elite_k]

    topk = [{"rank":i+1, "p50_us":sc, "config":cfg} for i,(sc,cfg) in enumerate(elites)]
    (ART/"topk.json").write_text(json.dumps({"shape":shape, "topk":topk}, indent=2))

if __name__=="__main__":
    main()
