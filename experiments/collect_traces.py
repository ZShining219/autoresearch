"""Collect closed-loop traces for recovery-model training.

Each episode: complete-observation SMP on one demand seed -> q_true[D,L],
phase_idx[D,J]. Meta (graphs/W/static) saved once per scenario.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tscm.data import collect_trace, world_meta
from tscm.scenarios import resolve


def save_meta(meta, path):
    import scipy.sparse as sp
    np.savez_compressed(
        path,
        lane_index=np.array(meta["lane_index"]),
        static=meta["static"], lane2tls=meta["lane2tls"],
        tls_ids=np.array(meta["tls_ids"]),
        n_phases_max=meta["n_phases_max"],
        groups=np.array([np.array(g) for g in meta["groups"]], dtype=object),
        graph_MS=meta["graph"]["MS"].numpy(),
        graph_MD=meta["graph"]["MD"].numpy(),
        graph_LS=meta["graph"]["LS"].numpy(),
        graph_LD=meta["graph"]["LD"].numpy(),
        move_keys=np.array(meta["graph"]["move_keys"], dtype=object),
        **{f"W_{k}_data": v.data for k, v in meta["Ws"].items()},
        **{f"W_{k}_idx": v.indices for k, v in meta["Ws"].items()},
        **{f"W_{k}_ptr": v.indptr for k, v in meta["Ws"].items()},
        **{f"W_{k}_shape": np.array(v.shape) for k, v in meta["Ws"].items()},
        **{f"legal_{k}": np.array(v) for k, v in meta["legal"].items()},
        allow_pickle=True)


def load_meta(path):
    import scipy.sparse as sp
    import torch
    z = np.load(path, allow_pickle=True)
    tls_ids = [str(t) for t in z["tls_ids"]]
    Ws, legal = {}, {}
    for t in tls_ids:
        Ws[t] = sp.csr_matrix((z[f"W_{t}_data"], z[f"W_{t}_idx"], z[f"W_{t}_ptr"]),
                              shape=tuple(z[f"W_{t}_shape"]))
        legal[t] = z[f"legal_{t}"].tolist()
    return {"lane_index": [str(l) for l in z["lane_index"]],
            "static": z["static"], "lane2tls": z["lane2tls"],
            "tls_ids": tls_ids, "Ws": Ws, "legal": legal,
            "n_phases_max": int(z["n_phases_max"]),
            "groups": [g.tolist() for g in z["groups"]],
            "graph": {"MS": torch.tensor(z["graph_MS"]),
                      "MD": torch.tensor(z["graph_MD"]),
                      "LS": torch.tensor(z["graph_LS"]),
                      "LD": torch.tensor(z["graph_LD"]),
                      "move_keys": z["move_keys"].tolist()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="resco4x4")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(__file__), "..", "data",
                                       "traces", args.scenario)
    os.makedirs(out_dir, exist_ok=True)
    first = args.seeds.split(",")[0]
    net, rou = resolve(args.scenario, seed=int(first))
    meta_path = os.path.join(out_dir, "meta.npz")
    if not os.path.exists(meta_path):
        meta = world_meta(net, rou)
        save_meta(meta, meta_path)
        print("meta saved:", meta_path,
              "lanes:", len(meta["lane_index"]), "moves:", meta["graph"]["n_moves"])
    for s in args.seeds.split(","):
        seed = int(s)
        net, rou = resolve(args.scenario, seed=seed)
        tr = collect_trace(net, rou, horizon=args.horizon,
                           interval=args.interval, seed=seed)
        path = os.path.join(out_dir, f"ep_seed{seed}.npz")
        np.savez_compressed(path, q_true=tr["q_true"], z_sched=tr["z_sched"],
                            phase_idx=tr["phase_idx"],
                            tls_ids=np.array(tr["tls_ids"]))
        print(f"ep_seed{seed}: D={len(tr['q_true'])} ATT={tr['metrics']['avg_travel_time']:.1f} -> {path}")


if __name__ == "__main__":
    main()
