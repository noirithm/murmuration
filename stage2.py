"""Stage 2: choose exactly K of N candidates under budget and capacity limits.

    maximise  sum_i v_i x_i,  v_i = p_i * D_i - c_i
    s.t.      sum x_i = K,   sum c_i x_i <= C,   sum u_i x_i <= H,   x_i in {0,1}

Inequalities become equalities with binary-encoded slack variables (exact range [0, ub]).
Each equality is squared and added with penalty P = 1 + sum|v_i|, which exceeds any objective
gain from violating an integer-valued constraint. With FORCE_SLACK the slack encoding is used for
every inequality even when it cannot bind for any K-subset (the code reports which are
non-binding). The QUBO is minimised by QAOA (own Qiskit circuit, Aer, CVaR objective)."""
import itertools
import logging

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

import config as cfg
from qtools import circuit_stats

log = logging.getLogger("stage2")


def slack_weights(ub):
    if ub <= 0:
        return []
    m = int(ub).bit_length()
    return [2 ** j for j in range(m - 1)] + [int(ub) - (2 ** (m - 1) - 1)]


def build_qubo(v, cost, units, K, C, H, force_slack=False, penalty=None):
    n = len(v)
    P = float(penalty) if penalty is not None else 1.0 + float(np.abs(v).sum())
    blocks, non_binding = [("cardinality", np.ones(n), K, [])], []
    for name, a, cap in (("budget", cost, C), ("capacity", units, H)):
        a_sorted = np.sort(a)
        if a_sorted[-K:].sum() <= cap:
            non_binding.append(name)            # cannot bind for any K-subset
            if not force_slack:
                continue
        ub = int(cap - a_sorted[:K].sum())
        if ub < 0:
            raise ValueError(f"{name} constraint is infeasible")
        blocks.append((name, a, cap, slack_weights(ub)))
    m = n + sum(len(b[3]) for b in blocks)
    Q, const, pos, slack_info = np.zeros((m, m)), 0.0, n, {}
    Q[np.arange(n), np.arange(n)] += -np.asarray(v, dtype=float)
    for name, a, rhs, w in blocks:
        coef = np.zeros(m)
        coef[:n] = a
        coef[pos:pos + len(w)] = w
        slack_info[name] = {"weights": [int(x) for x in w], "qubits": list(range(pos, pos + len(w)))}
        pos += len(w)
        Q += P * np.outer(coef, coef)
        Q[np.diag_indices(m)] += -2 * rhs * P * coef
        const += P * rhs ** 2
    return {"Q": Q, "const": const, "n": n, "m": m, "penalty": P,
            "slack": slack_info, "redundant_constraints": non_binding}


def energies(Q, const):
    m = len(Q)
    bits = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(float)   # qubit k = bit k
    return (bits @ Q * bits).sum(1) + const, bits


def to_ising(Q, const):
    q = np.diag(Q).copy()
    Qp = 2 * np.triu(Q, 1)
    h = -q / 2 - (Qp.sum(1) + Qp.sum(0)) / 4
    m = len(Q)
    J = {(k, l): float(Qp[k, l] / 4) for k in range(m) for l in range(k + 1, m)
         if abs(Qp[k, l]) > 1e-12}
    return h, J, const + q.sum() / 2 + Qp.sum() / 4


def ising_energies(h, J, c0, bits):
    Z = 1 - 2 * bits
    E = c0 + Z @ h
    for (k, l), j in J.items():
        E = E + j * Z[:, k] * Z[:, l]
    return E


def qaoa_circuit(h, J, p):
    m = len(h)
    gam = [Parameter(f"gamma_{i}") for i in range(p)]
    bet = [Parameter(f"beta_{i}") for i in range(p)]
    qc = QuantumCircuit(m)
    qc.h(range(m))
    for layer in range(p):
        for (k, l), j in J.items():
            qc.rzz(2 * gam[layer] * float(j), k, l)
        for k in range(m):
            qc.rz(2 * gam[layer] * float(h[k]), k)
        for k in range(m):
            qc.rx(2 * bet[layer], k)
    return qc, gam + bet


def cvar(probs, E, alpha, order=None):
    order = np.argsort(E) if order is None else order
    pm, Eo = probs[order], E[order]
    before = np.cumsum(pm) - pm
    take = np.minimum(pm, np.clip(alpha - before, 0, None))
    return float((take * Eo).sum() / alpha)


def evaluate(x, v, cost, units, K, C, H):
    sel = np.flatnonzero(x)
    feasible = len(sel) == K and cost[sel].sum() <= C and units[sel].sum() <= H
    return bool(feasible), float(v[sel].sum())


def exact_optimum(v, cost, units, K, C, H):
    best = None
    for comb in itertools.combinations(range(len(v)), K):
        c = list(comb)
        if cost[c].sum() <= C and units[c].sum() <= H:
            val = v[c].sum()
            if best is None or val > best[0] + 1e-12:
                best = (float(val), comb)
    return best


def greedy(v, cost, units, K, C, H):
    sel = []
    for i in np.argsort(-v, kind="stable"):
        t = sel + [int(i)]
        if cost[t].sum() <= C and units[t].sum() <= H:
            sel = t
        if len(sel) == K:
            break
    return sorted(sel)


def run(stage1, stage2_data, limits, backend=None, seed=cfg.SEED, force_slack=None):
    force_slack = cfg.FORCE_SLACK if force_slack is None else force_slack
    N, K, C, H = limits["N"], limits["K"], limits["C"], limits["H"]
    cand = stage1[stage1["top10_flag"] == 1].sort_values("rank")
    assert len(cand) == N
    tbl = stage2_data.loc[cand.index, ["contact_cost", "expected_deposit_value",
                                       "capacity_units", "customer_segment"]].copy()
    tbl.insert(0, "predicted_probability", cand["predicted_probability"])
    tbl["expected_net_value"] = (tbl["predicted_probability"] * tbl["expected_deposit_value"]
                                 - tbl["contact_cost"])
    ids = list(tbl.index)
    v = tbl["expected_net_value"].to_numpy(float)
    cost = tbl["contact_cost"].to_numpy(int)
    units = tbl["capacity_units"].to_numpy(int)

    exact_val, exact_comb = exact_optimum(v, cost, units, K, C, H)
    greedy_sel = greedy(v, cost, units, K, C, H)

    # penalty design: smallest tested weight whose QUBO minimum equals the brute-force optimum
    vmax = float(np.abs(v).max())
    tried = []
    for P in (1 + vmax, 2 * (1 + vmax), 1 + float(np.abs(v).sum())):
        qubo = build_qubo(v, cost, units, K, C, H, force_slack=force_slack, penalty=P)
        Q, const, n, m = qubo["Q"], qubo["const"], qubo["n"], qubo["m"]
        E, bits = energies(Q, const)
        ok = tuple(int(i) for i in np.flatnonzero(bits[int(E.argmin()), :n])) == tuple(exact_comb)
        tried.append({"penalty": P, "qubo_minimum_matches_exact": bool(ok)})
        if ok:
            break
    else:
        raise RuntimeError("no tested penalty preserves the exact optimum")
    h, J, c0 = to_ising(Q, const)
    assert np.allclose(E, ising_energies(h, J, c0, bits)), "Ising/QUBO mismatch"
    qubo_min_x = tuple(int(i) for i in np.flatnonzero(bits[int(E.argmin()), :n]))
    qubo_matches_exact = qubo_min_x == tuple(exact_comb)
    log.info("qubits=%d penalty=%.1f non_binding=%s slack=%s qubo_min_matches_exact=%s",
             m, qubo["penalty"], qubo["redundant_constraints"], qubo["slack"], qubo_matches_exact)

    scale = max(float(np.abs(h).max()), max(abs(j) for j in J.values()))
    qc, params = qaoa_circuit(h / scale, {k: j / scale for k, j in J.items()}, cfg.QAOA_REPS)
    sim = AerSimulator(method="statevector", seed_simulator=seed)
    warned = [False]

    def probs(theta):
        bound = qc.assign_parameters(dict(zip(params, theta)))
        try:
            b = bound.copy()
            b.save_statevector()
            sv = sim.run(b).result().get_statevector().data
        except Exception as exc:
            if not warned[0]:
                log.warning("Aer statevector failed (%s); using slower quantum_info.Statevector", exc)
                warned[0] = True
            sv = Statevector(bound).data
        return np.abs(sv) ** 2

    order = np.argsort(E)
    rng = np.random.default_rng(seed)
    p_ = cfg.QAOA_REPS
    best = None
    for r in range(cfg.QAOA_RESTARTS):
        if r == 0:
            theta0 = np.concatenate([np.linspace(0.2, 0.8, p_), np.linspace(0.6, 0.2, p_)])
        else:
            theta0 = np.concatenate([rng.uniform(0, 1.0, p_), rng.uniform(0, 0.8, p_)])
        res = minimize(lambda t: cvar(probs(t), E, cfg.CVAR_ALPHA, order), theta0, method="COBYLA",
                       options={"maxiter": cfg.QAOA_MAXITER, "rhobeg": 0.3})
        log.info("restart %d: CVaR=%.3f", r, res.fun)
        if best is None or res.fun < best.fun:
            best = res
    theta = best.x

    pr = probs(theta)
    x_all = bits[:, :n].astype(int)
    feas_mask = (x_all.sum(1) == K) & (x_all @ cost <= C) & (x_all @ units <= H)
    opt_vec = np.isin(np.arange(n), exact_comb).astype(int)
    opt_mask = (x_all == opt_vec).all(1)

    bound = qc.assign_parameters(dict(zip(params, theta)))
    meas = bound.copy()
    meas.measure_all()
    counts = AerSimulator(seed_simulator=seed).run(meas, shots=cfg.STAGE2_SHOTS,
                                                    seed_simulator=seed).result().get_counts()
    rows = []
    for key, c in counts.items():
        z = np.array([int(b) for b in key[::-1]])    # qiskit keys are little-endian: reverse
        feas, val = evaluate(z[:n], v, cost, units, K, C, H)
        rows.append({"bitstring_qubit_m-1_to_0": key, "x_bits_customer0_to_N-1": "".join(map(str, z[:n])),
                     "count": c, "energy": float(E[int(key, 2)]), "feasible": feas, "objective": val})
    samples = pd.DataFrame(rows).sort_values("count", ascending=False)
    feas_samples = samples[samples["feasible"]]
    quantum_ok = len(feas_samples) > 0
    if quantum_ok:
        pick = feas_samples.sort_values(["objective", "count"], ascending=False).iloc[0]
        sel_idx = [i for i, b in enumerate(pick["x_bits_customer0_to_N-1"]) if b == "1"]
        best_bits = pick["bitstring_qubit_m-1_to_0"]
    else:
        sel_idx, best_bits = list(exact_comb), None   # fallback is classical: README must say so
    feas, val = evaluate(np.isin(np.arange(n), sel_idx).astype(int), v, cost, units, K, C, H)

    # diagnostics: how much better than uniform random sampling, and the strict decode
    xs = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    feas_x = (xs.sum(1) == K) & (xs @ cost <= C) & (xs @ units <= H)
    strict = samples.sort_values(["energy", "count"], ascending=[True, False]).iloc[0]
    strict_ok, strict_val = evaluate(np.array([int(b) for b in strict["x_bits_customer0_to_N-1"]]),
                                     v, cost, units, K, C, H)
    opt_str = "".join(str(b) for b in opt_vec)
    p_opt, p_feas = float(pr[opt_mask].sum()), float(pr[feas_mask].sum())

    result = {
        "candidates": ids, "selected_ids": sorted(ids[i] for i in sel_idx),
        "feasible": feas, "objective": val, "budget_total": int(cost[sel_idx].sum()),
        "budget_limit": C, "capacity_total": int(units[sel_idx].sum()), "capacity_limit": H,
        "best_bitstring": best_bits, "quantum_feasible_sample_found": quantum_ok,
        "exact_optimum_ids": sorted(ids[i] for i in exact_comb), "exact_optimum_value": exact_val,
        "gap_abs": exact_val - val, "gap_pct": 100 * (exact_val - val) / abs(exact_val),
        "greedy_ids": sorted(ids[i] for i in greedy_sel), "greedy_value": float(v[greedy_sel].sum()),
        "qubo": {"qubits": m, "decision_qubits": n, "penalty": qubo["penalty"],
                 "slack": qubo["slack"], "non_binding_constraints": qubo["redundant_constraints"],
                 "force_slack": force_slack, "penalty_search": tried, "qubo_minimum_matches_exact": bool(qubo_matches_exact)},
        "qaoa": {"reps": cfg.QAOA_REPS, "optimizer": "COBYLA", "objective": f"CVaR_{cfg.CVAR_ALPHA}",
                 "restarts": cfg.QAOA_RESTARTS, "hamiltonian_scale": scale,
                 "optimized_parameters": dict(zip([p.name for p in params], map(float, theta))),
                 "final_cvar": float(best.fun), "shots": cfg.STAGE2_SHOTS,
                 "simulator": "qiskit_aer AerSimulator (statevector, noiseless)",
                 "decode_rule": "best feasible measured x by objective (true constraints checked on x)",
                 "ideal_prob_optimum_x": p_opt, "ideal_prob_feasible_x": p_feas,
                 "uniform_prob_optimum_x": 1 / 2 ** n, "uniform_prob_feasible_x": float(feas_x.mean()),
                 "enrichment_optimum_vs_uniform": p_opt * 2 ** n,
                 "enrichment_feasible_vs_uniform": p_feas / float(feas_x.mean()),
                 "optimum_hits_in_shots": int(samples[samples["x_bits_customer0_to_N-1"] == opt_str]["count"].sum()),
                 "optimum_hits_expected_uniform": cfg.STAGE2_SHOTS / 2 ** n,
                 "shots_feasible_fraction": float(feas_samples["count"].sum() / cfg.STAGE2_SHOTS),
                 "strict_min_energy_sample": {
                     "bitstring": strict["bitstring_qubit_m-1_to_0"], "feasible": strict_ok,
                     "objective": strict_val,
                     "is_exact_optimum": bool(strict_ok and abs(strict_val - exact_val) < 1e-9)}},
        "circuit": circuit_stats(bound, backend, seed) if backend is not None else None,
    }
    ctx = {"circuit": bound, "n": n, "v": v, "cost": cost, "units": units,
           "K": K, "C": C, "H": H, "exact_comb": exact_comb, "exact_val": exact_val}
    return result, tbl, samples, ctx


def noise_sweep(ctx, scales, shots, seed):
    """Chi(r)ps angle: sample the SAME optimised circuit under growing gate noise and see what the
    noisy chip proposes: feasible mass, distinct feasible portfolios, near-optimal (>=95%) ones."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    n, v, cost, units = ctx["n"], ctx["v"], ctx["cost"], ctx["units"]
    K, C, H = ctx["K"], ctx["C"], ctx["H"]
    opt = tuple(int(i) for i in ctx["exact_comb"])
    rows = []
    for s in scales:
        nm = None
        if s > 0:
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(min(1e-3 * s, 0.5), 1), ["rx", "rz", "h"])
            nm.add_all_qubit_quantum_error(depolarizing_error(min(1e-2 * s, 0.9), 2), ["rzz"])
        meas = ctx["circuit"].copy()
        meas.measure_all()
        counts = AerSimulator(noise_model=nm, seed_simulator=seed).run(
            meas, shots=shots, seed_simulator=seed).result().get_counts()
        feas_ct, opt_ct, distinct, near, best = 0, 0, set(), set(), -np.inf
        for key, c in counts.items():
            z = np.array([int(b) for b in key[::-1]])[:n]
            ok, val = evaluate(z, v, cost, units, K, C, H)
            if ok:
                sel = tuple(int(i) for i in np.flatnonzero(z))
                feas_ct += c
                distinct.add(sel)
                best = max(best, val)
                if val >= 0.95 * ctx["exact_val"]:
                    near.add(sel)
                if sel == opt:
                    opt_ct += c
        rows.append({"noise_scale": s, "shots": shots, "feasible_fraction": feas_ct / shots,
                     "optimum_fraction": opt_ct / shots, "distinct_feasible_portfolios": len(distinct),
                     "distinct_near_optimal_portfolios": len(near),
                     "best_feasible_objective": None if best == -np.inf else best,
                     "exact_optimum": ctx["exact_val"]})
        log.info("stage2 noise scale %.1f done", s)
    return pd.DataFrame(rows)
