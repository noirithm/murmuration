"""Circuit statistics required by the brief."""
from qiskit import transpile


def circuit_stats(qc, backend, seed):
    t = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=seed)
    ops = t.count_ops()
    two_q = t.num_nonlocal_gates()
    total = sum(v for k, v in ops.items() if k not in ("measure", "barrier"))
    used = {t.find_bit(q).index for inst in t.data for q in inst.qubits}
    try:
        sw = transpile(qc, coupling_map=backend.coupling_map,
                       basis_gates=["rz", "sx", "x", "cx", "swap"],
                       optimization_level=1, seed_transpiler=seed)
        swaps = int(sw.count_ops().get("swap", 0))
    except Exception:
        swaps = None
    return {"backend": backend.name, "logical_qubits": qc.num_qubits,
            "physical_qubits_used": len(used), "original_depth": qc.depth(),
            "original_ops": dict(qc.count_ops()), "transpiled_depth": t.depth(),
            "transpiled_ops": dict(ops), "one_qubit_gates": int(total - two_q),
            "two_qubit_gates": int(two_q), "swap_count": swaps}
