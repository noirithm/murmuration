"""Stage 1: classical candidates, quantum-kernel candidates, hybrid blend.

All preprocessing, feature selection, PCA, scaling and calibration are fitted on training data
only. Model choice uses repeated stratified CV on train; the test set is never touched.
The candidate set is fixed in advance (below). Choosing among ~10 candidates on 35 positives is
optimistic, so differences between candidates are within noise and are reported as such."""
import logging

import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

import config as cfg
from data import CAT, NUM
from metrics import classification_report

log = logging.getLogger("stage1")


def _preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])


class SklearnModel:
    """Classical candidate: preprocessing + estimator, everything fitted on the training fold."""

    def __init__(self, name, factory):
        self.name, self.factory = name, factory

    def fit_predict(self, Xtr, ytr, Xte):
        model = make_pipeline(_preprocessor(), self.factory()).fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]


def classical_candidates():
    return [
        SklearnModel("classical_logreg", lambda: LogisticRegressionCV(
            Cs=[0.01, 0.03, 0.1, 0.3, 1.0], cv=5, scoring="neg_log_loss",
            max_iter=2000, random_state=cfg.SEED)),
        SklearnModel("classical_rf", lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt",
            random_state=cfg.SEED, n_jobs=-1)),
        SklearnModel("classical_hgb", lambda: HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=15,
            l2_regularization=1.0, random_state=cfg.SEED)),
    ]


class QuantumKernelModel:
    """Fidelity kernel k(x,x') = |<phi(x)|phi(x')>|^2 from a ZZ feature map (exact statevectors,
    noiseless), then an SVM with Platt-scaled probabilities.
    encoder='pca'        : unsupervised, 5 principal components -> 5 qubits
    encoder='supervised' : the 5 preprocessed columns with the largest |coef| * std from a
                           regularised logistic regression fitted on the training fold
    `scale` sets the kernel bandwidth."""

    def __init__(self, encoder, scale, n_qubits=cfg.N_QUBITS_S1, reps=cfg.FEATURE_MAP_REPS):
        self.encoder, self.n, self.scale = encoder, n_qubits, scale
        self.name = f"quantum_{encoder}_s{scale}"
        self.fmap = zz_feature_map(n_qubits, reps=reps, entanglement="linear")

    def _fit_encoder(self, Xtr, ytr):
        pre = _preprocessor().fit(Xtr)
        Z = pre.transform(Xtr)
        if self.encoder == "pca":
            proj = PCA(self.n, random_state=cfg.SEED).fit(Z)
            reduce = proj.transform
        else:
            lr = LogisticRegression(C=0.1, max_iter=2000).fit(Z, ytr)
            cols = np.argsort(-(np.abs(lr.coef_[0]) * Z.std(0)))[: self.n]
            reduce = lambda A: A[:, cols]
        mm = MinMaxScaler((0, np.pi)).fit(reduce(Z))
        return lambda X: np.clip(mm.transform(reduce(pre.transform(X))), 0, np.pi) * self.scale

    def _states(self, X):
        return np.array([Statevector(self.fmap.assign_parameters(list(x))).data for x in X])

    @staticmethod
    def _kernel(A, B):
        return np.abs(A.conj() @ B.T) ** 2

    def fit_predict(self, Xtr, ytr, Xte):
        enc = self._fit_encoder(Xtr, ytr)
        Str, Ste = self._states(enc(Xtr)), self._states(enc(Xte))
        svc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=cfg.SEED)
        svc.fit(self._kernel(Str, Str), ytr)
        return svc.predict_proba(self._kernel(Ste, Str))[:, 1]


def cross_validate(classical, quantum, X, y, seed=cfg.SEED):
    """Repeated stratified CV on TRAIN only. P@k, R@k use k = 10% of rows (test: top 10 of 100)."""
    models = classical + quantum
    k = max(1, int(0.10 * len(X)))
    rskf = RepeatedStratifiedKFold(n_splits=cfg.CV_FOLDS, n_repeats=cfg.CV_REPEATS, random_state=seed)
    oof = {m.name: np.zeros((cfg.CV_REPEATS, len(X))) for m in models}
    for i, (tr, va) in enumerate(rskf.split(X, y)):
        rep = i // cfg.CV_FOLDS
        for m in models:
            oof[m.name][rep, va] = m.fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[va])
        log.info("cv fold %d/%d", i + 1, cfg.CV_FOLDS * cfg.CV_REPEATS)

    c_names = [m.name for m in classical]
    oof["classical_ensemble"] = np.mean([oof[n] for n in c_names], axis=0)

    def summarise(name):
        reps = [classification_report(y.values, oof[name][r], k) for r in range(cfg.CV_REPEATS)]
        return pd.DataFrame(reps).mean()

    best_c = max(c_names + ["classical_ensemble"], key=lambda n: summarise(n)["pr_auc"])
    best_q = max([m.name for m in quantum], key=lambda n: summarise(n)["pr_auc"])
    oof["hybrid_blend"] = 0.5 * (oof[best_c] + oof[best_q])
    table = pd.DataFrame({n: summarise(n) for n in oof}).T
    table.index.name = "model"
    return table, best_c, best_q


def run(Xtr, ytr, Xte, seed=cfg.SEED):
    classical = classical_candidates()
    quantum = [QuantumKernelModel(e, s) for e in cfg.QUANTUM_ENCODERS for s in cfg.QUANTUM_SCALES]
    cv, best_c, best_q = cross_validate(classical, quantum, Xtr, ytr, seed)
    finalists = [best_c, best_q, "hybrid_blend"]
    chosen = cv.loc[finalists, "pr_auc"].idxmax()        # rule fixed in advance: best CV PR-AUC
    log.info("CV results:\n%s\nbest classical: %s | best quantum: %s | chosen: %s",
             cv.round(3).to_string(), best_c, best_q, chosen)

    preds = pd.DataFrame({m.name: m.fit_predict(Xtr, ytr, Xte) for m in classical}, index=Xte.index)
    preds["classical_ensemble"] = preds[[m.name for m in classical]].mean(axis=1)
    qmodel = next(m for m in quantum if m.name == best_q)
    preds[best_q] = qmodel.fit_predict(Xtr, ytr, Xte)
    preds["hybrid_blend"] = 0.5 * (preds[best_c] + preds[best_q])

    out = pd.DataFrame({"predicted_probability": preds[chosen].clip(0, 1)})
    order = out.reset_index().sort_values(["predicted_probability", "customer_id"],
                                          ascending=[False, True], kind="stable")
    out["rank"] = pd.Series(np.arange(1, len(order) + 1), index=order["customer_id"].values)
    out["top10_flag"] = (out["rank"] <= 10).astype(int)
    out = out.sort_values("rank")
    for c in preds.columns:
        out[f"p_{c}"] = preds[c]
    return out, cv, chosen
