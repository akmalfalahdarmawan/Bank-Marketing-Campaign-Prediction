import os
import re
import numpy as np
import pandas as pd
import cloudpickle
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing Campaign", layout="wide")

# ==== Matplotlib theme (single render, no duplicates) ====
plt.rcdefaults()  # reset style global

RC_LIGHT = dict(
    fig="#ffffff", ax="#ffffff", save="#ffffff",
    text="#111827", lbl="#111827", tick="#111827",
    edge="#e5e7eb", grid="#e5e7eb"
)
RC_DARK = dict(
    fig="#0e1117", ax="#0e1117", save="#0e1117",
    text="#e5e7eb", lbl="#e5e7eb", tick="#e5e7eb",
    edge="#2a2f3b", grid="#2a2f3b"
)

def resolve_chart_mode(choice: str) -> str:
    if choice == "Light": return "light"
    if choice == "Dark":  return "dark"
    try:
        base = st.get_option("theme.base")
        return "dark" if str(base).lower() == "dark" else "light"
    except Exception:
        return "light"

def apply_chart_theme(fig, mode: str):
    C = RC_DARK if mode == "dark" else RC_LIGHT
    fig.patch.set_facecolor(C["fig"])
    for ax in fig.get_axes():
        ax.set_facecolor(C["ax"])
        ax.tick_params(colors=C["tick"])
        ax.xaxis.label.set_color(C["lbl"])
        ax.yaxis.label.set_color(C["lbl"])
        for sp in ax.spines.values():
            sp.set_color(C["edge"])
        ax.grid(color=C["grid"])

# ================= MODEL LOADER =================
@st.cache_resource
def load_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        model_path = os.path.join(base_dir, "final_champion_model.pkl")
        with open(model_path, "rb") as f:
            return cloudpickle.load(f)
    except Exception as e:
        st.exception(e); st.stop()

# ================= NAME MAPPING (robust) =================
def _unwrap_estimator(m):
    if hasattr(m, "named_steps"):
        last_key = list(m.named_steps.keys())[-1]
        return m.named_steps[last_key], m
    return m, None

def _as_list_of_str(cols, input_features):
    if cols is None: return []
    if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
        if len(cols) == 0: return []
        if isinstance(cols[0], (int, np.integer)):
            return [input_features[i] for i in cols]
        return [str(c) for c in cols]
    if isinstance(cols, slice):
        idx = list(range(*cols.indices(len(input_features))))
        return [input_features[i] for i in idx]
    if isinstance(cols, (int, np.integer)):
        return [input_features[int(cols)]]
    return [str(cols)]

def _last_step(est):
    return est.steps[-1][1] if hasattr(est, "steps") else est

def _names_from_one_hot(enc, cols):
    names = []
    try:
        for col_name, cat_list in zip(cols, enc.categories_):
            for cat in cat_list:
                names.append(f"{col_name}={cat}")
        return names
    except Exception:
        return [f"{c} (onehot)" for c in cols]

def _names_from_transformer(trans, cols, input_features):
    cols = _as_list_of_str(cols, input_features)
    t = _last_step(trans)
    if hasattr(t, "get_feature_names_out"):
        try:
            return list(t.get_feature_names_out(cols))
        except Exception:
            pass
    try:
        from sklearn.preprocessing import OneHotEncoder
        if isinstance(t, OneHotEncoder):
            return _names_from_one_hot(t, cols)
    except Exception:
        pass
    return cols

def get_feature_names_hard(model):
    est, pipe = _unwrap_estimator(model)
    input_features = None
    for obj in (model, est):
        if hasattr(obj, "feature_names_in_"):
            input_features = list(obj.feature_names_in_)
            break
    if input_features is None:
        n = getattr(est, "n_features_in_", 0) or 0
        return [f"x{j}" for j in range(n)]

    from sklearn.compose import ColumnTransformer
    ct = None
    if pipe is not None:
        for _, step in pipe.named_steps.items():
            if hasattr(step, "transformers_") and hasattr(step, "transform"):
                ct = step; break

    if ct is None or not hasattr(ct, "transformers_"):
        if hasattr(est, "feature_names_in_"):
            return list(est.feature_names_in_)
        n = getattr(est, "n_features_in_", 0) or 0
        return [f"x{j}" for j in range(n)]

    used = set(); out_names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder": continue
        col_names = _as_list_of_str(cols, input_features)
        used.update(col_names)
        if trans == "drop": continue
        if trans == "passthrough":
            out_names.extend(col_names); continue
        out_names.extend(_names_from_transformer(trans, col_names, input_features))

    if getattr(ct, "remainder", "drop") == "passthrough":
        rem_cols = [c for c in input_features if c not in used]
        out_names.extend(rem_cols)
    return out_names

def prettify_name(s: str) -> str:
    s = s.replace("ever_contacted", "Ever contacted")
    s = s.replace("pdays", "Days since last contact")
    s = s.replace("balance", "Account balance (€)")
    s = s.replace("campaign", "Contacts in current campaign")
    s = s.replace("housing", "Has housing loan")
    s = s.replace("loan", "Has personal loan")
    s = re.sub(r"^month[_=](\w+)$", r"Month=\1", s)
    s = re.sub(r"^job[_=](.+)$", r"Job=\1", s)
    s = re.sub(r"^contact[_=](.+)$", r"Contact=\1", s)
    s = re.sub(r"^housing[_=](.+)$", r"Housing=\1", s)
    s = re.sub(r"^loan[_=](.+)$", r"Loan=\1", s)
    s = re.sub(r"^poutcome[_=](.+)$", r"Previous outcome=\1", s)
    return s

def get_feature_importance(model):
    est, _ = _unwrap_estimator(model)
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, float)
        s = vals.sum();  return vals / s if s != 0 else vals
    if hasattr(est, "coef_"):
        coefs = est.coef_
        coefs = np.mean(np.abs(coefs), axis=0) if coefs.ndim > 1 else np.abs(coefs)
        s = coefs.sum(); return coefs / s if s != 0 else coefs
    return None

# ================= Bullet chart =================
def draw_bullet(prob: float, thr: float = 0.35):
    fig, ax = plt.subplots(figsize=(5.6, 1.1))
    ax.barh([0], [1.0], height=0.34, alpha=0.12)
    ax.barh([0], [prob], height=0.34)
    ax.axvline(thr, linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("Probability")
    ax.grid(axis="x", alpha=.25)
    px = float(np.clip(prob, 0.06, 0.94))
    ax.text(px, -0.10, f"{prob:.1%}", ha="center", va="top", fontsize=10)
    fig.tight_layout()
    return fig

# ================= PDP helpers =================
NUMERIC_FEATURES = {
    "age": (18, 95, 20),
    "balance": (-5000, 20000, 20),  # izinkan negatif wajar (overdraft)
    "campaign": (1, 50, 25),
    "pdays": (-1, 180, 15),
}
CATEGORICAL_FEATURES = {
    "job": ["admin.","blue-collar","entrepreneur","housemaid","management",
            "retired","self-employed","services","student","technician","unemployed","unknown"],
    "housing": ["yes","no"],
    "loan": ["yes","no"],
    "contact": ["cellular","telephone","unknown"],
    "month": ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    "poutcome": ["unknown","failure","other","success"],
}

def build_row_from_sidebar(vals: dict) -> pd.DataFrame:
    row = pd.DataFrame([{
        "age": vals["age"], "job": vals["job"], "balance": vals["balance"],
        "housing": vals["housing"], "loan": vals["loan"], "contact": vals["contact"],
        "month": vals["month"], "campaign": vals["campaign"], "pdays": vals["pdays"],
        "poutcome": vals["poutcome"]
    }])
    row["ever_contacted"] = (row["pdays"] != -1).astype(int)
    return row

def prepare_for_model(row: pd.DataFrame, model) -> pd.DataFrame:
    out = row.copy()
    # safety: konsistensi pdays ↔ poutcome
    out.loc[out["pdays"] == -1, "poutcome"] = "unknown"
    out["pdays"] = out["pdays"].replace(-1, 999)
    for c in ["job","housing","loan","contact","month","poutcome"]:
        out[c] = out[c].astype(str).str.strip().str.lower()
    try:
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            missing = [c for c in expected if c not in out.columns]
            for c in missing:
                if any(k in c for k in ["job","housing","loan","contact","month","poutcome","education","default"]):
                    out[c] = "unknown"
                else:
                    out[c] = 0
            out = out.reindex(columns=expected)
    except Exception:
        pass
    return out

def predict_proba_single(model, row_for_model: pd.DataFrame) -> float:
    return float(model.predict_proba(row_for_model)[:, 1][0])

def compute_pdp(model, base_row: pd.DataFrame, feat: str):
    x_vals, y_probs = [], []
    if feat in NUMERIC_FEATURES:
        lo, hi, npoints = NUMERIC_FEATURES[feat]
        xs = [-1] + list(np.linspace(0, hi, npoints-1, dtype=int)) if feat == "pdays" \
             else list(np.linspace(lo, hi, npoints))
        for v in xs:
            tmp = base_row.copy()
            tmp.loc[:, feat] = v
            tmp["ever_contacted"] = (tmp["pdays"] != -1).astype(int)
            y = predict_proba_single(model, prepare_for_model(tmp, model))
            x_vals.append(v); y_probs.append(y)
    elif feat in CATEGORICAL_FEATURES:
        for v in CATEGORICAL_FEATURES[feat]:
            tmp = base_row.copy()
            tmp.loc[:, feat] = v
            tmp["ever_contacted"] = (tmp["pdays"] != -1).astype(int)
            y = predict_proba_single(model, prepare_for_model(tmp, model))
            x_vals.append(v); y_probs.append(y)
    return x_vals, y_probs

# ================= APP =================
def main():
    st.markdown("<h1 style='font-weight:800;letter-spacing:.2px;'>🤖 Bank Marketing Campaign Prediction Using Machine Learning</h1>", unsafe_allow_html=True)
    st.caption("Tip: change the chart theme (Light/Dark/Auto) in the sidebar.")

    with st.spinner("Loading model..."):
        model = load_model()

    # Sidebar inputs (dengan validasi agar konsisten)
    st.sidebar.header("Enter Client Data")
    chart_theme_choice = st.sidebar.selectbox("Chart theme (figures only)", ["Auto", "Light", "Dark"], index=0)

    age = st.sidebar.slider("Age", 18, 95, 40)
    job = st.sidebar.selectbox("Job",
        ["admin.","blue-collar","entrepreneur","housemaid","management",
         "retired","self-employed","services","student","technician","unemployed","unknown"])

    balance = st.sidebar.number_input(
        "Current Balance (€)", value=1500, min_value=-5000, max_value=100000, step=50,
        help="Negatif berarti overdraft. Range diset agar masuk akal."
    )

    housing = st.sidebar.selectbox("Has Housing Loan?", ["yes","no"])
    loan = st.sidebar.selectbox("Has Personal Loan?", ["yes","no"])
    contact = st.sidebar.selectbox("Contact Type", ["cellular","telephone","unknown"])
    month = st.sidebar.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    campaign = st.sidebar.slider("Contacts in Current Campaign", 1, 50, 1)

    pdays = st.sidebar.number_input(
        "Days Since Last Contact (-1 if never)", value=-1, min_value=-1, max_value=180, step=1,
        help="Isi -1 kalau belum pernah dihubungi; kalau pernah, isi 0–180."
    )
    # CONDITIONAL: poutcome tergantung pdays
    if pdays == -1:
        poutcome = st.sidebar.selectbox(
            "Previous Outcome", ["unknown"], index=0,
            help="Karena belum pernah dihubungi (pdays=-1), outcome harus 'unknown'."
        )
    else:
        poutcome = st.sidebar.selectbox(
            "Previous Outcome", ["failure","other","success"], index=0,
            help="Karena pernah dihubungi (pdays≥0), pilih hasil kampanye sebelumnya."
        )

    chart_mode = resolve_chart_mode(chart_theme_choice)

    # Precompute names & importances
    names_transformed = get_feature_names_hard(model)
    importances = get_feature_importance(model)

    # Predict button
    if st.button("Predict"):
        vals = dict(age=age, job=job, balance=balance, housing=housing, loan=loan,
                    contact=contact, month=month, campaign=campaign, pdays=pdays, poutcome=poutcome)
        row = build_row_from_sidebar(vals)
        row_m = prepare_for_model(row, model)
        try:
            proba = predict_proba_single(model, row_m)
            st.session_state["proba"] = proba
            st.session_state["base_row"] = row
        except Exception as e:
            st.error("Input–Model mismatch. Details:"); st.exception(e)

    # Read last prediction/input from session
    proba = st.session_state.get("proba", None)
    base_row = st.session_state.get(
        "base_row",
        build_row_from_sidebar(dict(age=age, job=job, balance=balance, housing=housing, loan=loan,
                                    contact=contact, month=month, campaign=campaign, pdays=pdays, poutcome=poutcome))
    )

    if proba is not None:
        st.metric("Subscription Probability", f"{proba:.2%}")

    # ======= Customer Input (top) =======
    st.subheader("Customer Input (details)")
    display = base_row.copy()
    display["ever_contacted"] = np.where(display["ever_contacted"] == 1, "yes", "no")
    cols_order = ["age","job","balance","housing","loan","contact","month",
                  "campaign","pdays","ever_contacted","poutcome"]
    display = display[[c for c in cols_order if c in display.columns]]
    st.dataframe(display, use_container_width=True)

    # ======= Probability + Feature Importance =======
    col_prob, col_fi = st.columns([1, 1])

    with col_prob:
        st.subheader("Probability (Bullet Chart)")
        if proba is None:
            st.info("Click Predict to see the probability.")
        else:
            thr = 0.35
            fig_b = draw_bullet(proba, thr)
            apply_chart_theme(fig_b, chart_mode)
            st.pyplot(fig_b, use_container_width=True)
            st.caption(f"Decision threshold: {thr:.2f}")

    with col_fi:
        st.subheader("Top Feature (Global)")
        if importances is None or len(importances) == 0:
            st.info("The model does not expose feature importances / coefficients.")
        else:
            n = min(len(names_transformed), len(importances))
            names = [prettify_name(nm) for nm in names_transformed[:n]]
            vals = np.asarray(importances[:n], float)
            K = min(8, n)
            idx = np.argsort(vals)[-K:][::-1]
            top_names = [names[i] for i in idx]
            top_vals  = [vals[i] for i in idx]
            fig_fi, ax_fi = plt.subplots(figsize=(5.6, 3.0))
            ax_fi.barh(range(K), top_vals[::-1])
            ax_fi.set_yticks(range(K))
            ax_fi.set_yticklabels(top_names[::-1], fontsize=9)
            ax_fi.set_xlabel("Relative Importance")
            ax_fi.grid(axis="x", alpha=.3)
            fig_fi.tight_layout()
            apply_chart_theme(fig_fi, chart_mode)
            st.pyplot(fig_fi, use_container_width=True)

    # ======= Quick insight =======
    st.subheader("Quick Insight")
    thr = 0.35
    if proba is None:
        st.write("- **No score yet.** Click Predict above.")
    else:
        decision = "YES — prioritize this lead." if proba >= thr else "NO — deprioritize; consider nurture."
        if importances is not None and len(importances) > 0:
            n2 = min(len(names_transformed), len(importances))
            names2 = [prettify_name(nm) for nm in names_transformed[:n2]]
            vals2 = np.asarray(importances[:n2], float)
            idx3 = np.argsort(vals2)[-3:][::-1]
            top3 = ", ".join(names2[i] for i in idx3)
            st.write(
                f"- **Decision** (threshold {thr:.2f}): {decision}\n"
                f"- **Top Feature overall**: {top3}.\n"
                f"- **Score**: {proba:.2%}."
            )
        else:
            st.write(f"- **Decision** (threshold {thr:.2f}): {decision}\n- **Score**: {proba:.2%}.")

    # ======= PDP (compact) =======
    st.markdown("### Partial Dependence (What-if for one feature)")
    pdp_feat = st.selectbox(
        "Choose a feature to vary:",
        ["age","balance","campaign","pdays","job","housing","loan","contact","month","poutcome"],
        index=0
    )
    x_vals, y_probs = compute_pdp(model, base_row, pdp_feat)

    if len(x_vals) > 0:
        fig_p, ax_p = plt.subplots(figsize=(6.2, 2.8))
        if pdp_feat in CATEGORICAL_FEATURES:
            ax_p.plot(range(len(x_vals)), y_probs, marker="o")
            ax_p.set_xticks(range(len(x_vals)))
            ax_p.set_xticklabels(x_vals, fontsize=9)
            ax_p.set_xlabel(pdp_feat)
        else:
            ax_p.plot(x_vals, y_probs, marker="o", linewidth=1.5)
            ax_p.set_xlabel(pdp_feat)
        ax_p.set_ylabel("Predicted probability")
        ax_p.grid(alpha=.3)
        fig_p.tight_layout()
        apply_chart_theme(fig_p, chart_mode)
        st.pyplot(fig_p, use_container_width=True)
    else:
        st.info("Unable to compute PDP for this feature.")

    st.markdown("<div style='font-size:.9rem;color:#6b7280;text-align:center;padding-top:8px;'>Developed by <b>Akmal Falah Darmawan</b></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
