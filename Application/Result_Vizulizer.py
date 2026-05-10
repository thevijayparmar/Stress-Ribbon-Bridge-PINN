import streamlit as st
import numpy as np
import onnxruntime as ort
import json
import math
import plotly.graph_objects as go
import os

# ==========================================
# PAGE CONFIGURATION & STATE INIT (STANDALONE)
# ==========================================
st.set_page_config(page_title="AI Bridge Solver", layout="wide")

# ==========================================
# 0. LOAD RESOURCES (DYNAMIC CACHING)
# ==========================================
@st.cache_resource
def load_models_and_dict(span_category, epoch_choice):
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    
    # Exact file mapping based on directory structure
    if span_category == "Short":
        folder = "Short_Span"
        dict_file = "ShortSpan_Translation_Dictionary_Short.json"
        if epoch_choice == "500th Epoch":
            surr_file = "ShortSpan_500Epochs_surrogate.onnx"
            pinn_file = "ShortSpan_500Epochs_pinn.onnx"
        else:
            surr_file = "ShortSpan_BEST454_surrogate.onnx"
            pinn_file = "ShortSpan_BEST454_pinn.onnx"
    else:
        folder = "Long_Span"
        dict_file = "LongSpan_Translation_Dictionary_Long.json"
        if epoch_choice == "500th Epoch":
            surr_file = "LongSpan_500Epochs_surrogate.onnx"
            pinn_file = "LongSpan_500Epochs_pinn.onnx"
        else:
            surr_file = "LongSpan_BEST491_surrogate.onnx"
            pinn_file = "LongSpan_BEST491_pinn.onnx"
            
    try:
        dict_path = os.path.join(base_path, folder, dict_file)
        surr_path = os.path.join(base_path, folder, surr_file)
        pinn_path = os.path.join(base_path, folder, pinn_file)
        
        with open(dict_path, 'r') as f:
            scale_bounds = json.load(f)
            
        session_surr = ort.InferenceSession(surr_path)
        session_pinn = ort.InferenceSession(pinn_path)
        return scale_bounds, session_surr, session_pinn
    except Exception as e:
        st.error(f"Error loading AI models for {span_category} span. Ensure the `{folder}` directory contains `{dict_file}`, `{surr_file}`, and `{pinn_file}`. Details: {e}")
        st.stop()


# ==========================================
# 1. EXACT MATH FUNCTIONS
# ==========================================
g_ratio = 0.5 * (math.sqrt(5) - 1)

def safe_a_min(L): return max(L / 700.0, 1e-3)

def x0(L, dz, a): 
    if L <= 0 or a <= 0: return L / 2
    S = math.sinh(L / (2 * a))
    if S == 0: return L / 2
    return L / 2 - a * math.asinh(dz / (2 * a * S))

def sag_pair(L, dz, a):
    xv = x0(L, dz, a)
    return a * (math.cosh(xv / a) - 1), a * (math.cosh((L - xv) / a) - 1)

def tension_pair(L, dz, a, w):
    xv = x0(L, dz, a)
    H = w * a
    return math.hypot(H, w * (0 - xv)), math.hypot(H, w * (L - xv)), max(math.hypot(H, w * (0 - xv)), math.hypot(H, w * (L - xv)))

def a_min_Tmax(L, dz, w):
    low = safe_a_min(L); high = 1e7
    while abs(high - low) > 1e-6 * (low + high) * 0.5:
        a1 = low + (1 - g_ratio) * (high - low); a2 = low + g_ratio * (high - low)
        if tension_pair(L, dz, a1, w)[2] < tension_pair(L, dz, a2, w)[2]: high = a2
        else: low = a1
    return 0.5 * (high + low)

def a_for_Tmax(L, dz, w, T_target):
    low = a_min_Tmax(L, dz, w)
    if tension_pair(L, dz, low, w)[2] >= T_target: return low
    high = low * 2
    while tension_pair(L, dz, high, w)[2] < T_target and high < 1e9: high *= 2
    while abs(high - low) > 1e-6 * (low + high) * 0.5:
        mid = 0.5 * (low + high)
        if tension_pair(L, dz, mid, w)[2] < T_target: low = mid
        else: high = mid
    return 0.5 * (low + high)

def a_for_sag(L, dz, w, target):
    if target < dz - 1e-4: return None
    high = 1e8; low = safe_a_min(L)
    if max(sag_pair(L, dz, high)) > target: return None
    while max(sag_pair(L, dz, low)) <= target:
        low *= 0.5; 
        if low < 1e-6: break
    while abs(high - low) > 1e-6 * (low + high) * 0.5:
        mid = 0.5 * (low + high)
        if max(sag_pair(L, dz, mid)) > target: low = mid
        else: high = mid
    return high

def compute_all_metrics(L, dz, w, Tallow, SagRec, a_opt, H_pred=None, mode_flag=None, is_ai=False):
    if not a_opt or np.isnan(a_opt):
        return {k: "NaN" for k in range(18)}, 0
    xv = x0(L, dz, a_opt)
    H = H_pred if is_ai else w * a_opt
    V1, V2 = w * (0 - xv), w * (L - xv)
    T1, T2 = math.hypot(H, V1), math.hypot(H, V2)
    sag1, sag2 = a_opt * (math.cosh(xv / a_opt) - 1), a_opt * (math.cosh((L - xv) / a_opt) - 1)
    abs_max_sag = dz if (xv < 0 or xv > L) else max(sag1, sag2)
    slope1 = math.degrees(math.atan(math.sinh(-xv / a_opt)))
    slope2 = math.degrees(math.atan(math.sinh((L - xv) / a_opt)))
    cable_len = a_opt * (math.sinh((L - xv) / a_opt) - math.sinh(-xv / a_opt))
    a_minT = a_min_Tmax(L, dz, w)
    xv_minT = x0(L, dz, a_minT)
    T_min = max(math.hypot(w*a_minT, w*(0-xv_minT)), math.hypot(w*a_minT, w*(L-xv_minT)))
    possible = Tallow >= (T_min - 1e-6)

    m = {}
    m["🧮 Out_Param_a_m"] = a_opt
    m["➰ Out_CableLength_m"] = cable_len
    m["🎯 Out_SagAchieved"] = "Yes" if abs(abs_max_sag - SagRec) < 0.05 else "No"
    m["🕹️ Out_ControlMode"] = "Design-Sag" if mode_flag == 1 else ("Capacity Lim" if mode_flag == 0 else "AI-Eval")
    m["📉 Out_SagStart_m"] = sag1
    m["📉 Out_SagEnd_m"] = sag2
    m["↗️ Out_SlopeStart_deg"] = slope1
    m["↘️ Out_SlopeEnd_deg"] = slope2
    m["💪 Out_TensionStart_kN"] = T1
    m["💪 Out_TensionEnd_kN"] = T2
    m["⬇️ Out_V1_kN"] = abs(V1)
    m["⬇️ Out_V2_kN"] = abs(V2)
    m["↔️ Out_H_kN"] = H
    m["⚓ Out_Tmin_kN"] = T_min
    m["🧮 Out_a_minT_m"] = a_minT
    m["📍 Out_Vertex_x0_m"] = xv
    m["⚠️ Out_Req_Tallow_kN"] = max(T1, T2)
    m["🏗️ Out_Geometry_Possible"] = "Yes" if possible else "No"
    return m, abs_max_sag

def evaluateMath(L, dz, w, Tallow, SagRec):
    a_minT = a_min_Tmax(L, dz, w)
    T_min = tension_pair(L, dz, a_minT, w)[2]
    if Tallow < (T_min - 1e-6): return {"possible": False, "metrics": {}}

    a_target = a_for_sag(L, dz, w, SagRec)
    T_req = tension_pair(L, dz, a_target, w)[2] if a_target else None

    if T_req is not None and T_req <= Tallow: a_opt, mode = a_target, 1
    else: a_opt, mode = a_for_Tmax(L, dz, w, Tallow), 0

    metrics, max_sag = compute_all_metrics(L, dz, w, Tallow, SagRec, a_opt, mode_flag=mode, is_ai=False)
    return {"possible": True, "a": a_opt, "H": w * a_opt, "sag": max_sag, "x0": metrics["📍 Out_Vertex_x0_m"], "metrics": metrics}

# ==========================================
# 2. XAI ENGINE (5 Unique Suggestions)
# ==========================================
def get_xai_insight(L, dz, w, Tallow, SagRec, baseRes):
    if not baseRes['possible']: return "⚠️ <b>Geometry Failed:</b> Tension exceeds capacity.", "#dc3545"
    baseH = baseRes['H']
    u = baseRes['metrics']['⚠️ Out_Req_Tallow_kN'] / Tallow
    rL = evaluateMath(L * 1.05, dz, w, Tallow, SagRec)
    rW = evaluateMath(L, dz, w * 1.05, Tallow, SagRec)
    rSag = evaluateMath(L, dz, w, Tallow, SagRec * 0.95)
    rDz = evaluateMath(L, dz * 1.10, w, Tallow, SagRec)
    
    insights = []
    if u > 0.95: insights.append(f"🚨 <b>Capacity Limit Imminent:</b> Utilization is at {u*100:.1f}%. Minor loads may cause failure. Upsize cables.")
    else: insights.append(f"✅ <b>Capacity Health:</b> Utilization is healthy at {u*100:.1f}%.")
    if rL['possible']: insights.append(f"📏 <b>Span Impact:</b> A 5% increase in Span (+{L*0.05:.1f}m) adds <b>+{rL['H'] - baseH:.1f} kN</b> of tension. Evaluate abutment stability.")
    else: insights.append(f"📏 <b>Span Impact:</b> Adding just 5% more Span (+{L*0.05:.1f}m) causes structural failure.")
    if rW['possible']: insights.append(f"⚖️ <b>Load Sensitive:</b> A 5% UDL increase (+{w*0.05:.1f} kN/m) adds <b>+{rW['H'] - baseH:.1f} kN</b>. Prioritize lightweight decks.")
    else: insights.append(f"⚖️ <b>Load Sensitive:</b> Adding 5% load causes immediate failure.")
    if rSag['possible']: insights.append(f"🎯 <b>Sag Efficiency:</b> Tightening sag by 5% (-{SagRec*0.05:.2f}m) sharply increases tension by <b>+{rSag['H'] - baseH:.1f} kN</b>.")
    else: insights.append(f"🎯 <b>Sag Efficiency:</b> Tightening sag by 5% forces cable tension beyond capacity.")
    if rDz['possible']: insights.append(f"📐 <b>Asymmetry Shift:</b> A 10% dz increase (+{dz*0.10:.2f}m) changes tension by <b>+{rDz['H'] - baseH:.1f} kN</b>.")
    else: insights.append("📐 <b>Asymmetry Shift:</b> Increasing height difference by 10% breaks the physical limit.")

    return "<ul style='margin:0; padding-left:25px; font-size: 0.95rem;'>" + "".join([f"<li style='margin-bottom:6px;'>{i}</li>" for i in insights]) + "</ul>", "#00bcd4"

# ==========================================
# UI LAYOUT
# ==========================================

# -----------------------------------------------------
# SIDEBAR: Independent Controls & AI Settings
# -----------------------------------------------------
st.sidebar.markdown("### 🤖 AI Model Selection")
epoch_choice = st.sidebar.radio(
    "Select Training Epoch:",
    ["500th Epoch", "Best Epoch"],
    index=0,
    help="Choose the model snapshot. 500th is the final training step, Best represents the epoch with minimum validation loss."
)
st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Design Parameters")

# Direct inputs replacing the old session_state/override logic
L = st.sidebar.slider("Span (L) [m]", 1.0, 2000.0, value=120.0, step=1.0)
dz = st.sidebar.slider("Height Diff (dz) [m]", 0.0, 50.0, value=2.0, step=0.5)
w = st.sidebar.slider("UDL (w) [kN/m]", 0.1, 200.0, value=25.0, step=0.1)
Tallow = st.sidebar.slider("Capacity (Tallow) [kN]", 10.0, 100000.0, value=50000.0, step=10.0)
SagRec = st.sidebar.slider("Target Sag [m]", 0.1, 500.0, value=6.0, step=0.1)

# Failsafes to completely prevent division by zero in math functions
L = max(L, 1.0)
w = max(w, 0.1)
Tallow = max(Tallow, 1.0)
SagRec = max(SagRec, 0.1)

# Determine Category and provide UI feedback
span_category = "Short" if L <= 300.0 else "Long"
st.sidebar.markdown(f"**Detected AI Pipeline:** `{span_category} Span`")

# -----------------------------------------------------
# MAIN SOLVER LOGIC & DYNAMIC AI PIPELINE
# -----------------------------------------------------

# Dynamically load models based on the selected span and epoch
scale_bounds, surrogateSession, pinnSession = load_models_and_dict(span_category, epoch_choice)
surr_input_name = surrogateSession.get_inputs()[0].name
pinn_input_name = pinnSession.get_inputs()[0].name

mathRes = evaluateMath(L, dz, w, Tallow, SagRec)

# AI INFERENCE (ONNX)
surrMetrics, pinnMetrics = {}, {}
if mathRes['possible']:
    in_b, out_b = scale_bounds['Inputs'], scale_bounds['Outputs']
    nL = (L - in_b['In_Span_m']['min']) / (in_b['In_Span_m']['max'] - in_b['In_Span_m']['min'])
    nDz = (dz - in_b['In_HeightDiff_m']['min']) / (in_b['In_HeightDiff_m']['max'] - in_b['In_HeightDiff_m']['min'])
    nW = (w - in_b['In_UDL_kNm']['min']) / (in_b['In_UDL_kNm']['max'] - in_b['In_UDL_kNm']['min'])
    nT = (Tallow - in_b['In_Tallow_kN']['min']) / (in_b['In_Tallow_kN']['max'] - in_b['In_Tallow_kN']['min'])
    nS = (SagRec - in_b['In_RecSag_m']['min']) / (in_b['In_RecSag_m']['max'] - in_b['In_RecSag_m']['min'])
    
    tensor_X = np.array([[nL, nDz, nW, nT, nS]], dtype=np.float32)
    out_surr = surrogateSession.run(None, {surr_input_name: tensor_X})[0][0]
    out_pinn = pinnSession.run(None, {pinn_input_name: tensor_X})[0][0]
    
    def denorm(val, key): return float(val * (out_b[key]['max'] - out_b[key]['min']) + out_b[key]['min'])
    surr_a, surr_H = denorm(out_surr[0], 'Out_Param_a_m'), denorm(out_surr[1], 'Out_H_kN')
    pinn_a, pinn_H = denorm(out_pinn[0], 'Out_Param_a_m'), denorm(out_pinn[1], 'Out_H_kN')

    surrMetrics, _ = compute_all_metrics(L, dz, w, Tallow, SagRec, surr_a, H_pred=surr_H, is_ai=True)
    pinnMetrics, _ = compute_all_metrics(L, dz, w, Tallow, SagRec, pinn_a, H_pred=pinn_H, is_ai=True)

# -----------------------------------------------------
# UI RENDERING
# -----------------------------------------------------
st.markdown("### 🌉 Bridge Viewport")

# --- COMPACTED TOP ROW TILES ---
st.markdown(f"""
<div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
    <div style="flex: 1; background: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #dee2e6; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <div style="font-size: 0.8rem; color: #6c757d; font-weight: bold; text-transform: uppercase;">Span (L)</div>
        <div style="font-size: 1.3rem; font-weight: 900; color: #007bff;">{L:.2f} m</div>
    </div>
    <div style="flex: 1; background: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #dee2e6; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <div style="font-size: 0.8rem; color: #6c757d; font-weight: bold; text-transform: uppercase;">Height Diff (dz)</div>
        <div style="font-size: 1.3rem; font-weight: 900; color: #007bff;">{dz:.2f} m</div>
    </div>
    <div style="flex: 1; background: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #dee2e6; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <div style="font-size: 0.8rem; color: #6c757d; font-weight: bold; text-transform: uppercase;">UDL (w)</div>
        <div style="font-size: 1.3rem; font-weight: 900; color: #007bff;">{w:.2f} kN/m</div>
    </div>
    <div style="flex: 1; background: #fffcfc; padding: 12px; border-radius: 8px; border: 1px solid #f5c6cb; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <div style="font-size: 0.8rem; color: #dc3545; font-weight: bold; text-transform: uppercase;">Capacity (Tallow)</div>
        <div style="font-size: 1.3rem; font-weight: 900; color: #dc3545;">{Tallow:.2f} kN</div>
    </div>
    <div style="flex: 1; background: #f9fdfa; padding: 12px; border-radius: 8px; border: 1px solid #c3e6cb; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <div style="font-size: 0.8rem; color: #28a745; font-weight: bold; text-transform: uppercase;">Target Sag</div>
        <div style="font-size: 1.3rem; font-weight: 900; color: #28a745;">{SagRec:.2f} m</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- PLOTLY VIEWPORT ---
fig = go.Figure()
if mathRes['possible']:
    a, xv = mathRes['a'], mathRes['x0']
    x_vals = np.linspace(0, L, 300)
    c0 = math.cosh(xv / a)
    y_vals = [(a * math.cosh((x - xv) / a)) - (a * c0) for x in x_vals] 
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='#007bff', width=5)))
    fig.add_trace(go.Scatter(x=[0, L], y=[0, -dz], mode='markers', marker=dict(color='#333', size=12)))
    fig.add_trace(go.Scatter(x=[0, L], y=[0, 0], mode='lines', line=dict(color='#ccc', dash='dash')))
else:
    st.error("Failed Geometry! The required tension strictly exceeds your allowable capacity.")

fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='white', showlegend=False)
st.plotly_chart(fig, use_container_width=True)


# EXPLAINABLE AI (XAI)
st.markdown("---")
xai_text, xai_color = get_xai_insight(L, dz, w, Tallow, SagRec, mathRes)
st.markdown(f"""<div style="background: #f0f8ff; border-left: 6px solid {xai_color}; padding: 15px 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); color: #111; margin-bottom: 25px;"><h4 style="margin-top:0; color:#00838f;">🧠 Explainable AI (XAI) Structural Sensitivity</h4>{xai_text}</div>""", unsafe_allow_html=True)


# 18 COMPARATIVE MEGA TILES
st.markdown("---")
st.markdown("### 📊 Comparative Solved Metrics (All 18 Parameters)")

def calc_diff_acc(e, p):
    if e == "NaN" or p == "NaN" or isinstance(e, str): return str(p), "", "", "transparent", "transparent"
    try:
        diff = float(p) - float(e); sign = "+" if diff > 0 else ""
        acc = max(0, 100 - (abs(diff)/abs(float(e))*100)) if float(e) != 0 else 0
        bg = "#d4edda" if acc > 95 else ("#fff3cd" if acc > 80 else "#f8d7da")
        tc = "#155724" if acc > 95 else ("#856404" if acc > 80 else "#721c24")
        return f"{float(p):.2f}", f"{sign}{diff:.2f}", f"Acc: {acc:.1f}%", bg, tc
    except: return str(p), "", "", "transparent", "transparent"

parameters = [
    ("🧮 Out_Param_a_m", "m", True), ("➰ Out_CableLength_m", "m", True),
    ("🎯 Out_SagAchieved", "", False), ("🕹️ Out_ControlMode", "", False),
    ("📉 Out_SagStart_m", "m", True), ("📉 Out_SagEnd_m", "m", True),
    ("↗️ Out_SlopeStart_deg", "°", True), ("↘️ Out_SlopeEnd_deg", "°", True),
    ("💪 Out_TensionStart_kN", "kN", True), ("💪 Out_TensionEnd_kN", "kN", True),
    ("⬇️ Out_V1_kN", "kN", True), ("⬇️ Out_V2_kN", "kN", True),
    ("↔️ Out_H_kN", "kN", True), ("⚓ Out_Tmin_kN", "kN", True),
    ("🧮 Out_a_minT_m", "m", True), ("📍 Out_Vertex_x0_m", "m", True),
    ("⚠️ Out_Req_Tallow_kN", "kN", True), ("🏗️ Out_Geometry_Possible", "", False)
]

tooltips = {
    "🧮 Out_Param_a_m": "The catenary parameter 'a', representing the horizontal tension divided by the uniform weight.",
    "➰ Out_CableLength_m": "The total unstressed 3D spatial length of the main bearing cable.",
    "🎯 Out_SagAchieved": "Indicates whether the target sag was physically reachable without violating tension limits.",
    "🕹️ Out_ControlMode": "The governing limit for the solver: either hitting the target 'Sag' or capped by allowable 'Tension'.",
    "📉 Out_SagStart_m": "The vertical dip measured from the starting abutment down to the lowest vertex.",
    "📉 Out_SagEnd_m": "The vertical dip measured from the ending abutment down to the lowest vertex.",
    "↗️ Out_SlopeStart_deg": "The angle of inclination of the cable at the starting abutment.",
    "↘️ Out_SlopeEnd_deg": "The angle of inclination of the cable at the ending abutment.",
    "💪 Out_TensionStart_kN": "The total axial force exerted by the cable on the starting abutment.",
    "💪 Out_TensionEnd_kN": "The total axial force exerted by the cable on the ending abutment.",
    "⬇️ Out_V1_kN": "The vertical downward force reaction at the starting abutment.",
    "⬇️ Out_V2_kN": "The vertical downward force reaction at the ending abutment.",
    "↔️ Out_H_kN": "The constant horizontal tension component extending throughout the entire cable.",
    "⚓ Out_Tmin_kN": "The absolute minimum possible tension required to span this gap, assuming infinite sag.",
    "🧮 Out_a_minT_m": "The catenary parameter 'a' that corresponds to the absolute minimum tension state.",
    "📍 Out_Vertex_x0_m": "The horizontal distance from the start point to the lowest point (vertex) of the cable.",
    "⚠️ Out_Req_Tallow_kN": "The maximum tension occurring in the cable, which dictates the required material capacity.",
    "🏗️ Out_Geometry_Possible": "Checks if the required tension is physically less than or equal to the allowable capacity."
}

grid_html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 15px;">\n'
for key, unit, is_num in parameters:
    m_val = mathRes['metrics'].get(key, "NaN") if mathRes['possible'] else "NaN"
    s_val = surrMetrics.get(key, "NaN") if mathRes['possible'] else "NaN"
    p_val = pinnMetrics.get(key, "NaN") if mathRes['possible'] else "NaN"

    if is_num and m_val != "NaN":
        s_str, s_diff, s_acc, s_bg, s_tc = calc_diff_acc(m_val, s_val)
        p_str, p_diff, p_acc, p_bg, p_tc = calc_diff_acc(m_val, p_val)
        try: m_str = f"{float(m_val):.2f}"
        except: m_str = str(m_val)
    else:
        m_str, s_str, p_str = str(m_val), str(s_val), str(p_val)
        s_diff, s_acc, s_bg, s_tc = "", "", "transparent", "transparent"
        p_diff, p_acc, p_bg, p_tc = "", "", "transparent", "transparent"

    tt_html = f'<span title="{tooltips.get(key, "")}" style="cursor: help; color: #aaa; float: right; font-size: 1rem;">(?)</span>'

    grid_html += f"""<div style="background:#ffffff; border:1px solid #e9ecef; border-radius:10px; padding:15px; box-shadow:0 4px 6px rgba(0,0,0,0.05); color:#111;">
<h5 style="margin:0 0 10px 0; color:#555; border-bottom:1px solid #eee; padding-bottom:5px; position: relative;">{key} {tt_html}</h5>
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<div style="color:#007bff; font-weight:bold; font-size:0.9rem;">Exact Math</div>
<span style="font-size:1.1rem; font-family:monospace; font-weight:bold;">{m_str} <span style="font-size:0.8rem;color:#888;">{unit}</span></span>
</div>
<div style="background:#fcf9fd; border-left:4px solid #9b59b6; padding:8px; border-radius:4px; margin-bottom:8px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
<span style="color:#9b59b6; font-size:0.85rem; font-weight:bold;">Surrogate AI</span>
<span style="font-family:monospace; font-weight:bold;">{s_str} <span style="font-size:0.8rem;color:#888;">{unit}</span></span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem;">
<span style="color:#555;">Diff: <b>{s_diff}</b></span>
<span style="background:{s_bg}; color:{s_tc}; padding:2px 6px; border-radius:10px; font-weight:bold;">{s_acc}</span>
</div>
</div>
<div style="background:#fdfaf7; border-left:4px solid #e67e22; padding:8px; border-radius:4px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
<span style="color:#e67e22; font-size:0.85rem; font-weight:bold;">PINN AI</span>
<span style="font-family:monospace; font-weight:bold;">{p_str} <span style="font-size:0.8rem;color:#888;">{unit}</span></span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem;">
<span style="color:#555;">Diff: <b>{p_diff}</b></span>
<span style="background:{p_bg}; color:{p_tc}; padding:2px 6px; border-radius:10px; font-weight:bold;">{p_acc}</span>
</div>
</div>
</div>
"""
grid_html += '</div>'
st.markdown(grid_html, unsafe_allow_html=True)