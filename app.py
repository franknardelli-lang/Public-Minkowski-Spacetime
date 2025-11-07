# Streamlit + Plotly Minkowski diagram (visualization only, no dragging yet)

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---- If you already have these in minkowski_math.py, you can import instead:
# from minkowski_math import lorentz_transform, relative_velocity, spacetime_intervals

def lorentz_transform(points, v):
    """(t,x) -> (t',x') for velocity v (c=1). points: Nx2 array [t,x]."""
    gamma = 1.0 / math.sqrt(1.0 - v*v + 1e-15)
    t, x = points[:, 0], points[:, 1]
    t_p = gamma * (t - v * x)
    x_p = gamma * (x - v * t)
    return np.column_stack((t_p, x_p))

def relative_velocity(u, v):
    """Einstein velocity addition u ⊕ (-v) = (u - v)/(1 - u v)."""
    return (u - v) / (1.0 - u*v + 1e-15)

def spacetime_intervals(events, c=1.0, v_frame=0.0):
    """Invariant intervals among A,B,C as seen in selected frame (for readout)."""
    tp = lorentz_transform(events, v_frame)
    (tA, xA), (tB, xB), (tC, xC) = tp

    def S2(a, b):  # Minkowski (+,-) with c=1
        return (c*(b[0] - a[0]))**2 - (b[1] - a[1])**2

    return S2((tA,xA),(tB,xB)), S2((tA,xA),(tC,xC)), S2((tB,xB),(tC,xC))

# --------------------------------------------------------------------
# Plot construction
# --------------------------------------------------------------------
def make_hyperbolae_traces(x_extent, t_extent):
    """Timelike (blue) and spacelike (red) hyperbolae background."""
    traces = []
    xs = np.linspace(-x_extent, x_extent, 600)
    ts = np.linspace(-t_extent, t_extent, 600)

    # Timelike: t^2 - x^2 = s^2  =>  t = ±sqrt(x^2 + s^2)
    for s in [1, 2, 3, 4, 5]:
        t = np.sqrt(xs**2 + s**2)
        traces += [
            go.Scatter(x=xs, y= t, mode="lines", line=dict(color="blue", width=1), opacity=0.18, hoverinfo="skip", showlegend=False),
            go.Scatter(x=xs, y=-t, mode="lines", line=dict(color="blue", width=1), opacity=0.18, hoverinfo="skip", showlegend=False),
        ]

    # Spacelike: x^2 - t^2 = s^2  =>  x = ±sqrt(t^2 + s^2)
    for s in [1, 2, 3, 4, 5]:
        x = np.sqrt(ts**2 + s**2)
        traces += [
            go.Scatter(x= x, y=ts, mode="lines", line=dict(color="red", width=1), opacity=0.18, hoverinfo="skip", showlegend=False),
            go.Scatter(x=-x, y=ts, mode="lines", line=dict(color="red", width=1), opacity=0.18, hoverinfo="skip", showlegend=False),
        ]
    return traces

def build_figure(events, vB, vC, frame, x_extent, t_extent,fig_height=720):
    """Assemble the full diagram in the *selected frame*."""
    # Active frame velocity
    v_frame = 0.0 if frame == "A" else (vB if frame == "B" else vC)
    gamma   = 1.0 / math.sqrt(1.0 - v_frame*v_frame + 1e-15)

    # Transform event coordinates -> (t',x') in active frame
    tp = lorentz_transform(events, v_frame)
    (tA, xA), (tB, xB), (tC, xC) = tp

    # Relative velocities B,C as seen in active frame
    vb_rel = relative_velocity(vB, v_frame)
    vc_rel = relative_velocity(vC, v_frame)

    xlim = (-x_extent, x_extent)
    tlim = (-t_extent, t_extent)

    fig = go.Figure()

    # Axes
    fig.add_shape(type="line", x0=xlim[0], x1=xlim[1], y0=0, y1=0, line=dict(color="black", width=1))
    fig.add_shape(type="line", x0=0, x1=0, y0=tlim[0], y1=tlim[1], line=dict(color="black", width=1))

    # Background hyperbolae
    for tr in make_hyperbolae_traces(x_extent, t_extent):
        fig.add_trace(tr)

    # Worldlines for observers B and C (in active frame)
    t_world = np.linspace(-t_extent, t_extent, 400)
    for (v_obs, x0, t0) in [(vB, xB, tB), (vC, xC, tC)]:
        x_world = v_obs * t_world + (x0 - v_obs * t0)  # x = v t + (x0 - v t0)
        fig.add_trace(go.Scatter(
            x=x_world, y=t_world, mode="lines",
            line=dict(color="gray", dash="dot"), opacity=0.55,
            hoverinfo="skip", showlegend=False
        ))

    # Simultaneity (cyan) + time-axis (magenta) for B and C, relative to active frame
    X = np.linspace(-x_extent, x_extent, 600)
    for (x0, t0, vrel) in [(xB, tB, vb_rel), (xC, tC, vc_rel)]:
        # Simultaneity: t = t0 + vrel (X - x0)
        t_sim = t0 + vrel * (X - x0)
        fig.add_trace(go.Scatter(x=X, y=t_sim, mode="lines",
                                 line=dict(color="cyan", width=2),
                                 hoverinfo="skip", showlegend=False))
        # Time axis: slope = 1/vrel, vertical if vrel≈0
        if abs(vrel) < 1e-9:
            fig.add_trace(go.Scatter(x=[x0, x0], y=[tlim[0], tlim[1]], mode="lines",
                                     line=dict(color="magenta", width=2),
                                     hoverinfo="skip", showlegend=False))
        else:
            slope = 1.0 / vrel
            t_line = t0 + slope * (X - x0)
            fig.add_trace(go.Scatter(x=X, y=t_line, mode="lines",
                                     line=dict(color="magenta", width=2),
                                     hoverinfo="skip", showlegend=False))

    # Light cones from A (highlight red), B, C (gray)
    xx = np.linspace(-x_extent, x_extent, 600)
    for (x0, t0, color, alpha) in [(xA, tA, "red", 0.35),
                                   (xB, tB, "gray", 0.25),
                                   (xC, tC, "gray", 0.25)]:
        fig.add_trace(go.Scatter(x=xx, y=t0 + (xx - x0), mode="lines",
                                 line=dict(color=color, dash="dash", width=1),
                                 opacity=alpha, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xx, y=t0 - (xx - x0), mode="lines",
                                 line=dict(color=color, dash="dash", width=1),
                                 opacity=alpha, hoverinfo="skip", showlegend=False))

    # Event markers + labels
    for (x, t, label, color) in [(xA, tA, "A", "red"),
                                 (xB, tB, "B", "blue"),
                                 (xC, tC, "C", "green")]:
        fig.add_trace(go.Scatter(
            x=[x], y=[t], mode="markers+text", text=[label], textposition="top center",
            marker=dict(size=10, color=color),
            hovertemplate=f"{label}: (t=%{{y:.2f}}, x=%{{x:.2f}})<extra></extra>",
            showlegend=False
        ))

    frame_color = {"A": "red", "B": "blue", "C": "green"}.get(frame, "black")
    fig.update_layout(
        height=fig_height,
        xaxis=dict(range=xlim, title="Space (x)"),
        yaxis=dict(range=tlim, title="Time (t)", scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=50, b=10),
        plot_bgcolor="white",
        title=f"Interactive Minkowski Spacetime Diagram — Frame {frame}  (γ={gamma:.3f}, v={v_frame:.2f}c)",
        title_font=dict(color=frame_color),
        showlegend=False,
    )
    return fig, v_frame

# --------------------------------------------------------------------
# Streamlit UI (visualization only)
# --------------------------------------------------------------------
st.set_page_config(page_title="Minkowski (Plotly, Visual Only)", layout="wide")
st.title("🕘 Minkowski Spacetime Diagram — Visualization (no dragging yet)")

# Session defaults (base frame coordinates)
if "events" not in st.session_state:
    st.session_state.events = np.array([[0.0, 0.0],   # A: (t,x)
                                        [4.0, 2.0],   # B
                                        [1.0, 3.0]],  # C
                                       dtype=float)

# Sidebar controls
st.sidebar.header("Parameters")

# Events (base frame) — editable
st.sidebar.subheader("Event A")
tA = st.sidebar.number_input("t_A", value=float(st.session_state.events[0, 0]), step=0.1, format="%.2f")
xA = st.sidebar.number_input("x_A", value=float(st.session_state.events[0, 1]), step=0.1, format="%.2f")
st.sidebar.subheader("Event B")
tB = st.sidebar.number_input("t_B", value=float(st.session_state.events[1, 0]), step=0.1, format="%.2f")
xB = st.sidebar.number_input("x_B", value=float(st.session_state.events[1, 1]), step=0.1, format="%.2f")
st.sidebar.subheader("Event C")
tC = st.sidebar.number_input("t_C", value=float(st.session_state.events[2, 0]), step=0.1, format="%.2f")
xC = st.sidebar.number_input("x_C", value=float(st.session_state.events[2, 1]), step=0.1, format="%.2f")

st.session_state.events = np.array([[tA, xA], [tB, xB], [tC, xC]], dtype=float)

# Velocities & frame
st.sidebar.subheader("Observer Velocities (v/c)")
vB = st.sidebar.slider("v_B", -0.99, 0.99, 0.60, 0.01)
vC = st.sidebar.slider("v_C", -0.99, 0.99, -0.40, 0.01)
frame = st.sidebar.radio("Current Frame", ["A", "B", "C"], index=0)

# Axes extents
st.sidebar.subheader("Diagram Limits")
x_extent = st.sidebar.slider("Space extent ±X", 4, 20, 10, 1)
t_extent = st.sidebar.slider("Time extent ±T",  4, 20, 10, 1)

# Reset
if st.sidebar.button("Reset defaults"):
    st.session_state.events = np.array([[0.0, 0.0], [4.0, 2.0], [1.0, 3.0]], dtype=float)
    vB, vC, frame = 0.60, -0.40, "A"

# Build + render
fig, v_frame = build_figure(st.session_state.events, vB, vC, frame, x_extent, t_extent)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

# Readouts
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Spacetime Intervals (invariant)")
    S2_AB, S2_AC, S2_BC = spacetime_intervals(st.session_state.events, c=1.0, v_frame=v_frame)

    def describe(S2):
        if S2 > 0:   return f"timelike (S = {math.sqrt(S2):.3f})"
        if S2 < 0:   return f"spacelike (S = {math.sqrt(-S2):.3f} i)"
        return "lightlike (S = 0)"

    st.markdown(
        f"- **S²_AB = {S2_AB:.3f}** — {describe(S2_AB)}\n"
        f"- **S²_AC = {S2_AC:.3f}** — {describe(S2_AC)}\n"
        f"- **S²_BC = {S2_BC:.3f}** — {describe(S2_BC)}"
    )

with col2:
    st.subheader("Lorentz Transform (active frame)")
    st.latex(r"t' = \gamma (t - v x), \quad x' = \gamma (x - v t)")
    st.write(f"**Frame {frame}**:  **v = {v_frame:.2f} c**,  **γ = {1.0 / math.sqrt(1.0 - v_frame**2 + 1e-15):.3f}**")
    st.caption("Hyperbolae: blue (timelike), red (spacelike). Cyan: simultaneity. Magenta: time axes. Dotted: light cones.")
