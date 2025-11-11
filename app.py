import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Set page config
st.set_page_config(
    page_title="Minkowski Spacetime Diagram",
    page_icon="🌌",
    layout="wide"
)

# Title
st.title("🌌 Interactive Minkowski Spacetime Diagram")
st.markdown("Explore special relativity through interactive spacetime diagrams")

# Helper functions from minkowski_math.py
def lorentz_transform(points, v):
    """Apply Lorentz transformation to a set of (t, x) points."""
    gamma = 1 / np.sqrt(1 - v*v)
    t, x = points[:, 0], points[:, 1]
    t_p = gamma * (t - v * x)
    x_p = gamma * (x - v * t)
    return np.column_stack((t_p, x_p))

def relative_velocity(u, v):
    """Velocity addition formula for special relativity."""
    return (u - v) / (1 - u * v)

def draw_static_elements(ax):
    """Draw spacetime hyperbolae on the Minkowski diagram."""
    s_vals = [1, 2, 3, 4, 5]
    for s in s_vals:
        x = np.linspace(s, 10, 400)
        t = np.sqrt(x**2 - s**2)
        ax.plot(x, t, 'b-', alpha=0.2)
        ax.plot(-x, t, 'b-', alpha=0.2)
        ax.plot(x, -t, 'b-', alpha=0.2)
        ax.plot(-x, -t, 'b-', alpha=0.2)
    for s in s_vals:
        x = np.linspace(-10, 10, 400)
        t = np.sqrt(x**2 + s**2)
        ax.plot(x, t, 'r-', alpha=0.2)
        ax.plot(x, -t, 'r-', alpha=0.2)

# Initialize session state
if 'eventA' not in st.session_state:
    st.session_state.eventA = np.array([0.0, 0.0])
    st.session_state.eventB = np.array([4.0, 2.0])
    st.session_state.eventC = np.array([1.0, 3.0])
    st.session_state.v_b = 0.6
    st.session_state.v_c = -0.4
    st.session_state.current_frame = 'A'

# Sidebar controls
st.sidebar.header("⚙️ Controls")

# Frame selection
st.sidebar.subheader("Reference Frame")
frame = st.sidebar.radio(
    "Select Frame:",
    ['A', 'B', 'C'],
    index=['A', 'B', 'C'].index(st.session_state.current_frame),
    horizontal=True
)
st.session_state.current_frame = frame

st.sidebar.markdown("---")

# Velocity controls
st.sidebar.subheader("Velocities (v/c)")
v_b = st.sidebar.slider("Event B velocity", -0.99, 0.99, st.session_state.v_b, 0.01)
v_c = st.sidebar.slider("Event C velocity", -0.99, 0.99, st.session_state.v_c, 0.01)
st.session_state.v_b = v_b
st.session_state.v_c = v_c

st.sidebar.markdown("---")

# Event coordinates
st.sidebar.subheader("Event Coordinates")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("**Event A**")
    tA = st.number_input("t_A", value=float(st.session_state.eventA[0]), format="%.2f", key="tA")
    xA = st.number_input("x_A", value=float(st.session_state.eventA[1]), format="%.2f", key="xA")
    
with col2:
    st.markdown("**Event B**")
    tB = st.number_input("t_B", value=float(st.session_state.eventB[0]), format="%.2f", key="tB")
    xB = st.number_input("x_B", value=float(st.session_state.eventB[1]), format="%.2f", key="xB")

col3, col4 = st.sidebar.columns(2)
with col3:
    st.markdown("**Event C**")
    tC = st.number_input("t_C", value=float(st.session_state.eventC[0]), format="%.2f", key="tC")
    
with col4:
    st.markdown("**&nbsp;**")
    xC = st.number_input("x_C", value=float(st.session_state.eventC[1]), format="%.2f", key="xC")

# Update events
st.session_state.eventA = np.array([tA, xA])
st.session_state.eventB = np.array([tB, xB])
st.session_state.eventC = np.array([tC, xC])

# Main plot
v_frame = 0 if frame == 'A' else (v_b if frame == 'B' else v_c)
pts = np.array([st.session_state.eventA, st.session_state.eventB, st.session_state.eventC])
Atp = lorentz_transform(pts, v_frame)
(tA_p, xA_p), (tB_p, xB_p), (tC_p, xC_p) = Atp

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.set_xlabel('Space (x)', fontsize=12)
ax.set_ylabel('Time (t)', fontsize=12)

# Set title with color
color_map = {'A': 'red', 'B': 'blue', 'C': 'green'}
title_color = color_map.get(frame, 'black')
ax.set_title(f'Interactive Minkowski Spacetime Diagram — Frame {frame}', 
             color=title_color, fontsize=16, weight='bold')

ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3)

# Draw static elements
draw_static_elements(ax)

# Worldlines
t_world = np.linspace(-10, 10, 200)
for (v, x0, t0, color) in [(v_b, xB_p, tB_p, 'blue'), (v_c, xC_p, tC_p, 'green')]:
    x_world = v * t_world + (x0 - v * t0)
    ax.plot(x_world, t_world, color=color, alpha=0.3, linewidth=2, linestyle='--')

# Simultaneity & Time axes
vb_rel = relative_velocity(v_b, v_frame)
vc_rel = relative_velocity(v_c, v_frame)
X = np.linspace(-10, 10, 300)

for (x0, t0, vrel, color) in [(xB_p, tB_p, vb_rel, 'b'), (xC_p, tC_p, vc_rel, 'g')]:
    # Simultaneity line
    t_sim = t0 + vrel * (X - x0)
    ax.plot(X, t_sim, 'c-', linewidth=2, alpha=0.6)
    
    # Time axis
    slope = np.inf if abs(vrel) < 1e-3 else 1 / vrel
    if np.isinf(slope):
        x_line = np.full_like(X, x0)
        t_line = np.linspace(-10, 10, 300)
    else:
        x_line = X
        t_line = t0 + slope * (X - x0)
    ax.plot(x_line, t_line, 'm-', linewidth=2, alpha=0.6)

# Light cones
t_range = np.linspace(-10, 10, 400)
for (x0, t0, color, alpha) in [(xA_p, tA_p, 'r', 0.4), (xB_p, tB_p, 'gray', 0.2), (xC_p, tC_p, 'gray', 0.2)]:
    t_plus = t0 + (t_range - x0)
    t_minus = t0 - (t_range - x0)
    ax.plot(t_range, t_plus, linestyle='--', color=color, alpha=alpha, linewidth=1.5)
    ax.plot(t_range, t_minus, linestyle='--', color=color, alpha=alpha, linewidth=1.5)

# Plot events
ax.plot(xA_p, tA_p, 'ro', markersize=12, label='Event A', zorder=5)
ax.plot(xB_p, tB_p, 'bo', markersize=12, label='Event B', zorder=5)
ax.plot(xC_p, tC_p, 'go', markersize=12, label='Event C', zorder=5)

# Event labels
for (x, t, name, color) in [(xA_p, tA_p, 'A', 'r'), (xB_p, tB_p, 'B', 'b'), (xC_p, tC_p, 'C', 'g')]:
    ax.text(x + 0.5, t + 0.5, name, color=color, weight='bold', fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

# Frame info
gamma = 1 / np.sqrt(1 - v_frame**2)
ax.text(0.05, 0.95, f"Frame {frame}: v = {v_frame:.2f}c, γ = {gamma:.3f}",
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

ax.legend(loc='upper right', fontsize=10)

st.pyplot(fig)
plt.close()

# Information panels
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Spacetime Intervals")
    st.markdown("**Formula:** $S^2 = (c \\Delta t)^2 - (\\Delta x)^2$")
    
    c = 1
    def S2(a, b):
        return (c*(b[0]-a[0]))**2 - (b[1]-a[1])**2
    
    S2_AB = S2((tA_p, xA_p), (tB_p, xB_p))
    S2_AC = S2((tA_p, xA_p), (tC_p, xC_p))
    S2_BC = S2((tB_p, xB_p), (tC_p, xC_p))
    
    def interval_type(S2_val):
        if S2_val > 0:
            return f"timelike (S = {np.sqrt(S2_val):.3f})"
        elif S2_val < 0:
            return f"spacelike (S = {np.sqrt(-S2_val):.3f}i)"
        else:
            return "lightlike (S = 0)"
    
    st.write(f"**A↔B:** $S^2_{{AB}} = {S2_AB:.3f}$ — {interval_type(S2_AB)}")
    st.write(f"**A↔C:** $S^2_{{AC}} = {S2_AC:.3f}$ — {interval_type(S2_AC)}")
    st.write(f"**B↔C:** $S^2_{{BC}} = {S2_BC:.3f}$ — {interval_type(S2_BC)}")

with col2:
    st.subheader("🔄 Lorentz Transform")
    st.markdown(f"**Frame {frame} Transform:**")
    st.latex(r"t' = \gamma (t - v x)")
    st.latex(r"x' = \gamma (x - v t)")
    st.write(f"**Current values:** $v = {v_frame:.2f}c$, $\\gamma = {gamma:.3f}$")

# Legend
st.subheader("🎨 Diagram Color Key")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("🟣 **Magenta**: Time Axes")
with col2:
    st.markdown("🔵 **Cyan**: Simultaneity Lines")
with col3:
    st.markdown("⚫ **Gray**: Worldlines")
with col4:
    st.markdown("📍 **Dashed**: Light Cones")

# Info section
with st.expander("ℹ️ About This Visualization"):
    st.markdown("""
    This interactive Minkowski spacetime diagram helps visualize concepts from special relativity:
    
    - **Events A, B, C**: Points in spacetime with coordinates (t, x)
    - **Reference Frames**: Switch between different observers' perspectives
    - **Lorentz Transformation**: See how coordinates change between frames
    - **Spacetime Intervals**: Invariant measures between events
    - **Light Cones**: Regions of causally connected events (45° lines)
    - **Worldlines**: Paths of objects through spacetime
    
    Adjust the velocities and event positions to explore how special relativity works!
    """)