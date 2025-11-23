import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import io
import json
from datetime import datetime
import pandas as pd

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

@st.cache_data
def compute_hyperbola_lines():
    """Compute spacetime hyperbola coordinates (cached for performance)."""
    s_vals = [1, 2, 3, 4, 5]
    timelike_lines = []
    spacelike_lines = []

    # Timelike hyperbolae (blue)
    for s in s_vals:
        x = np.linspace(s, 10, 400)
        t = np.sqrt(x**2 - s**2)
        timelike_lines.extend([
            (x, t), (-x, t), (x, -t), (-x, -t)
        ])

    # Spacelike hyperbolae (red)
    for s in s_vals:
        x = np.linspace(-10, 10, 400)
        t = np.sqrt(x**2 + s**2)
        spacelike_lines.extend([
            (x, t), (x, -t)
        ])

    return timelike_lines, spacelike_lines

def draw_static_elements(ax):
    """Draw spacetime hyperbolae on the Minkowski diagram."""
    timelike_lines, spacelike_lines = compute_hyperbola_lines()

    for x, t in timelike_lines:
        ax.plot(x, t, 'b-', alpha=0.2)

    for x, t in spacelike_lines:
        ax.plot(x, t, 'r-', alpha=0.2)

# Initialize session state
if 'eventA' not in st.session_state:
    st.session_state.eventA = np.array([0.0, 0.0])
    st.session_state.eventB = np.array([4.0, 2.0])
    st.session_state.eventC = np.array([1.0, 3.0])
    st.session_state.v_b = 0.6
    st.session_state.v_c = -0.4
    st.session_state.current_frame = 'A'
    st.session_state.input_mode = 'Sliders'  # 'Sliders' or 'Precision'
    st.session_state.default_eventA = np.array([0.0, 0.0])
    st.session_state.default_eventB = np.array([4.0, 2.0])
    st.session_state.default_eventC = np.array([1.0, 3.0])
    st.session_state.default_v_b = 0.6
    st.session_state.default_v_c = -0.4

# Sidebar controls
st.sidebar.header("⚙️ Controls")

# Preset scenarios
st.sidebar.subheader("📚 Preset Scenarios")
preset_scenarios = {
    'Custom': None,
    'Default': {
        'eventA': [0.0, 0.0],
        'eventB': [4.0, 2.0],
        'eventC': [1.0, 3.0],
        'v_b': 0.6,
        'v_c': -0.4
    },
    'Twin Paradox': {
        'eventA': [0.0, 0.0],  # Departure
        'eventB': [5.0, 4.0],  # Turnaround point
        'eventC': [10.0, 0.0],  # Return
        'v_b': 0.8,
        'v_c': -0.8
    },
    'Simultaneity': {
        'eventA': [0.0, 0.0],
        'eventB': [0.0, 5.0],  # Simultaneous in frame A
        'eventC': [0.0, -5.0],  # Simultaneous in frame A
        'v_b': 0.5,
        'v_c': 0.5
    },
    'Train Platform': {
        'eventA': [0.0, -3.0],  # Front of train at platform start
        'eventB': [0.0, 3.0],   # Back of train at platform end
        'eventC': [2.0, 0.0],   # Lightning strike
        'v_b': 0.7,
        'v_c': 0.0
    },
    'Time Dilation': {
        'eventA': [0.0, 0.0],   # Start
        'eventB': [5.0, 0.0],   # Clock tick in rest frame
        'eventC': [5.0, 3.0],   # Clock tick in moving frame
        'v_b': 0.0,
        'v_c': 0.6
    },
    'Light Signal': {
        'eventA': [0.0, 0.0],   # Light emission
        'eventB': [5.0, 5.0],   # Light reception (45° worldline)
        'eventC': [3.0, -3.0],  # Light in opposite direction
        'v_b': 0.5,
        'v_c': 0.0
    }
}

preset = st.sidebar.selectbox(
    "Choose a scenario:",
    list(preset_scenarios.keys()),
    help="Select a preset to explore common special relativity scenarios"
)

# Apply preset if selected (and not Custom)
if preset != 'Custom' and preset_scenarios[preset] is not None:
    scenario = preset_scenarios[preset]
    st.session_state.eventA = np.array(scenario['eventA'])
    st.session_state.eventB = np.array(scenario['eventB'])
    st.session_state.eventC = np.array(scenario['eventC'])
    st.session_state.v_b = scenario['v_b']
    st.session_state.v_c = scenario['v_c']

st.sidebar.markdown("---")

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

# Input mode toggle
input_mode = st.sidebar.radio(
    "Input Mode:",
    ['Sliders', 'Precision'],
    index=['Sliders', 'Precision'].index(st.session_state.input_mode),
    horizontal=True,
    help="Sliders for quick exploration, Precision for exact values"
)
st.session_state.input_mode = input_mode

# Event coordinate inputs based on mode
if input_mode == 'Sliders':
    st.sidebar.markdown("**Event A** 🔴")
    tA = st.sidebar.slider("t_A", -10.0, 10.0, float(st.session_state.eventA[0]), 0.1, key="tA_slider")
    xA = st.sidebar.slider("x_A", -10.0, 10.0, float(st.session_state.eventA[1]), 0.1, key="xA_slider")

    st.sidebar.markdown("**Event B** 🔵")
    tB = st.sidebar.slider("t_B", -10.0, 10.0, float(st.session_state.eventB[0]), 0.1, key="tB_slider")
    xB = st.sidebar.slider("x_B", -10.0, 10.0, float(st.session_state.eventB[1]), 0.1, key="xB_slider")

    st.sidebar.markdown("**Event C** 🟢")
    tC = st.sidebar.slider("t_C", -10.0, 10.0, float(st.session_state.eventC[0]), 0.1, key="tC_slider")
    xC = st.sidebar.slider("x_C", -10.0, 10.0, float(st.session_state.eventC[1]), 0.1, key="xC_slider")
else:  # Precision mode
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("**Event A** 🔴")
        tA = st.number_input("t_A", value=float(st.session_state.eventA[0]), format="%.2f", key="tA_input")
        xA = st.number_input("x_A", value=float(st.session_state.eventA[1]), format="%.2f", key="xA_input")

    with col2:
        st.markdown("**Event B** 🔵")
        tB = st.number_input("t_B", value=float(st.session_state.eventB[0]), format="%.2f", key="tB_input")
        xB = st.number_input("x_B", value=float(st.session_state.eventB[1]), format="%.2f", key="xB_input")

    col3, col4 = st.sidebar.columns(2)
    with col3:
        st.markdown("**Event C** 🟢")
        tC = st.number_input("t_C", value=float(st.session_state.eventC[0]), format="%.2f", key="tC_input")

    with col4:
        st.markdown("**&nbsp;**")
        xC = st.number_input("x_C", value=float(st.session_state.eventC[1]), format="%.2f", key="xC_input")

# Update events
st.session_state.eventA = np.array([tA, xA])
st.session_state.eventB = np.array([tB, xB])
st.session_state.eventC = np.array([tC, xC])

# Reset button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset to Default", help="Reset all values to default configuration", use_container_width=True):
    st.session_state.eventA = st.session_state.default_eventA.copy()
    st.session_state.eventB = st.session_state.default_eventB.copy()
    st.session_state.eventC = st.session_state.default_eventC.copy()
    st.session_state.v_b = st.session_state.default_v_b
    st.session_state.v_c = st.session_state.default_v_c
    st.session_state.current_frame = 'A'
    st.rerun()

# Plot size control
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Settings")
plot_size = st.sidebar.select_slider(
    "Plot Size:",
    options=['Small', 'Medium', 'Large', 'X-Large'],
    value='Large',
    help="Adjust plot size for your screen"
)

# Map size to figure dimensions and display width
size_map = {
    'Small': {'figsize': (8, 8), 'width': 400, 'dpi': 100},
    'Medium': {'figsize': (10, 10), 'width': 600, 'dpi': 100},
    'Large': {'figsize': (12, 12), 'width': 900, 'dpi': 100},
    'X-Large': {'figsize': (14, 14), 'width': None, 'dpi': 100}  # None = use container width
}
plot_config = size_map[plot_size]
figsize = plot_config['figsize']
plot_width = plot_config['width']
plot_dpi = plot_config['dpi']

# Configuration export/import
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Save/Load Configuration")

# Export configuration
config_data = {
    'eventA': st.session_state.eventA.tolist(),
    'eventB': st.session_state.eventB.tolist(),
    'eventC': st.session_state.eventC.tolist(),
    'v_b': st.session_state.v_b,
    'v_c': st.session_state.v_c,
    'frame': st.session_state.current_frame,
    'timestamp': datetime.now().isoformat()
}

config_json = json.dumps(config_data, indent=2)

st.sidebar.download_button(
    label="💾 Export Config (JSON)",
    data=config_json,
    file_name=f"minkowski_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
    help="Download current configuration as JSON",
    use_container_width=True
)

# Import configuration
uploaded_file = st.sidebar.file_uploader(
    "📂 Import Config (JSON)",
    type=['json'],
    help="Upload a previously saved configuration"
)

if uploaded_file is not None:
    try:
        config = json.load(uploaded_file)
        st.session_state.eventA = np.array(config['eventA'])
        st.session_state.eventB = np.array(config['eventB'])
        st.session_state.eventC = np.array(config['eventC'])
        st.session_state.v_b = config['v_b']
        st.session_state.v_c = config['v_c']
        st.session_state.current_frame = config.get('frame', 'A')
        st.sidebar.success("✅ Configuration loaded successfully!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Error loading configuration: {str(e)}")

# Main plot
v_frame = 0 if frame == 'A' else (v_b if frame == 'B' else v_c)
pts = np.array([st.session_state.eventA, st.session_state.eventB, st.session_state.eventC])
Atp = lorentz_transform(pts, v_frame)
(tA_p, xA_p), (tB_p, xB_p), (tC_p, xC_p) = Atp

# Create figure with responsive sizing
fig, ax = plt.subplots(figsize=figsize, dpi=plot_dpi)
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

# Display plot with precise size control using st.image
buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=plot_dpi, bbox_inches='tight', facecolor='white')
buf.seek(0)

if plot_width is None:
    # X-Large: use full container width
    st.image(buf, use_container_width=True)
else:
    # Small/Medium/Large: use specific pixel width
    st.image(buf, width=plot_width)

# Export plot buttons
col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 3])

with col_exp1:
    # Export as PNG
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
    buf_png.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="📥 Download PNG",
        data=buf_png,
        file_name=f"minkowski_diagram_{timestamp}.png",
        mime="image/png",
        help="Download current plot as PNG image",
        use_container_width=True
    )

with col_exp2:
    # Export as PDF
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
    buf_pdf.seek(0)
    st.download_button(
        label="📄 Download PDF",
        data=buf_pdf,
        file_name=f"minkowski_diagram_{timestamp}.pdf",
        mime="application/pdf",
        help="Download current plot as PDF",
        use_container_width=True
    )

# Close the figure to free memory
plt.close(fig)

# Coordinate Display Panel
st.markdown("---")
st.subheader("📍 Event Coordinates Comparison")

# Create coordinate comparison table
col_coord1, col_coord2 = st.columns(2)

with col_coord1:
    st.markdown(f"**Rest Frame (Frame A)** 🔴")

    # Get coordinates in rest frame (Frame A)
    rest_eventA = st.session_state.eventA
    rest_eventB = st.session_state.eventB
    rest_eventC = st.session_state.eventC

    # Create dataframe for rest frame
    rest_data = {
        'Event': ['A 🔴', 'B 🔵', 'C 🟢'],
        't (time)': [f"{rest_eventA[0]:.3f}", f"{rest_eventB[0]:.3f}", f"{rest_eventC[0]:.3f}"],
        'x (space)': [f"{rest_eventA[1]:.3f}", f"{rest_eventB[1]:.3f}", f"{rest_eventC[1]:.3f}"]
    }
    rest_df = pd.DataFrame(rest_data)

    # Style the dataframe
    styled_rest_df = rest_df.style.set_properties(**{
        'background-color': '#f0f2f6',
        'color': 'black',
        'border-color': 'white'
    })
    st.dataframe(styled_rest_df, hide_index=True, use_container_width=True)

with col_coord2:
    st.markdown(f"**Current Frame (Frame {frame})** {'🔴' if frame == 'A' else '🔵' if frame == 'B' else '🟢'}")

    # Get coordinates in current frame (already computed)
    current_data = {
        'Event': ['A 🔴', 'B 🔵', 'C 🟢'],
        "t' (time)": [f"{tA_p:.3f}", f"{tB_p:.3f}", f"{tC_p:.3f}"],
        "x' (space)": [f"{xA_p:.3f}", f"{xB_p:.3f}", f"{xC_p:.3f}"]
    }
    current_df = pd.DataFrame(current_data)

    # Style the dataframe with frame-specific color
    if frame == 'A':
        bg_color = '#ffe6e6'  # Light red
    elif frame == 'B':
        bg_color = '#e6f2ff'  # Light blue
    else:
        bg_color = '#e6ffe6'  # Light green

    styled_current_df = current_df.style.set_properties(**{
        'background-color': bg_color,
        'color': 'black',
        'border-color': 'white'
    })
    st.dataframe(styled_current_df, hide_index=True, use_container_width=True)

# Show transformation details
if frame != 'A':
    st.info(f"💡 **Transformation:** Coordinates transformed with velocity **v = {v_frame:.2f}c** and Lorentz factor **γ = {gamma:.3f}**")
else:
    st.info("💡 **Note:** Currently viewing rest frame - no transformation applied")

# Proper time display
st.markdown("### ⏱️ Proper Time from Origin")

def calculate_proper_time(t, x):
    """Calculate proper time (invariant interval from origin)"""
    s_squared = t**2 - x**2
    if s_squared > 0:
        return np.sqrt(s_squared), "timelike"
    elif s_squared < 0:
        return np.sqrt(-s_squared), "spacelike"
    else:
        return 0.0, "lightlike"

col_tau1, col_tau2, col_tau3 = st.columns(3)

# Event A
tau_A, type_A = calculate_proper_time(rest_eventA[0], rest_eventA[1])
with col_tau1:
    if type_A == "timelike":
        st.metric("Event A 🔴", f"τ = {tau_A:.3f}", delta="Timelike", delta_color="normal")
    elif type_A == "spacelike":
        st.metric("Event A 🔴", f"σ = {tau_A:.3f}", delta="Spacelike", delta_color="inverse")
    else:
        st.metric("Event A 🔴", f"s = {tau_A:.3f}", delta="Lightlike", delta_color="off")

# Event B
tau_B, type_B = calculate_proper_time(rest_eventB[0], rest_eventB[1])
with col_tau2:
    if type_B == "timelike":
        st.metric("Event B 🔵", f"τ = {tau_B:.3f}", delta="Timelike", delta_color="normal")
    elif type_B == "spacelike":
        st.metric("Event B 🔵", f"σ = {tau_B:.3f}", delta="Spacelike", delta_color="inverse")
    else:
        st.metric("Event B 🔵", f"s = {tau_B:.3f}", delta="Lightlike", delta_color="off")

# Event C
tau_C, type_C = calculate_proper_time(rest_eventC[0], rest_eventC[1])
with col_tau3:
    if type_C == "timelike":
        st.metric("Event C 🟢", f"τ = {tau_C:.3f}", delta="Timelike", delta_color="normal")
    elif type_C == "spacelike":
        st.metric("Event C 🟢", f"σ = {tau_C:.3f}", delta="Spacelike", delta_color="inverse")
    else:
        st.metric("Event C 🟢", f"s = {tau_C:.3f}", delta="Lightlike", delta_color="off")

st.caption("💡 **τ** (tau) = proper time for timelike intervals, **σ** (sigma) = proper distance for spacelike intervals. These values are invariant across all frames!")

st.markdown("---")

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