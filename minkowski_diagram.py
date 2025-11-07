import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from functools import partial
from minkowski_math import lorentz_transform, relative_velocity
from minkowski_plot import draw_static_elements
from minkowski_widgets import enable_tab_navigation


class MinkowskiDiagram:
    def __init__(self):
        # --- Figure setup ---
        self.fig = plt.figure(figsize=(14, 8))
        left, bottom, width, height = 0.06, 0.15, 0.55, 0.8
        self.ax = self.fig.add_axes([left, bottom, width, height])
        plt.subplots_adjust(left=0.1, bottom=0.3, top=0.85, right=0.83)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Space (x)')
        self.ax.set_ylabel('Time (t)')
        self.ax.set_title('Interactive Minkowski Spacetime Diagram')
        self.ax.axhline(0, color='black')
        self.ax.axvline(0, color='black')

        # --- Events and velocities ---
        self.eventA = np.array([0.0, 0.0])
        self.eventB = np.array([2.0, 4.0])
        self.eventC = np.array([3.0, 1.0])
        self.v_b_value, self.v_c_value = 0.6, -0.4
        self.current_frame = 'A'

        # --- Points and storage ---
        self.earth_dot, = self.ax.plot([], [], 'ro', label="Event A")
        self.b_dot,     = self.ax.plot([], [], 'bo', label="Event B")
        self.c_dot,     = self.ax.plot([], [], 'go', label="Event C")
        self.dynamic_lines, self.texts = [], []

        # --- Sliders ---
        axcolor = 'lightgoldenrodyellow'
        self.ax_vel_b = plt.axes([0.1, 0.065, 0.35, 0.04], facecolor=axcolor)
        self.vel_slider_b = Slider(self.ax_vel_b, "B v/c", -0.99, 0.99, valinit=self.v_b_value)
        self.vel_slider_b.on_changed(self.update_plot)

        self.ax_vel_c = plt.axes([0.1, 0.03, 0.35, 0.04], facecolor=axcolor)
        self.vel_slider_c = Slider(self.ax_vel_c, "C v/c", -0.99, 0.99, valinit=self.v_c_value)
        self.vel_slider_c.on_changed(self.update_plot)

        # --- Frame buttons ---
        vertical_gap = 0.01
        control_width_btn = 0.07
        control_width_box = 0.075
        control_height = 0.04
        btn_x = left + width + 0.02
        btn_y_start = 0.85 - control_height

        self._make_frame_button(btn_x, btn_y_start, control_width_btn, control_height, 'A')
        self._make_frame_button(btn_x, btn_y_start - (control_height + vertical_gap),
                                control_width_btn, control_height, 'B')
        self._make_frame_button(btn_x, btn_y_start - 2 * (control_height + vertical_gap),
                                control_width_btn, control_height, 'C')

        # --- Coordinate boxes ---
        gap_horiz = 0.04
        coord_x_start = btn_x + control_width_btn + gap_horiz
        coord_y_start = btn_y_start
        self._make_coordinate_boxes(coord_x_start, coord_y_start,
                                    control_width_box, control_height, vertical_gap)

        # --- Velocity input boxes ---
        self.text_box_vb = self._make_textbox(0.53, 0.065, 0.06, 0.04,
                                              r"$v_{\mathrm{B}}$", self.v_b_value, self.update_v_b_from_text)
        self.text_box_vc = self._make_textbox(0.53, 0.03, 0.06, 0.04,
                                              r"$v_{\mathrm{C}}$", self.v_c_value, self.update_v_c_from_text)

        # --- Spacetime interval legend ---
        text_x = 0.60
        text_y_start = 0.36
        text_gap = 0.04
        self.text_formula = self.fig.text(
            text_x, text_y_start + 0.05,
            r"$S^2 = (c \Delta t)^2 - (\Delta x)^2$",
            fontsize=12, ha='left'
        )
        self.text_s_ab = self.fig.text(text_x, text_y_start, '', fontsize=10, ha='left')
        self.text_s_ac = self.fig.text(text_x, text_y_start - text_gap, '', fontsize=10, ha='left')
        self.text_s_bc = self.fig.text(text_x, text_y_start - 2 * text_gap, '', fontsize=10, ha='left')

        # --- Legend (Color Key) ---
        self.text_x, self.legend_y_start = text_x, text_y_start + 0.15
        self._make_color_key()

        # --- Textbox usability helpers ---
        self.all_textboxes = [
            self.text_box_tA, self.text_box_xA,
            self.text_box_tB, self.text_box_xB,
            self.text_box_tC, self.text_box_xC,
            self.text_box_vb, self.text_box_vc
        ]
        for box in self.all_textboxes:
            self.enable_textbox_autoclear(box)

        enable_tab_navigation(self.fig, self.all_textboxes)

        # --- Mouse handlers ---
        self.dragging = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # --- Draw base elements ---
        draw_static_elements(self.ax)
        self.update_plot(None)
        plt.show()

            # -------------------------------------------------------
    # --- Click-to-clear behavior for TextBoxes ---
    def enable_textbox_autoclear(self, box):
        """Clears textbox content on click for fast editing."""
        def on_click(event):
            if event.inaxes == box.ax:
                box.set_val("")  # clear the box instantly for new entry
                self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_connect("button_press_event", on_click)

    # -------------------------------------------------------
    # --- Frame control and appearance ---
    def _make_frame_button(self, x, y, w, h, frame):
        """Create a frame selection button with persistent highlight."""
        ax_btn = plt.axes([x, y, w, h])
        btn = Button(ax_btn, f"Frame {frame}")
        btn.on_clicked(partial(self._on_frame_button_click, frame))
        if not hasattr(self, "_frame_buttons"):
            self._frame_buttons = {}
        self._frame_buttons[frame] = (btn, ax_btn)
        return btn

    def _on_frame_button_click(self, frame, event=None):
        """Triggered when a frame button is clicked."""
        self.change_frame(frame)

    def change_frame(self, frame):
        """Switch between reference frames and update plot."""
        self.current_frame = frame
        self.update_plot(None)
        self._highlight_active_frame(frame)
        self._update_frame_info_color(frame)
        self._update_plot_title(frame)

    def _highlight_active_frame(self, active):
        color_map = {'A': '#ffcccc', 'B': '#b3d9ff', 'C': '#c8facc'}
        default_color = (0.92, 0.92, 0.92, 1.0)
        for f, (btn, ax_btn) in self._frame_buttons.items():
            if f == active:
                color = color_map.get(f, '#ffeb99')
                ax_btn.set_facecolor(color)
                btn.color = btn.hovercolor = color
            else:
                ax_btn.set_facecolor(default_color)
                btn.color = btn.hovercolor = default_color
        self.fig.canvas.draw_idle()

    def _update_frame_info_color(self, frame):
        color_map = {'A': 'r', 'B': 'b', 'C': 'g'}
        color = color_map.get(frame, 'black')
        if hasattr(self, "texts") and self.texts:
            info_text = self.texts[-1]
            if info_text.get_text().startswith("Frame"):
                info_text.set_color(color)
                self.fig.canvas.draw_idle()

    def _update_plot_title(self, frame):
        color_map = {'A': 'red', 'B': 'blue', 'C': 'green'}
        color = color_map.get(frame, 'black')
        title = f"Interactive Minkowski Spacetime Diagram — Frame {frame}"
        self.ax.set_title(title, color=color, fontsize=14, weight='bold')
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------
    # --- Widget construction ---
    def _make_textbox(self, x, y, w, h, label, init, callback, fontsize=12):
        ax_box = plt.axes([x, y, w, h])
        box = TextBox(ax_box, label, initial=f"{init:.2f}", label_pad=0.02)
        box.label.set_fontsize(fontsize)
        box.on_submit(callback)
        return box

    def _make_coordinate_boxes(self, x_start, y_start, w, h, gap):
        def add_box(x, y, label, init, callback, fontsize=12):
            ax_box = plt.axes([x, y, w, h])
            box = TextBox(ax_box, label, initial=f"{init:.2f}", label_pad=0.02)
            box.label.set_fontsize(fontsize)
            box.on_submit(callback)
            return box

        self.text_box_tA = add_box(x_start, y_start, r"$t_{\mathrm{A}}$", self.eventA[0], self.update_eventA_t)
        self.text_box_xA = add_box(x_start + w + 0.04, y_start, r"$x_{\mathrm{A}}$", self.eventA[1], self.update_eventA_x)
        self.text_box_tB = add_box(x_start, y_start - (h + gap), r"$t_{\mathrm{B}}$", self.eventB[0], self.update_eventB_t)
        self.text_box_xB = add_box(x_start + w + 0.04, y_start - (h + gap), r"$x_{\mathrm{B}}$", self.eventB[1], self.update_eventB_x)
        self.text_box_tC = add_box(x_start, y_start - 2*(h + gap), r"$t_{\mathrm{C}}$", self.eventC[0], self.update_eventC_t)
        self.text_box_xC = add_box(x_start + w + 0.04, y_start - 2*(h + gap), r"$x_{\mathrm{C}}$", self.eventC[1], self.update_eventC_x)

    def _make_color_key(self):
        text_x, y = self.text_x, self.legend_y_start
        self.fig.text(text_x, y + 0.08, "Diagram Color Key:", fontsize=11, weight='bold', ha='left')
        self.fig.text(text_x, y + 0.05, "— Magenta: Time Axes", color='m', fontsize=10, ha='left')
        self.fig.text(text_x, y + 0.02, "— Cyan: Simultaneity Lines", color='c', fontsize=10, ha='left')
        self.fig.text(text_x, y - 0.01, "— Gray: Worldlines", color='gray', fontsize=10, ha='left')
        self.fig.text(text_x, y - 0.04, "— Dashed Lines: Light Cones", color='r', fontsize=10, ha='left')

    # -------------------------------------------------------
    # --- Event update logic ---
    def update_eventA_t(self, text): self._safe_update(self.eventA, 0, text)
    def update_eventA_x(self, text): self._safe_update(self.eventA, 1, text)
    def update_eventB_t(self, text): self._safe_update(self.eventB, 0, text)
    def update_eventB_x(self, text): self._safe_update(self.eventB, 1, text)
    def update_eventC_t(self, text): self._safe_update(self.eventC, 0, text)
    def update_eventC_x(self, text): self._safe_update(self.eventC, 1, text)

    def _safe_update(self, event, idx, text):
        try:
            event[idx] = float(text)
            self.update_plot(None)
        except ValueError:
            pass

    def update_v_b_from_text(self, text):
        try:
            val = np.clip(float(text), -0.99, 0.99)
            self.v_b_value = val
            self.vel_slider_b.set_val(val)
            self.update_plot(None)
        except ValueError:
            pass

    def update_v_c_from_text(self, text):
        try:
            val = np.clip(float(text), -0.99, 0.99)
            self.v_c_value = val
            self.vel_slider_c.set_val(val)
            self.update_plot(None)
        except ValueError:
            pass

    # -------------------------------------------------------
    # --- Mouse drag event handling ---
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.earth_dot.contains(event)[0]:
            self.dragging = 'A'
        elif self.b_dot.contains(event)[0]:
            self.dragging = 'B'
        elif self.c_dot.contains(event)[0]:
            self.dragging = 'C'

    def on_release(self, event):
        self.dragging = None

    def on_motion(self, event):
        if self.dragging is None or event.inaxes != self.ax:
            return
        t, x = event.ydata, event.xdata
        if self.dragging == 'A':
            target = self.eventA
        elif self.dragging == 'B':
            target = self.eventB
        else:
            target = self.eventC
        target[:] = [t, x]
        self.update_plot(None)

    # -------------------------------------------------------
    # --- Plot update logic ---
    def update_plot(self, val):
        for ln in self.dynamic_lines:
            ln.remove()
        self.dynamic_lines.clear()
        for tx in self.texts:
            tx.remove()
        self.texts.clear()

        v_b, v_c = self.vel_slider_b.val, self.vel_slider_c.val
        self.v_b_value, self.v_c_value = v_b, v_c
        self.text_box_vb.set_val(f"{v_b:.2f}")
        self.text_box_vc.set_val(f"{v_c:.2f}")

        v_frame = 0 if self.current_frame == 'A' else (v_b if self.current_frame == 'B' else v_c)
        pts = np.array([self.eventA, self.eventB, self.eventC])
        Atp = lorentz_transform(pts, v_frame)
        (tA, xA), (tB, xB), (tC, xC) = Atp

        # --- Worldlines ---
        t_world = np.linspace(-10, 10, 200)
        for (v, x0, t0) in [(v_b, xB, tB), (v_c, xC, tC)]:
            x_world = v * t_world + (x0 - v * t0)
            self.dynamic_lines.append(self.ax.plot(x_world, t_world, color='gray', alpha=0.5)[0])

        # --- Simultaneity & Time axes ---
        vb_rel, vc_rel = relative_velocity(v_b, v_frame), relative_velocity(v_c, v_frame)
        X = np.linspace(-10, 10, 300)
        for (x0, t0, vrel, color) in [(xB, tB, vb_rel, 'b'), (xC, tC, vc_rel, 'g')]:
            t_sim = t0 + vrel * (X - x0)
            self.dynamic_lines.append(self.ax.plot(X, t_sim, 'c-', lw=2)[0])
            slope = np.inf if abs(vrel) < 1e-3 else 1 / vrel
            if np.isinf(slope):
                x_line, t_line = np.full_like(X, x0), np.linspace(-10, 10, 300)
            else:
                x_line, t_line = X, t0 + slope * (X - x0)
            self.dynamic_lines.append(self.ax.plot(x_line, t_line, 'm-', lw=2)[0])

        # --- Light Cones (restored) ---
        t_range = np.linspace(-10, 10, 400)
        for (x0, t0, color) in [(xA, tA, 'r'), (xB, tB, 'gray'), (xC, tC, 'gray')]:
            t_plus = t0 + (t_range - x0)
            t_minus = t0 - (t_range - x0)
            self.dynamic_lines.append(self.ax.plot(t_range, t_plus, linestyle='--', color=color, alpha=0.3, lw=1.2)[0])
            self.dynamic_lines.append(self.ax.plot(t_range, t_minus, linestyle='--', color=color, alpha=0.3, lw=1.2)[0])

        # --- Update event points ---
        self.earth_dot.set_data([xA], [tA])
        self.b_dot.set_data([xB], [tB])
        self.c_dot.set_data([xC], [tC])

        # --- Event labels ---
        for (x, t, name, color) in [(xA, tA, 'A', 'r'), (xB, tB, 'B', 'b'), (xC, tC, 'C', 'g')]:
            self.texts.append(self.ax.text(
                x + 0.3*np.sign(x + 0.01), t + 0.3*np.sign(t + 0.01),
                name, color=color, weight='bold', fontsize=11,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
            ))

        # --- Frame info text ---
        gamma = 1 / np.sqrt(1 - v_frame**2)
        self.texts.append(self.ax.text(
            0.05, 0.05, f"Frame {self.current_frame}: v={v_frame:.2f}c, γ={gamma:.3f}",
            transform=self.ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8)
        ))

        # --- Update legends ---
        self.update_spacetime_interval_text()
        self._update_lorentz_info(v_frame)
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------
    def update_spacetime_interval_text(self):
        c = 1
        v_frame = 0 if self.current_frame == 'A' else (
            self.v_b_value if self.current_frame == 'B' else self.v_c_value)
        Apts = np.array([self.eventA, self.eventB, self.eventC])
        Atp = lorentz_transform(Apts, v_frame)
        (tA, xA), (tB, xB), (tC, xC) = Atp

        def S2(a, b): return (c*(b[0]-a[0]))**2 - (b[1]-a[1])**2
        S2_AB, S2_AC, S2_BC = S2((tA,xA),(tB,xB)), S2((tA,xA),(tC,xC)), S2((tB,xB),(tC,xC))

        def interval(S2):
            if S2 > 0:  return f"timelike (S = {np.sqrt(S2):.3f})"
            elif S2 < 0: return f"spacelike (S = {np.sqrt(-S2):.3f}i)"
            else:        return "lightlike (S = 0)"

        self.text_s_ab.set_text(fr"$S_{{AB}}^2 = {S2_AB:.3f},\; {interval(S2_AB)}$")
        self.text_s_ac.set_text(fr"$S_{{AC}}^2 = {S2_AC:.3f},\; {interval(S2_AC)}$")
        self.text_s_bc.set_text(fr"$S_{{BC}}^2 = {S2_BC:.3f},\; {interval(S2_BC)}$")

    def _update_lorentz_info(self, v_frame):
        if hasattr(self, "_lorentz_texts"):
            for t in self._lorentz_texts:
                t.remove()

        text_x = self.text_x
        lorentz_y_start = 0.21
        gamma = 1 / np.sqrt(1 - v_frame**2)
        frame = self.current_frame

        t1 = self.fig.text(text_x, lorentz_y_start,
                           f"Lorentz Transform (Frame {frame}):",
                           fontsize=11, weight='bold', ha='left')
        t2 = self.fig.text(text_x, lorentz_y_start - 0.03,
                           r"$t' = \gamma (t - v x)$,   $x' = \gamma (x - v t)$",
                           fontsize=11, ha='left')
        t3 = self.fig.text(text_x, lorentz_y_start - 0.06,
                           fr"$v = {v_frame:.2f}c$,   $\gamma = {gamma:.3f}$",
                           fontsize=10, ha='left')
        self._lorentz_texts = [t1, t2, t3]
        self.fig.canvas.draw_idle()


# -------------------------------------------------------
# Allow running directly or via main.py
if __name__ == "__main__":
    MinkowskiDiagram()
