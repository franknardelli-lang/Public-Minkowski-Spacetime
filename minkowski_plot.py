import numpy as np

def draw_static_elements(ax):
    """Draw spacetime hyperbolae on the Minkowski diagram."""
    s_vals = [1, 2, 3, 4, 5]
    for s in s_vals:
        x = np.linspace(s, 10, 400)
        t = np.sqrt(x**2 - s**2)
        ax.plot( x,  t, 'b-', alpha=0.2)
        ax.plot(-x,  t, 'b-', alpha=0.2)
        ax.plot( x, -t, 'b-', alpha=0.2)
        ax.plot(-x, -t, 'b-', alpha=0.2)
    for s in s_vals:
        x = np.linspace(-10, 10, 400)
        t = np.sqrt(x**2 + s**2)
        ax.plot(x,  t, 'r-', alpha=0.2)
        ax.plot(x, -t, 'r-', alpha=0.2)
