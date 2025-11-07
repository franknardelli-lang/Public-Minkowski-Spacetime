def enable_textbox_autoselect(fig, box):
    """Highlight (select) all text when clicking into a TextBox — works in all Matplotlib versions."""
    def on_click(event):
        if event.inaxes == box.ax:
            fig.canvas.flush_events()
            try:
                box._select_text(0, len(box.text))
            except AttributeError:
                box.cursor_index = len(box.text)
                box._rendercursor()
    fig.canvas.mpl_connect('button_press_event', on_click)


def enable_tab_navigation(fig, boxes):
    """Allow Tab key to move between TextBoxes."""
    def on_key(event):
        if event.key == 'tab':
            for i, b in enumerate(boxes):
                if event.inaxes == b.ax:
                    next_box = boxes[(i + 1) % len(boxes)]
                    try:
                        next_box._select_text(0, len(next_box.text))
                    except AttributeError:
                        next_box.cursor_index = len(next_box.text)
                        next_box._rendercursor()
                    fig.canvas.draw_idle()
                    break
    fig.canvas.mpl_connect('key_press_event', on_key)
