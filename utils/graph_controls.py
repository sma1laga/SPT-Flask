from tkinter import filedialog
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


fullscreen_window = None  # Global variable to manage fullscreen state

def toggle_fullscreen_for_plot(plot_frame, figure):
    """Toggle a new fullscreen window for displaying the plots."""
    global fullscreen_window

    if fullscreen_window:
        # Close the fullscreen window if it exists
        fullscreen_window.destroy()
        fullscreen_window = None
    else:
        # Create a new fullscreen window
        fullscreen_window = tk.Toplevel(plot_frame)
        fullscreen_window.attributes('-fullscreen', True)
        fullscreen_window.configure(bg="#ffffff")

        # Add a close button
        close_btn = tk.Button(
            fullscreen_window, text="Exit Fullscreen", command=lambda: toggle_fullscreen_for_plot(plot_frame, figure),
            font=("Helvetica", 12, "bold"), bg="#007acc", fg="white"
        )
        close_btn.pack(pady=10)

        # Display the plot in the fullscreen window
        canvas = FigureCanvasTkAgg(figure, master=fullscreen_window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
def save_plot(figure):
    """Save the current plot as an image file."""
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        figure.savefig(file_path)

def show_toolbar(canvas, parent):
    """Show the Matplotlib toolbar for zooming and panning."""
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")