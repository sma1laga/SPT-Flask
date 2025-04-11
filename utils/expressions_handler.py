import tkinter as tk
from tkinter import ttk
import webbrowser

class ExpressionHandler:
    def __init__(self, page):
        self.page = page
        self.expr_tokens = []

    def create_functions_and_operators_section(self, parent=None):
        """
        Create a single, combined Quick Functions & Operators section,
        plus a PDF button. Removed the extra 'Clear Expression' button.
        """
        if parent is None:
            parent = self.page

        # Create a controls frame with minimal padding
        controls_frame = tk.Frame(parent, bg="#f0f4f8")
        controls_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        # --- Combined Quick Functions & Operators Section ---
        combined_frame = tk.LabelFrame(
            controls_frame,
            text="Quick Functions & Operators",
            font=("Helvetica", 9, "bold"),
            bg="#f0f4f8",
            padx=4, pady=4
        )
        combined_frame.grid(row=0, column=0, padx=2, pady=2, sticky="nw")

        # Quick Functions (arranged in a grid)
        quick_functions = [
            ("Rect", "rect(t)", "Rectangular pulse"),
            ("Tri", "tri(t)", "Triangular function"),
            ("Sin", "sin(t)", "Sine function"),
            ("Step", "step(t)", "Step function"),
            ("Cos", "cos(t)", "Cosine function"),
            ("Sign", "sign(t)", "Signum function"),
            ("Delta", "delta(t)", "Approx. delta"),
            ("e^(iwt)", "exp_iwt(t)", "Complex exponential"),
            ("Exp", "exp(t)", "Exponential function"),  # <-- New button
            ("1/t", "inv_t(t)", "Reciprocal function"),
            ("Si", "si(t)", "Sinc function"),
            ("Si^2", "si(t)**2", "Squared sinc")
        ]
        cols = 4
        for i, (name, func, tooltip) in enumerate(quick_functions):
            btn = ttk.Button(combined_frame, text=name, command=lambda f=func: self.add_operand(f))
            btn.grid(row=i // cols, column=i % cols, padx=2, pady=2, sticky="nsew")
            self._add_tooltip(btn, tooltip)

        # Operators (place them below quick functions in the same frame)
        operators = ["+", "-", "*", "/"]
        op_frame = tk.Frame(combined_frame, bg="#f0f4f8")
        op_frame.grid(row=(len(quick_functions) // cols) + 1, column=0, columnspan=cols, pady=5, sticky="w")
        for j, op in enumerate(operators):
            btn = ttk.Button(op_frame, text=op, width=3, command=lambda o=op: self.add_operator(o))
            btn.pack(side="left", padx=3)
            self._add_tooltip(btn, f"Insert '{op}' operator")

        # --- PDF Button (renamed) ---

    def _add_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget, bg="#f0f4f8", padx=2, pady=2)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        label = tk.Label(tooltip, text=text, font=("Helvetica", 8), bg="#ffffe0", relief="solid", borderwidth=1)
        label.pack()

        def on_enter(event):
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 20
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def on_leave(event):
            tooltip.withdraw()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def add_operand(self, operand):
        """Insert a function (operand) into the appropriate entry widget."""
        if hasattr(self.page, "func1_entry"):
            if getattr(self.page, "last_focused_input", "func1") == "func2":
                self.page.func2_entry.insert("end", operand)
            else:
                self.page.func1_entry.insert("end", operand)
        else:
            self.page.func_entry.insert("end", operand)

    def add_operator(self, operator):
        """Insert an operator into the appropriate entry widget."""
        if hasattr(self.page, "func1_entry"):
            if getattr(self.page, "last_focused_input", "func1") == "func2":
                self.page.func2_entry.insert("end", operator)
            else:
                self.page.func1_entry.insert("end", operator)
        else:
            self.page.func_entry.insert("end", operator)

    def open_definitions_pdf(self):
        """Open the Definitions PDF file."""
        pdf_path = r".\definitions.pdf"
        try:
            webbrowser.open_new(pdf_path)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not open PDF: {e}")
