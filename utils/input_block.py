import tkinter as tk
from tkinter import ttk

def open_input_block_editor(block):
    """
    Opens a small editor window for a block.
    
    Parameters:
      block: the block object (e.g. an instance of Node with type "Block")
             that must support update_text(new_text) and can have an attribute 'value'.
    """
    editor = tk.Toplevel()
    editor.title("Edit Block")
    editor.geometry("300x200")
    editor.resizable(False, False)
    
    # Label and input field.
    label = ttk.Label(editor, text="Enter function for block:")
    label.pack(pady=(10,5))
    
    input_field = ttk.Entry(editor, width=30)
    input_field.pack(pady=5)
    
    # Frame for function buttons.
    btn_frame = ttk.Frame(editor)
    btn_frame.pack(pady=5)
    
    # List of function buttons.
    functions = ["d/dt", "H", "Re", "Im", "h_BP", "h_LP"]
    
    def add_func(func):
        """Append the given function text to the input field."""
        current = input_field.get()
        # Append with a space for readability.
        new_text = current + " " + func if current else func
        input_field.delete(0, tk.END)
        input_field.insert(0, new_text)
    
    for func in functions:
        btn = ttk.Button(btn_frame, text=func, command=lambda f=func: add_func(f))
        btn.pack(side="left", padx=2)
    
    # OK and Cancel buttons.
    def on_ok():
        value = input_field.get().strip()
        if value:
            # Store the value in the block and update its label.
            block.value = value
            block.update_text(value)
        editor.destroy()
    
    def on_cancel():
        editor.destroy()
    
    btn_ok = ttk.Button(editor, text="OK", command=on_ok)
    btn_ok.pack(pady=(10,2))
    btn_cancel = ttk.Button(editor, text="Cancel", command=on_cancel)
    btn_cancel.pack()

