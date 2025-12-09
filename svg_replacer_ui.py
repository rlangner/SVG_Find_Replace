import tkinter as tk
from tkinter import filedialog, ttk
import sys
import os
from svg_replacer import replace_groups_in_svg


class TextRedirector:
    """Redirects stdout to a Tkinter Text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SVG Group Replacer")
        self.geometry("850x600")

        # ----- INPUT FILE SECTION -----
        frame_input = ttk.LabelFrame(self, text="Input SVG File")
        frame_input.pack(fill="x", padx=10, pady=5)

        self.input_path = tk.StringVar()
        ttk.Entry(frame_input, textvariable=self.input_path, width=90).pack(side="right", padx=5)
        ttk.Button(frame_input, text="Browse", command=self.pick_file_input).pack(side="right")

        # ----- LOOKUP FILE SECTION -----
        frame_lookup = ttk.LabelFrame(self, text="Lookup SVG File")
        frame_lookup.pack(fill="x", padx=10, pady=5)

        self.lookup_path = tk.StringVar()
        ttk.Entry(frame_lookup, textvariable=self.lookup_path, width=90).pack(side="right", padx=5)
        ttk.Button(frame_lookup, text="Browse", command=self.pick_file_lookup).pack(side="right")

        # ----- OUTPUT FILE SECTION -----
        frame_output = ttk.LabelFrame(self, text="Output SVG File")
        frame_output.pack(fill="x", padx=10, pady=5)

        self.output_path = tk.StringVar()
        ttk.Entry(frame_output, textvariable=self.output_path, width=90).pack(side="right", padx=5)
        ttk.Button(frame_output, text="Browse", command=self.pick_file_output).pack(side="right")

        # ----- RUN BUTTON -----
        ttk.Button(self, text="Run Find and Replace", command=self.run_find_replace).pack(
            padx=10, pady=10
        )

        # ----- TEXT OUTPUT -----
        self.text_box = tk.Text(self, height=18)
        self.text_box.pack(fill="both", expand=True, padx=10, pady=10)

        # Redirect stdout to text box
        sys.stdout = TextRedirector(self.text_box)

    # ------------------------------------------------------------------

    def pick_file_input(self):
        file_path = filedialog.askopenfilename(
            title="Select SVG File",
            filetypes=[("SVG Files", "*.svg")]
        )
        if file_path:
            self.input_path.set(file_path)
            # auto-fill output path
            out_path = os.path.join(os.path.dirname(file_path), "output.svg")
            self.output_path.set(out_path)

    def pick_file_lookup(self):
        file_path = filedialog.askopenfilename(
            title="Select SVG File",
            filetypes=[("SVG Files", "*.svg")]
        )
        if file_path:
            self.lookup_path.set(file_path)

    def pick_file_output(self):
        file_path = filedialog.asksaveasfilename(
            title="Select Output SVG File",
            defaultextension=".svg",
            filetypes=[("SVG Files", "*.svg")]
        )
        if file_path:
            self.output_path.set(file_path)

    # ------------------------------------------------------------------

    def run_find_replace(self):
        input_file = self.input_path.get()
        lookup_file = self.lookup_path.get()
        output_file = self.output_path.get()

        print(f"Input: {input_file}\nLookup: {lookup_file}")
        replace_groups_in_svg(input_file, lookup_file, output_file)
        print("\nDone.\n")


# ----------------------------------------------------------------------

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
