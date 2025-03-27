import tkinter as tk
from tkinter import messagebox

class Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Model Coefficient Input")

        self.coefficients = {}
        self.create_widgets()

    def create_widgets(self):
        labels = ["a", "b", "c", "d", "D_u", "D_v"]

        for i, label in enumerate(labels):
            tk.Label(self.root, text=f"{label}: ").grid(row=i, column=0, pady=5, padx=5)
            entry = tk.Entry(self.root)
            entry.grid(row=i, column=1, pady=5, padx=5)
            self.coefficients[label] = entry

        tk.Button(self.root, text="Confirm", command=self.get_values).grid(row=len(labels), column=0, columnspan=2, pady=10)

    def get_values(self):
        try:
            values = {key: float(entry.get()) for key, entry in self.coefficients.items()}
            self.root.destroy()  # Close the window after successful input
            self.values = values
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all coefficients.")

    def run(self):
        self.root.mainloop()
        return self.values


