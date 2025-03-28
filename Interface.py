import tkinter as tk

class Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Set Parameters")
        self.root.geometry("400x300+500+200")

        # Створення змінних
        self.a = tk.DoubleVar(value=1.1)
        self.b = tk.DoubleVar(value=0.4)
        self.c = tk.DoubleVar(value=0.4)
        self.d = tk.DoubleVar(value=0.1)
        self.D_u = tk.DoubleVar(value=0.1)
        self.D_v = tk.DoubleVar(value=0.1)

        # Створення віджетів
        self.create_widgets()

        # Запуск головного циклу
        self.root.mainloop()

    def create_widgets(self):
        # Створення міток і полів введення
        params = ['a', 'b', 'c', 'd', 'D_u', 'D_v']
        variables = [self.a, self.b, self.c, self.d, self.D_u, self.D_v]

        for i, (param, var) in enumerate(zip(params, variables)):
            tk.Label(self.root, text=f"{param}:").grid(row=i, column=0, sticky="w", padx=10, pady=5)
            entry = tk.Entry(self.root, textvariable=var)
            entry.grid(row=i, column=1, sticky="ew", padx=10, pady=5)

        # Кнопка підтвердження
        submit_btn = tk.Button(self.root, text="Submit", command=self.submit)
        submit_btn.grid(row=len(params), column=0, columnspan=2, pady=10)

        # Адаптивність
        self.root.grid_rowconfigure(list(range(len(params))), weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def submit(self):
        # Отримання значень
        self.params = {
            'a': self.a.get(),
            'b': self.b.get(),
            'c': self.c.get(),
            'd': self.d.get(),
            'D_u': self.D_u.get(),
            'D_v': self.D_v.get(),
        }
        self.root.destroy()

    def run(self):
        return self.params


