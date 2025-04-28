import tkinter as tk
from tkinter import messagebox

class Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Задайте параметри")
        self.root.geometry("400x300+500+200")

        # Створення змінних
        self.T = tk.IntVar(value=20)        # час спостереження
        self.S = tk.IntVar(value=10)         # сторона області спостереження
        self.ru = tk.DoubleVar(value=0.2)    # коефіцієнт безумовного росту хижака
        self.rv = tk.DoubleVar(value=-0.2)   # коефіцієнт безумовного росту жертви
        self.auv = tk.DoubleVar(value=0.2)   # коефіцієнт впливу хижака на кількість жертви
        self.avu = tk.DoubleVar(value=-0.2)  # коефіцієнт впливу жертви на кількість хижаків
        self.auu = tk.DoubleVar(value=0.0)   # коефіцієнт ефекту надлишкової популяції жертви
        self.avv = tk.DoubleVar(value=0.0)   # коефіцієнт ефекту надлишкової популяції хижаків
        self.D_u = tk.DoubleVar(value=0.1)  # просторова дифузія жертви
        self.D_v = tk.DoubleVar(value=0.1)  # просторова дифузія хижаків

        # Створення віджетів
        self.create_widgets()

        # Запуск головного циклу
        self.root.mainloop()

    def create_widgets(self):
        # Створення міток і полів введення
        params = [
            'Час спостереження',
            'Сторона області спостереження',
            'Коефіцієнт безумовного росту хижака',
            'Коефіцієнт безумовного росту жертви',
            'Коефіцієнт впливу хижака на кількість жертви',
            'Коефіцієнт впливу жертви на кількість хижаків',
            'Коефіцієнт ефекту надлишкової популяції жертви',
            'Коефіцієнт ефекту надлишкової популяції хижаків',
            'Просторова дифузія жертви',
            'Просторова дифузія хижаків'
        ]
        variables = [self.T, self.S, self.ru, self.rv, self.auv, self.avu, self.auu, self.avv, self.D_u, self.D_v]

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
        # Перевірка коректності введення
        try:
            T_value = self.T.get()
            S_value = self.S.get()
            if T_value <= 0 or S_value <= 0:
                raise ValueError("T та S повинні бути більше 0")
        except:
            tk.messagebox.showerror("Помилка",
                                    "Час спостереження та сторона області повинні бути більше 0")
            return

        # Отримання значень
        try:
            self.params = {
                'T': self.T.get(),
                'S': self.S.get(),
                'ru': self.ru.get(),
                'rv': self.rv.get(),
                'auv': self.auv.get(),
                'avu': self.avu.get(),
                'auu': self.auu.get(),
                'avv': self.avv.get(),
                'D_u': self.D_u.get(),
                'D_v': self.D_v.get(),
            }
            self.root.destroy()
        except:
            tk.messagebox.showerror("Помилка",
                                    "Введіть числові значення")

    def run(self):
        return self.params


