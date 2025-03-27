import tkinter as tk
from tkinter import messagebox


def submit():
    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
        c = float(entry_c.get())
        d = float(entry_d.get())
        D_u = float(entry_Du.get())
        D_v = float(entry_Dv.get())

        # Вивід значень або передача в модель
        print(f"a = {a}, b = {b}, c = {c}, d = {d}, D_u = {D_u}, D_v = {D_v}")
        messagebox.showinfo("Параметри збережено", "Коефіцієнти успішно задані!")

    except ValueError:
        messagebox.showerror("Помилка", "Введіть правильні числові значення.")


# Створення вікна
root = tk.Tk()
root.title("Параметри моделі Lotka-Volterra")

# Створення полів вводу
labels = ["Коефіцієнт a:", "Коефіцієнт b:", "Коефіцієнт c:", "Коефіцієнт d:", "Коефіцієнт D_u:", "Коефіцієнт D_v:"]
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0, pady=5, padx=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, pady=5, padx=5)
    entries.append(entry)

entry_a, entry_b, entry_c, entry_d, entry_Du, entry_Dv = entries

# Кнопка підтвердження
submit_button = tk.Button(root, text="Підтвердити", command=submit)
submit_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

root.mainloop()
