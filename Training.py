import numpy as np
import matplotlib.pyplot as plt

# Завантаження лише загального loss
loss = np.loadtxt("Test7/loss.dat")

plt.figure(figsize=(10, 6))
plt.plot(loss[:, 0], loss[:, 1], label="Втрати")
plt.xlabel("Ітерації")
plt.ylabel("Втрати")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.title("Загальні втрати")
plt.tight_layout()
plt.savefig("loss_restored.png", dpi=300)
plt.show()

