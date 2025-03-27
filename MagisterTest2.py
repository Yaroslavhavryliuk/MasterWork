import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import tf

# Параметри моделі
D_u = 0.1
D_v = 0.1
a = 1.1
b = 0.04
c = 0.4
d = 0.01


def uniform_with_noise_distribution(x, total_count=1000, noise_scale=0.05):
    """
    Рівномірний розподіл з невеликим шумом.
    x - координати
    total_count - загальна кількість особин
    noise_scale - амплітуда шуму
    """
    num_points = x.shape[0]
    # Початковий рівномірний розподіл
    base_value = total_count / num_points
    noise = np.random.normal(loc=0.0, scale=base_value * noise_scale, size=(num_points, 1))
    distribution = base_value + noise
    return np.maximum(distribution, 0)  # Уникаємо негативних значень

# Функція для кластерного розподілу з початковою кількістю
def clustered_distribution(x, centers, total_count=1000, intensity=5):
    """
    x - координати
    centers - центри кластерів
    total_count - бажана загальна чисельність
    intensity - концентрація навколо центрів
    """
    # Ініціалізація порожнього розподілу
    distribution = np.zeros((x.shape[0], 1))

    # Додавання значень від кожного кластера
    for center in centers:
        distance = np.linalg.norm(x - center, axis=1, keepdims=True)
        distribution += np.exp(-intensity * distance)

    # Нормалізація, щоб загальна сума дорівнювала total_count
    distribution_sum = np.sum(distribution)
    if distribution_sum > 0:
        distribution = (distribution / distribution_sum) * total_count

    return distribution


def exponential_distribution(x, center, total_count=1000, scale=2):
    """
    center - центр екологічної ніші
    total_count - бажана загальна чисельність
    scale - коефіцієнт спадання
    """
    # Відстань до центру
    distance = np.linalg.norm(x[:, :2] - center, axis=1, keepdims=True)
    # Початковий експоненційний розподіл
    distribution = np.exp(-distance / scale)

    # Нормалізація під задану загальну кількість
    distribution_sum = np.sum(distribution)
    if distribution_sum > 0:
        distribution = (distribution / distribution_sum) * total_count

    return distribution

# Приклади центрів кластерів
wolf_centers = np.array([[2, 2], [-3, -4], [4, -2]])  # Локації вовків
hare_center = np.array([0, 0])  # Локація центру (наприклад, ліс)

# Початкові кількості
initial_wolf_count = 100    # Наприклад, 100 вовків
initial_hare_count = 5000   # Наприклад, 5000 зайців

# Початкові умови
def ic_func_v(x):
    return uniform_with_noise_distribution(x, initial_wolf_count, noise_scale=0.2)  # Більший шум для хижаків

def ic_func_u(x):
    return uniform_with_noise_distribution(x, initial_hare_count, noise_scale=0.05)  # Менший шум для жертв


def pde(x, y):
    u, v = y[:, 0:1], y[:, 1:2]
    du_t = dde.grad.jacobian(u, x, i=0, j=2)
    dv_t = dde.grad.jacobian(v, x, i=0, j=2)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    dv_xx = dde.grad.hessian(v, x, i=0, j=0)
    dv_yy = dde.grad.hessian(v, x, i=1, j=1)
    eq_u = du_t - D_u * (du_xx + du_yy) - u * (a - b * v)
    eq_v = dv_t - D_v * (dv_xx + dv_yy) - v * (-c + d * u)
    return [eq_u, eq_v]


geom = dde.geometry.Rectangle([-10, -10], [10, 10])
timedomain = dde.geometry.TimeDomain(0, 24)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

ic_u = dde.icbc.IC(geomtime, ic_func_u, lambda _, on_initial: on_initial, component=0)
ic_v = dde.icbc.IC(geomtime, ic_func_v, lambda _, on_initial: on_initial, component=1)

print("Initial wolf sum:", np.sum(ic_func_v(np.random.rand(10000, 2))))
print("Initial hare sum:", np.sum(ic_func_u(np.random.rand(10000, 2))))

bc_u = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=0)
bc_v = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=1)

# Модифікація коефіцієнтів втрат
weights = tf.constant([0.1, 0.9], dtype=tf.float32)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic_u, ic_v, bc_u, bc_v],
    num_domain=400,
    num_boundary=80,
    num_initial=40,
    num_test=10000,
)

layer_size = [3] + [128] * 2 + [64] * 2 + [32] + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_output_transform(lambda x, y: tf.minimum(tf.nn.relu(y), 1000))

model = dde.Model(data, net)

model.compile("adam", lr=0.001, loss_weights=weights)
losshistory, train_state = model.train(iterations=20000)

# Дотреновування з іншим оптимізатором
model.compile("L-BFGS")
losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)


def plot_solution_separate(model, times, num_points=100):
    x = np.linspace(-10, 10, num_points)
    y = np.linspace(-10, 10, num_points)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack((X.flatten(), Y.flatten())).T

    for t in times:
        T = np.full((num_points * num_points, 1), t)
        input_data = np.hstack((XY, T))
        Z = model.predict(input_data)
        Z_u = Z[:, 0].reshape((num_points, num_points))
        Z_v = Z[:, 1].reshape((num_points, num_points))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        c = plt.pcolormesh(X, Y, Z_u, shading='auto', cmap='viridis')
        plt.title(f'u (Prey) at t = {t:.2f}')
        plt.colorbar(c)
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 2, 2)
        c = plt.pcolormesh(X, Y, Z_v, shading='auto', cmap='viridis')
        plt.title(f'v (Predator) at t = {t:.2f}')
        plt.colorbar(c)
        plt.xlabel('x')
        plt.ylabel('y')

        plt.show()

        # Середнє значення на межах
        boundary_mean_u = np.mean([Z_u[0, :], Z_u[-1, :], Z_u[:, 0], Z_u[:, -1]])
        boundary_mean_v = np.mean([Z_v[0, :], Z_v[-1, :], Z_v[:, 0], Z_v[:, -1]])

        # Середнє значення в центрі
        center_mean_u = np.mean(Z_u[1:-1, 1:-1])
        center_mean_v = np.mean(Z_v[1:-1, 1:-1])

        print(f"Boundary u: {boundary_mean_u}, Center u: {center_mean_u}")
        print(f"Boundary v: {boundary_mean_v}, Center v: {center_mean_v}")


times = []
for i in range(25):
    times.append(i)
plot_solution_separate(model, times)

