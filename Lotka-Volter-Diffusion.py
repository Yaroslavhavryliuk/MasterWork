import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import tf

# Параметри моделі
D_u = 0.1
D_v = 0.1
a = 1.1
b = 0.4
c = 0.4
d = 0.1


def ic_func_u(x):
    return np.exp(-5 * ((x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2))


def ic_func_v(x):
    return np.exp(-5 * ((x[:, 0:1] + 0.5) ** 2 + (x[:, 1:2] + 0.5) ** 2))


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

bc_u = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=0)
bc_v = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=1)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic_u, ic_v, bc_u, bc_v],
    num_domain=400,
    num_boundary=80,
    num_initial=40,
    num_test=10000,
)

layer_size = [3] + [32] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=30000)

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


times = []
for i in range(25):
    times.append(i)
plot_solution_separate(model, times)

