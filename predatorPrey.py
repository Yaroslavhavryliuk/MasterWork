import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from deepxde.backend import tf
from Interface import Interface

interface = Interface()
coefficients = interface.run()


# Параметри моделі
T = coefficients['T']         # максимальний час
S = coefficients['S']          # половина довжини сторони квадрата
Du = coefficients['D_u']         # коефіцієнт дифузії для зайців
Dv = coefficients['D_v']         # коефіцієнт дифузії для вовків
ru = coefficients['ru']          # коефіцієнт безумовного росту зайців
rv = coefficients['rv']         # коефіцієнт безумовного росту вовків
auv = coefficients['auv']         # вплив вовків на зайців
avu = coefficients['avu']        # вплив зайців на вовків
auu = coefficients['auu']         # ефект надлишкової популяції зайців
avv = coefficients['avv']         # ефект надлишкової популяції вовків
gridPoints = 10
np.random.seed(2)
delta = 0.01

# Геометрія: простір (x,y) в прямокутнику і час t
geom = dde.geometry.Rectangle([-S, -S], [S, S])
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

x = np.linspace(-S + delta, S - delta, gridPoints)
y = np.linspace(-S + delta, S - delta, gridPoints)
xx, yy = np.meshgrid(x, y)
tt = np.zeros_like(xx)
points = np.vstack([xx.ravel(), yy.ravel(), tt.ravel()]).T  # форма (N, 3)


def gen_random_interp(S,gridPoints):

    matrix = np.random.uniform(1, 2, (gridPoints, gridPoints))  # Random values between 1 and 2
    matrix[0, :] = 0  # Top border
    matrix[-1, :] = 0  # Bottom border
    matrix[:, 0] = 0  # Left border
    matrix[:, -1] = 0  # Right border

    x = np.linspace(-S, S, gridPoints)  # Map matrix columns to [-S, S]
    y = np.linspace(-S, S, gridPoints)  # Map matrix rows to [-S, S]

    interpolator = interp.RegularGridInterpolator((y, x), matrix, method="cubic", bounds_error=False, fill_value=0)

    def interpolated_function(x_query, y_query):
        points = np.array([y_query, x_query]).T  # Ensure correct shape for querying
        return interpolator(points)

    return interpolated_function


interp_u = gen_random_interp(S,gridPoints)
interp_v = gen_random_interp(S,gridPoints)


values_u = interp_u(points[:, 0], points[:, 1])[:, None]
values_v = interp_v(points[:, 0], points[:, 1])[:, None]


# Система PDE: визначаємо рівняння для u і v
def pde_system(x, y):
    # Розпаковка змінних
    u = y[:, 0:1]
    v = y[:, 1:2]
    # Обчислюємо похідні за допомогою функцій deepxde
    u_t = dde.grad.jacobian(y, x, i=0, j=2)      # [u,v] -> i,  [x,y,t] -> j
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)    # [u,v] -> component   [x,y,t] -> i    [x,y,t] -> j
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)    # [u,v] -> component   [x,y,t] -> i    [x,y,t] -> j

    v_t = dde.grad.jacobian(y, x, i=1, j=2)      # [u,v] -> i,  [x,y,t] -> j
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)    # [u,v] -> component   [x,y,t] -> i    [x,y,t] -> j
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)    # [u,v] -> component   [x,y,t] -> i    [x,y,t] -> j

    # Запис PDE, приведений до вигляду, зручному для DeepXDE:
    eq_u = u_t - Du*(u_xx + u_yy) - u*(ru - auu*u - auv*v)
    eq_v = v_t - Dv*(v_xx + v_yy) - v*(rv - avv*v - avu*u)
    return [eq_u, eq_v]

# Крайові умови: для всіх t при x або y на межі [-S, S] значення u та v = 0
bc_u = dde.DirichletBC(
    geomtime,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and (
        np.isclose(x[0], -S) or np.isclose(x[0], S) or
        np.isclose(x[1], -S) or np.isclose(x[1], S)
    ),
    component=0
)

bc_v = dde.DirichletBC(
    geomtime,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and (
        np.isclose(x[0], -S) or np.isclose(x[0], S) or
        np.isclose(x[1], -S) or np.isclose(x[1], S)
    ),
    component=1
)

# Початкові умови при t = 0
ic_u = dde.PointSetBC(points, values_u, component=0)
ic_v = dde.PointSetBC(points, values_v, component=1)


# Об'єднуємо всі умови та PDE в одну задачу
data = dde.data.TimePDE(
    geomtime,
    pde_system,
    [bc_u, bc_v, ic_u, ic_v],
    num_domain=5000,
    num_boundary=100,
    num_test=10000
)


def output_transform(x, y):
    return tf.math.softplus(y)

# Налаштування нейронної мережі (FNN)
layer_size = [3] + [128] + [64] * 3 + [32] + [2]  # 3 входи (t,x,y) і 2 виходи (u, v) #maybe change
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

# Компіляція та навчання моделі
model.compile("adam", lr=0.001)
model.net.apply_output_transform(output_transform)
losshistory, train_state = model.train(epochs=20000) #maybe change

# Дотреновування з іншим оптимізатором
model.compile("L-BFGS")
losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

model.save("model_dir")  # Зберігає в папку model_dir



