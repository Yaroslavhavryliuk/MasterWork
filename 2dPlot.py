import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from Interface import Interface

interface = Interface()
coefficients = interface.run()
print("Received coefficients:", coefficients)


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
gridPoints = 50
np.random.seed(2)

# Геометрія: простір (x,y) в прямокутнику і час t
geom = dde.geometry.Rectangle([-S, -S], [S, S])
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

x_vals = np.linspace(-S, S, gridPoints)
y_vals = np.linspace(-S, S, gridPoints)
xx, yy = np.meshgrid(x_vals, y_vals)
x_flat = xx.flatten()
y_flat = yy.flatten()
t_flat = np.zeros_like(x_flat)

# Об'єднаємо в масив (N, 3): [t, x, y]
initial_points = np.stack([t_flat, x_flat, y_flat], axis=1)
print("Initial condition points:", initial_points.shape)

def gen_random_interp(S,gridPoints):

    matrix = np.random.uniform(1, 2, (gridPoints, gridPoints))  # Random values between 1 and 2
    matrix[0, :] = 0  # Top border
    matrix[-1, :] = 0  # Bottom border
    matrix[:, 0] = 0  # Left border
    matrix[:, -1] = 0  # Right border

    x = np.linspace(-S, S, gridPoints)  # Map matrix columns to [-S, S]
    y = np.linspace(-S, S, gridPoints)  # Map matrix rows to [-S, S]

    interpolator = interp.RegularGridInterpolator((y, x), matrix, method="cubic", bounds_error=False, fill_value=0)
    print(interpolator)

    def interpolated_function(x_query, y_query):
        points = np.array([y_query, x_query]).T  # Ensure correct shape for querying
        print(interpolator(points))
        return interpolator(points)

    return interpolated_function



interp_u = gen_random_interp(S,gridPoints)
interp_v = gen_random_interp(S,gridPoints)


def init_u(x):
    print(x)
    # x має форму (N, 3): [t, x, y]
    # Початкова умова для u при t=0
    return interp_u(x[:, 1], x[:, 2])

def init_v(x):
    # x має форму (N, 3): [t, x, y]
    # Початкова умова для v при t=0
    return interp_v(x[:, 1], x[:, 2])

# Система PDE: визначаємо рівняння для u і v
def pde_system(x, y):
    # Розпаковка змінних
    u = y[:, 0:1]
    v = y[:, 1:2]
    # Обчислюємо похідні за допомогою функцій deepxde
    u_t = dde.grad.jacobian(y, x, i=0, j=0)
    u_xx = dde.grad.hessian(y, x, component=0, i=1, j=1)
    u_yy = dde.grad.hessian(y, x, component=0, i=2, j=2)

    v_t = dde.grad.jacobian(y, x, i=1, j=0)
    v_xx = dde.grad.hessian(y, x, component=1, i=1, j=1)
    v_yy = dde.grad.hessian(y, x, component=1, i=2, j=2)

    # Запис PDE, приведений до вигляду, зручному для DeepXDE:
    eq_u = u_t - Du*(u_xx + u_yy) - u*(ru - auu*u - auv*v)
    eq_v = v_t - Dv*(v_xx + v_yy) - v*(rv - avv*v - avu*u)
    return [eq_u, eq_v]

# Крайові умови: для всіх t при x або y на межі [-S, S] значення u та v = 0
bc_u = dde.DirichletBC(
    geomtime,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and (
        np.isclose(x[1], -S) or np.isclose(x[1], S) or
        np.isclose(x[2], -S) or np.isclose(x[2], S)
    ),
    component=0
)

bc_v = dde.DirichletBC(
    geomtime,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and (
        np.isclose(x[1], -S) or np.isclose(x[1], S) or
        np.isclose(x[2], -S) or np.isclose(x[2], S)
    ),
    component=1
)

# Початкові умови при t = 0
ic_u = dde.PointSetBC(initial_points, interp_u(x_flat, y_flat)[:, None], component=0)
ic_v = dde.PointSetBC(initial_points, interp_v(x_flat, y_flat)[:, None], component=1)

# Об'єднуємо всі умови та PDE в одну задачу
data = dde.data.TimePDE(
    geomtime,
    pde_system,
    [bc_u, bc_v, ic_u, ic_v],
    num_domain=2400,
    num_boundary=320,
    num_test=10000
)

# Налаштування нейронної мережі (FNN)
layer_size = [3] + [50] * 3 + [2]  # 3 входи (t,x,y) і 2 виходи (u, v) #maybe change
net = dde.maps.FNN(layer_size, "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
model.restore("PointSet/model_dir-10000.ckpt", verbose=1)

def plot_solution(t_val):
    # Create a spatial grid for (x, y)
    n_points = 50
    x_vals = np.linspace(-S, S, n_points)
    y_vals = np.linspace(-S, S, n_points)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare the input for the model: [t, x, y]
    t_grid = t_val * np.ones_like(X)
    X_input = np.vstack([t_grid.flatten(), X.flatten(), Y.flatten()]).T

    # Get predictions for u and v
    y_pred = model.predict(X_input)
    u_vals = y_pred[:, 0].reshape(X.shape)
    v_vals = y_pred[:, 1].reshape(X.shape)

    return X, Y, u_vals, v_vals

# Create the figure and initial plots
t0 = 0  # initial time
X, Y, u_vals, v_vals = plot_solution(t0)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Створення pcolormesh графіків
pcm_u = ax1.pcolormesh(X, Y, u_vals, shading='auto', cmap='viridis')
pcm_v = ax2.pcolormesh(X, Y, v_vals, shading='auto', cmap='viridis')

# Додаємо кольорові шкали
cbar1 = fig2.colorbar(pcm_u, ax=ax1)
cbar2 = fig2.colorbar(pcm_v, ax=ax2)

# Підписи
ax1.set_title(f'u (Prey) at t = {t0:.2f}')
ax2.set_title(f'v (Predator) at t = {t0:.2f}')
for ax in (ax1, ax2):
    ax.set_xlabel("x")
    ax.set_ylabel("y")

# Слайдер
plt.subplots_adjust(bottom=0.2)
ax_slider2 = plt.axes([0.2, 0.05, 0.6, 0.03])
slider2 = Slider(ax_slider2, "t", 0, T, valinit=t0, valstep=1)


# Функція оновлення
def update2D(val):
    t_val = slider2.val
    X, Y, u_vals, v_vals = plot_solution(t_val)

    # Оновлюємо дані
    pcm_u.set_array(u_vals.ravel())
    pcm_u.set_clim(vmin=u_vals.min(), vmax=u_vals.max())

    pcm_v.set_array(v_vals.ravel())
    pcm_v.set_clim(vmin=v_vals.min(), vmax=v_vals.max())

    ax1.set_title(f'u (Prey) at t = {t_val:.2f}')
    ax2.set_title(f'v (Predator) at t = {t_val:.2f}')

    fig2.canvas.draw_idle()


slider2.on_changed(update2D)
plt.show()
