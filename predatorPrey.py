import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
# Параметри моделі
T = 100.0         # максимальний час
S = 10.0          # половина довжини сторони квадрата
Du = 0.01         # коефіцієнт дифузії для зайців
Dv = 0.01         # коефіцієнт дифузії для вовків
ru = 0.2          # коефіцієнт безумовного росту зайців
rv = -0.1         # коефіцієнт безумовного росту вовків
auv = 0.2         # вплив вовків на зайців
avu = -0.2        # вплив зайців на вовків
auu = 0.0         # ефект надлишкової популяції зайців
avv = 0.0         # ефект надлишкової популяції вовків
gridPoints=5
np.random.seed(2)
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

# Геометрія: простір (x,y) в прямокутнику і час t
geom = dde.geometry.Rectangle([-S, -S], [S, S])
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)



def init_u(x):
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
ic_u = dde.IC(
    geomtime,
    init_u,
    lambda x, on_initial: np.isclose(x[0], 0),
    component=0
)

ic_v = dde.IC(
    geomtime,
    init_v,
    lambda x, on_initial: np.isclose(x[0], 0),
    component=1
)

# Об'єднуємо всі умови та PDE в одну задачу
data = dde.data.TimePDE(
    geomtime,
    pde_system,
    [bc_u, bc_v, ic_u, ic_v],
    num_domain=2540,
    num_boundary=80,
    num_initial=160
)

# Налаштування нейронної мережі (FNN)
layer_size = [3] + [50]*3 + [2]  # 3 входи (t,x,y) і 2 виходи (u, v) #maybe change
net = dde.maps.FNN(layer_size, "tanh", "Glorot uniform")
model = dde.Model(data, net)

# Компіляція та навчання моделі
model.compile("adam", lr=0.001) #maybe change
losshistory, train_state = model.train(epochs=10000) #maybe change

# Отримання прогнозу з моделі
#X = geomtime.uniform_points(10000, boundary=False)
#y_pred = model.predict(X)


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

fig = plt.figure(figsize=(12, 6))

# Create two 3D subplots for u and v
ax_u = fig.add_subplot(121, projection="3d")
ax_v = fig.add_subplot(122, projection="3d")

# Initial surface plots for u (Prey) and v (Predator)
surf_u = ax_u.plot_surface(X, Y, u_vals, cmap="viridis", edgecolor="none")
surf_v = ax_v.plot_surface(X, Y, v_vals, cmap="plasma", edgecolor="none")

ax_u.set_title("u (Prey)")
ax_v.set_title("v (Predator)")
for ax in (ax_u, ax_v):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Population")

# Adjust the layout to make room for the slider
plt.subplots_adjust(bottom=0.2)

# Create slider axes and the slider widget
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, "t", 0, T, valinit=t0, valstep=1)

# Callback function to update the plots when the slider is moved
def update(val):
    t_val = slider.val
    X, Y, u_vals, v_vals = plot_solution(t_val)
    
    # Clear the current axes and redraw the surfaces
    ax_u.clear()
    ax_v.clear()
    
    surf_u = ax_u.plot_surface(X, Y, u_vals, cmap="viridis", edgecolor="none")
    surf_v = ax_v.plot_surface(X, Y, v_vals, cmap="plasma", edgecolor="none")
    
    ax_u.set_title("u (Prey)")
    ax_v.set_title("v (Predator)")
    for ax in (ax_u, ax_v):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Population")
    
    # Redraw the figure canvas
    fig.canvas.draw_idle()

# Connect the update function to the slider
slider.on_changed(update)

plt.show()

