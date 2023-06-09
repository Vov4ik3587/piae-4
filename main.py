# Технология Кобба-Дугласа. Число входных ресурсов 3. 
# Постоянная отдача от масштаба. Ресурсы изменяются в пределах [0, 10]
# Локальное D-оптимальное планирование

# %% Импортируем нужные библиотеки

import numpy as np
import matplotlib.pyplot as plt


# %% Определяем необходимые функции

def generate_error(sqrt_variance, amount_tests):
    return np.random.normal(0, sqrt_variance, amount_tests)


def experiment(plan, true_theta):
    ARSS = 0
    Ahead = 0
    Num = 100
    for _ in range(Num):
        Y = calculate_response(plan, true_theta)
        # Y = np.log(Y)
        X = np.array(list(map(lambda x: [1.0, np.math.log(x[0]), np.math.log(x[1]), np.math.log(x[2])], plan)))
        theta_hat = OLS(X, Y)
        ARSS += RSS(Y, X, theta_hat)
        Ahead += np.dot(theta_hat - true_theta, theta_hat - true_theta)
    ARSS /= Num
    Ahead /= Num
    return ARSS, Ahead


def OLS(xf, y):
    theta_hat = np.linalg.inv(xf.T @ xf) @ xf.T @ y
    theta_hat[0] = np.math.e ** theta_hat[0]
    return theta_hat


def calculate_response(X, theta):
    signal = np.array([model_func_eta(x, theta) for x in X])  # Вычисляем сигнал
    power = np.vdot(signal - np.mean(signal), signal - np.mean(signal)) / len(
        signal)  # Вычисляем полезную мощность сигнала
    variance = power * noise_lvl  # Зашумляем
    response = signal + generate_error(np.sqrt(variance), N)  # Вычисляем отклик
    return response


def RSS(y, x, theta_hat):
    y_hat = np.dot(x, theta_hat)
    return np.dot(y - y_hat, y - y_hat)


def func(x, theta):
    return np.array(
        [model_func_eta(x, theta) / theta[0],
         np.math.log(x[0]) * model_func_eta(x, theta),
         np.math.log(x[1]) * model_func_eta(x, theta),
         np.math.log(x[2]) * model_func_eta(x, theta)
         ]
    )


def calculate_info_mat(factors, weights, theta):
    info_mat_tmp = np.array(
        [p * (np.vstack(func(x, theta)) @ np.vstack(func(x, theta)).T) for x, p in zip(factors, weights)]
    )
    return np.sum(info_mat_tmp, axis=0)


def D_functional(plan):
    return np.linalg.det(calculate_info_mat(plan['x'], plan['p'], theta_true))


def calculate_variance(x, info_mat, theta):
    return np.vstack(func(x, theta)).T @ np.linalg.inv(info_mat) @ np.vstack(func(x, theta))


def draw_plan(x, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.title(title)
    ax.scatter(x[0], x[1], x[2])
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plt.show()


# %% Определяем параметры модели

theta_true = np.array([0.1, 0.3, 0.5, 0.1])  # Сумма тет = 1 -- постоянная отдача
x_min, x_max = 0, 10  # Пределы изменения ресурсов
noise_lvl = 0.15  # Уровень шума


def model_func_eta(x, theta):
    return theta[0] * x[0] ** theta[1] * x[1] ** theta[2] * x[2] ** theta[3]


# %% Задаем начальный план

N = 18  # Число наблюдений
size_grid = 1001
grid = np.linspace(x_min, x_max, size_grid)

x = np.array(list(map(lambda x: [np.random.choice(grid), np.random.choice(grid), np.random.choice(grid)], range(N))))
plan = {
    'x': x,
    'p': np.array([1 / N for _ in range(N)]),
    'N': N
}

draw_plan(plan['x'].T, 'Начальный план')

# %% Задаем модель на основе начального плана

model_x = plan['x']

model = {
    'x': model_x,
    'y': calculate_response(model_x, theta_true),
    'theta_hat': []
}

# %% Оценим параметры модели

# Сначала приведем к линейной модели
X = np.array(list(map(lambda x: [1.0, np.math.log(x[0]), np.math.log(x[1]), np.math.log(x[2])], model['x'])))

# Используем МНК-оценку
model['theta_hat'] = np.linalg.inv(X.T @ X) @ X.T @ model['y']
model['theta_hat'][0] = np.math.e ** model['theta_hat'][0]  # Обратить переход к линейной модели

# %% Построим D-оптимальный план из начального

cur_plan = plan.copy()
cur_info_mat = calculate_info_mat(cur_plan['x'], cur_plan['p'], theta_true)

iteration = 0
while True:
    # Выберем точки, не содержащиеся в плане
    x_tmp = np.array(
        list(map(lambda x: [np.random.choice(grid), np.random.choice(grid), np.random.choice(grid)], range(size_grid))))
    x_s = np.array([x for x in x_tmp if x not in cur_plan['x'] and x[0] != 0 and x[1] != 0 and x[2] != 0])

    # Найдем значение дисперсии в точках вне плана
    cur_variance_s = np.array(
        [calculate_variance(x, cur_info_mat, theta_true)
         for x in x_s]
    )

    # Добавим точку с максимальной дисперсией в план и изменим его
    max_variance = np.max(cur_variance_s)
    picked_x_s_index = np.where(cur_variance_s == max_variance)[0]
    picked_x_s_index = picked_x_s_index[:1][0]
    # Перестраиваем план
    tmp = x_s[picked_x_s_index].reshape(1, 3)
    cur_plan['x'] = np.append(cur_plan['x'], x_s[picked_x_s_index].reshape(1, 3), axis=0)
    cur_plan['N'] += 1
    cur_plan['p'] = 1 / cur_plan['N'] * np.ones(cur_plan['N'])
    cur_info_mat = calculate_info_mat(cur_plan['x'], cur_plan['p'], theta_true)

    # Удалим точку с минимальной дисперсией из текущего плана
    x_j = cur_plan['x']
    cur_variance_j = np.array(
        [calculate_variance(x, cur_info_mat, theta_true)
         for x in x_j]
    )

    # Нашли точки с минимальной дисперсией
    min_variance = np.min(cur_variance_j)
    picked_x_j_index = np.where(cur_variance_j == min_variance)[0]
    picked_x_j_index = picked_x_j_index[:1][0]
    # Перестраиваем план, удаляя точку
    cur_plan['x'] = np.delete(cur_plan['x'], (picked_x_j_index), axis=0)
    cur_plan['N'] -= 1
    cur_plan['p'] = 1 / cur_plan['N'] * np.ones(cur_plan['N'])
    cur_info_mat = calculate_info_mat(
        cur_plan['x'], cur_plan['p'], theta_true)

    if picked_x_s_index == picked_x_j_index or iteration == 50:
        break
    iteration += 1

end_plan = cur_plan.copy()
draw_plan(end_plan['x'].T, 'Оптимальный план')

# %% Сравним значение D-функционалов начального плана и оптимального
print(f'Количество итераций: {iteration}')
print(f'D start plan = {D_functional(plan)}')
print(f'D start plan = {D_functional(end_plan)}')

# %% Найдем оценку параметров модели на оптимальном плане

opt_model_x = end_plan['x']

opt_model = {
    'x': opt_model_x,
    'y': calculate_response(opt_model_x, theta_true),
    'theta_hat': []
}

# Сначала приведем к линейной модели
X = np.array(list(map(lambda x: [1.0, np.math.log(x[0]), np.math.log(x[1]), np.math.log(x[2])], opt_model['x'])))

# Используем МНК-оценку
opt_model['theta_hat'] = np.linalg.inv(X.T @ X) @ X.T @ opt_model['y']
opt_model['theta_hat'][0] = np.math.e ** opt_model['theta_hat'][0]  # Обратить переход к линейной модели

# %% Сравним оценки параметров модели из начального и оптимального планов с истинными значениями

print(f'Истинные значения параметров тета: {theta_true}')
print(f"Оценка значений параметров тета начального плана: {model['theta_hat']}")
print(f"Оценка значений параметров тета локально-оптимального плана: {opt_model['theta_hat']}")

# %% Пункт 6. Проведем эксперимент

ARSS_start_plan, Ahead_start_plan = experiment(plan['x'], theta_true)
ARSS_opt_plan, Ahead_opt_plan = experiment(end_plan['x'], theta_true)

print('Начальный план')
print(f"Точки плана: {plan['x']}")
print(f"Среднее значение RSS: {ARSS_start_plan}")
print(f"Среднее значение нормы RSS: {Ahead_start_plan}")

print('Локально-оптимальный план план')
print(f"Точки плана: {end_plan['x']}")
print(f"Среднее значение RSS: {ARSS_opt_plan}")
print(f"Среднее значение нормы RSS: {Ahead_opt_plan}")

