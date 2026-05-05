import numpy as np
import matplotlib.pyplot as plt


# 1. Визначення функцій
# Тестова функція Розенброка (з методички)
def rosenbrock(X):
    # f(X) = 100(x1^2 - x2)^2 + (x1 - 1)^2
    return 100 * (X[0] ** 2 - X[1]) ** 2 + (X[0] - 1) ** 2


# Цільова функція для нашої системи рівнянь: Ф(X) = f1(X)^2 + f2(X)^2
def objective_system(X):
    f1 = X[0] ** 2 + X[1] ** 2 - 4  # Коло радіусом 2
    f2 = X[1] - X[0] ** 2  # Парабола
    return f1 ** 2 + f2 ** 2


# 2. Логіка досліджуючого пошуку
def exploratory_search(X_base, delta_X, func, q, eps1, allow_decrease=True):
    """
    Досліджуючий пошук по кожній координаті.
    allow_decrease: прапорець, який дозволяє або забороняє зменшення кроку
    (заборонено під час пошуку по зразку згідно з методичкою).
    """
    X1 = X_base.copy()
    n = len(X1)

    for i in range(n):
        while True:
            # 1. Пробуємо зробити крок вперед (+delta_x)
            X_temp_plus = X1.copy()
            X_temp_plus[i] += delta_X[i]

            # Якщо функція зменшилась, фіксуємо нову точку і йдемо до наступної координати
            if func(X_temp_plus) < func(X1):
                X1 = X_temp_plus
                break

                # 2. Якщо крок вперед не допоміг, пробуємо крок назад (-delta_x)
            X_temp_minus = X1.copy()
            X_temp_minus[i] -= delta_X[i]

            # Якщо функція зменшилась, фіксуємо точку
            if func(X_temp_minus) < func(X1):
                X1 = X_temp_minus
                break

            # 3. Якщо жоден напрямок не покращив результат
            if allow_decrease:
                # Зменшуємо крок по поточній змінній
                delta_X[i] /= q
                # Якщо крок став меншим за задану точність, повертаємось до старої точки
                if delta_X[i] < eps1:
                    break  # Переходимо до наступного i
                # Якщо крок ще достатньо великий, цикл while True повторить пошук з меншим кроком
            else:
                # Якщо ми в режимі пошуку по зразку, зменшувати крок не можна
                break

    return X1, delta_X


# 3. Основний алгоритм Хука-Дживса
def hooke_jeeves(func, X_start, delta_X_start, eps1, eps2, q=2, p=2, max_iter=1000):
    X0 = np.array(X_start, dtype=float)
    delta_X = np.array(delta_X_start, dtype=float)

    trajectory = [X0.copy()]
    steps = 0

    for _ in range(max_iter):
        # Етап 1: Досліджуючий пошук з базової точки X0
        X1, delta_X = exploratory_search(X0, delta_X, func, q, eps1, allow_decrease=True)
        steps += 1
        trajectory.append(X1.copy())

        # Перевірка умов закінчення пошуку
        norm_delta = np.linalg.norm(delta_X)
        func_diff = abs(func(X1) - func(X0))

        # Якщо точка не змінилась або досягнута необхідна точність
        if np.array_equal(X1, X0) or (norm_delta < eps1 and func_diff < eps2):
            return X1, steps, trajectory

        # Етап 2: Пошук по зразку
        while True:
            # Знаходимо наступну точку вздовж вдалого напрямку
            X_p2 = X1 + p * (X1 - X0)

            # Досліджуючий пошук з нової точки X_p2, АЛЕ без зменшення кроку
            X2, _ = exploratory_search(X_p2, delta_X.copy(), func, q, eps1, allow_decrease=False)
            steps += 1
            trajectory.append(X2.copy())

            # Якщо пошук по зразку виявився вдалим
            if func(X2) < func(X1):
                X0 = X1.copy()
                X1 = X2.copy()
                # Повторюємо пошук по зразку далі
            else:
                # Якщо не вдалося, робимо поточну точку базовою і повертаємось
                # до звичайного досліджуючого пошуку
                X0 = X1.copy()
                break

    return X1, steps, trajectory


# 4. Виконання завдання

if __name__ == "__main__":
    # Параметри алгоритму
    eps1 = 1e-5  # Критерій для кроку
    eps2 = 1e-5  # Критерій для значення функції
    q = 2.0  # Коефіцієнт зменшення кроку
    p = 2.0  # Коефіцієнт прискорення для пошуку по зразку

    # Тест 1: Функція Розенброка
    X0_rosen = [-1.2, 0.0]
    delta_X_rosen = [0.5, 0.5]

    print("--- Тестування: Функція Розенброка ---")
    min_rosen, steps_rosen, traj_rosen = hooke_jeeves(rosenbrock, X0_rosen, delta_X_rosen, eps1, eps2, q, p)
    print(f"Знайдений мінімум: X* = [{min_rosen[0]:.5f}, {min_rosen[1]:.5f}]")
    print(f"Значення функції: f(X*) = {rosenbrock(min_rosen):.5e}")
    print(f"Кількість кроків: {steps_rosen}\n")

    # Тест 2: Система нелінійних рівнянь
    # Початкове наближення (з графічного аналізу кола і параболи)
    X0_sys = [1.0, 1.0]
    delta_X_sys = [0.5, 0.5]

    print("--- Розв'язок системи нелінійних рівнянь ---")
    min_sys, steps_sys, traj_sys = hooke_jeeves(objective_system, X0_sys, delta_X_sys, eps1, eps2, q, p)
    print(f"Розв'язок системи: x1 = {min_sys[0]:.5f}, x2 = {min_sys[1]:.5f}")
    print(f"Значення цільової функції: Ф(X*) = {objective_system(min_sys):.5e}")
    print(f"Кількість кроків: {steps_sys}\n")

    # Запис траєкторії спуску у файл
    filename = "trajectory_hooke_jeeves.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write("Траєкторія спуску для системи нелінійних рівнянь:\n")
        file.write("Крок |        x1        |        x2        |      Ф(x1, x2)\n")
        file.write("-" * 65 + "\n")
        for i, point in enumerate(traj_sys):
            val = objective_system(point)
            file.write(f"{i:4d} | {point[0]:16.8f} | {point[1]:16.8f} | {val:16.8e}\n")
    print(f"Координати траєкторії спуску успішно збережені у файл: '{filename}'")

    # Візуалізація системи рівнянь та знайденого розв'язку
    x_vals = np.linspace(-3, 3, 400)
    y_vals = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    F1 = X ** 2 + Y ** 2 - 4
    F2 = Y - X ** 2

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, F1, levels=[0], colors='blue')
    plt.contour(X, Y, F2, levels=[0], colors='red')

    # Малюємо початкову точку та знайдений розв'язок
    plt.plot(X0_sys[0], X0_sys[1], 'go', label='Початкове наближення X(0)')
    plt.plot(min_sys[0], min_sys[1], 'r*', markersize=10, label='Знайдений розв\'язок X*')

    plt.title("Графіки рівнянь системи та точка розв'язку")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    # Створення кастомних ліній для легенди (contour не підтримує label напряму)
    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8),
                    Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12)]
    plt.legend(custom_lines, ['x1^2 + x2^2 - 4 = 0', 'x2 - x1^2 = 0', 'Початкове наближення', 'Розв\'язок'])
    plt.show()