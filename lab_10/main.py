import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


class ODESolver:
    """
    Клас для чисельного розв'язання задачі Коші для звичайних диференціальних рівнянь.
    """

    def __init__(self, func: Callable[[float, float], float], a: float, b: float, y0: float):
        """
        Ініціалізація розв'язувача.

        :param func: Функція f(x, y) правої частини диференціального рівняння y' = f(x, y).
        :param a: Початок відрізка інтегрування.
        :param b: Кінець відрізка інтегрування.
        :param y0: Початкова умова y(a) = y0.
        """
        self.f = func
        self.a = a
        self.b = b
        self.y0 = y0

    def rk4_step(self, x: float, y: float, h: float) -> float:
        """Виконує один крок обчислень методом Рунге-Кутта 4-го порядку."""
        k1 = self.f(x, y)
        k2 = self.f(x + h / 2, y + h * k1 / 2)
        k3 = self.f(x + h / 2, y + h * k2 / 2)
        k4 = self.f(x + h, y + h * k3)
        return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve_rk4_fixed(self, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Розв'язує ЗДР методом Рунге-Кутта 4-го порядку з фіксованим кроком.
        """
        x_vals = np.arange(self.a, self.b + h, h)
        y_vals = np.zeros_like(x_vals)
        y_vals[0] = self.y0

        for i in range(len(x_vals) - 1):
            y_vals[i + 1] = self.rk4_step(x_vals[i], y_vals[i], h)

        return x_vals, y_vals

    def solve_rk4_auto(self, h_init: float, epsilon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Розв'язує ЗДР методом Рунге-Кутта 4-го порядку з автоматичним вибором кроку.
        """
        x_vals, y_vals, h_vals = [self.a], [self.y0], [h_init]
        x, y, h = self.a, self.y0, h_init

        runge_k = 32  # Константа 2^(s+1) для s=4 (4-й порядок методу)

        while x < self.b:
            if x + h > self.b:
                h = self.b - x

            y_full = self.rk4_step(x, y, h)
            y_half_1 = self.rk4_step(x, y, h / 2)
            y_half_2 = self.rk4_step(x + h / 2, y_half_1, h / 2)

            # Обчислення локальної похибки за правилом Рунге
            error = (16 / 15) * abs(y_half_2 - y_full)

            if error > epsilon:
                h /= 2  # Точність недостатня, зменшуємо крок
                continue

            # Крок прийнято
            x += h
            y = y_half_2

            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)

            # Якщо похибка дуже мала, збільшуємо крок для економії ресурсів
            if error < epsilon / runge_k:
                h *= 2

        return np.array(x_vals), np.array(y_vals), np.array(h_vals)

    def solve_adams_fixed(self, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Розв'язує ЗДР методом прогнозу і корекції Адамса 2-го порядку.
        Повертає x, скореговані y та прогнозовані y (для оцінки похибки).
        """
        x_vals = np.arange(self.a, self.b + h, h)
        y_vals = np.zeros_like(x_vals)
        y_pred = np.zeros_like(x_vals)

        y_vals[0] = self.y0

        # Для запуску Адамса 2-го порядку потрібна друга точка (n=1).
        # Знаходимо її більш точним однокроковим методом (РК4).
        if len(x_vals) > 1:
            y_vals[1] = self.rk4_step(x_vals[0], y_vals[0], h)

        for i in range(1, len(x_vals) - 1):
            f_n = self.f(x_vals[i], y_vals[i])
            f_n_minus_1 = self.f(x_vals[i - 1], y_vals[i - 1])

            # Етап прогнозу (екстраполяція)
            y_pred[i + 1] = y_vals[i] + (h / 2) * (3 * f_n - f_n_minus_1)

            # Етап корекції (інтерполяція, 1 ітерація)
            f_n_plus_1 = self.f(x_vals[i + 1], y_pred[i + 1])
            y_vals[i + 1] = y_vals[i] + (h / 2) * (f_n_plus_1 + f_n)

        return x_vals, y_vals, y_pred


class ODEVisualizer:
    """Клас для побудови графіків розв'язків та аналізу похибок."""

    @staticmethod
    def plot_analysis(solver: ODESolver, exact_solution_func: Callable[[float], float], h: float, epsilon: float):
        # Отримання даних з розв'язувача
        x_rk, y_rk = solver.solve_rk4_fixed(h)
        x_rk_auto, y_rk_auto, h_rk_auto = solver.solve_rk4_auto(h, epsilon)
        x_ad, y_ad, y_pred_ad = solver.solve_adams_fixed(h)

        y_exact_rk = exact_solution_func(x_rk)
        y_exact_ad = exact_solution_func(x_ad)

        # Обчислення похибок
        error_rk_exact = np.abs(y_rk - y_exact_rk)
        error_ad_exact = np.abs(y_ad - y_exact_ad)
        error_ad_est = np.abs(y_ad - y_pred_ad)

        # Налаштування полотна для графіків
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Графік 1: Похибки методів відносно точного розв'язку
        axes[0, 0].plot(x_rk, error_rk_exact, label='РК4', marker='o', markersize=4)
        axes[0, 0].plot(x_ad, error_ad_exact, label='Адамс', marker='s', markersize=4)
        axes[0, 0].set_title("Локальна похибка відносно точного розв'язку")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("|y_n - y(x_n)|")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Графік 2: Оцінка похибки Адамса (Прогноз vs Корекція)
        axes[0, 1].plot(x_ad[2:], error_ad_est[2:], color='orange')
        axes[0, 1].set_title("Оцінка похибки Адамса |y_kor - y_pred|")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("Похибка")
        axes[0, 1].grid(True)

        # Графік 3: Динаміка кроку h(x)
        axes[1, 0].step(x_rk_auto, h_rk_auto, color='green', where='post')
        axes[1, 0].set_title("Залежність кроку h(x) (Автокрок)")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("Крок h")
        axes[1, 0].grid(True)

        # Графік 4: Порівняння розв'язків
        axes[1, 1].plot(x_rk, y_exact_rk, label='Точний розв.', color='black', linewidth=2)
        axes[1, 1].plot(x_rk_auto, y_rk_auto, label='РК4 (Авто)', linestyle='--', marker='.')
        axes[1, 1].plot(x_ad, y_ad, label='Адамс (Фікс.)', linestyle='-.')
        axes[1, 1].set_title("Графіки розв'язків")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("y(x)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()


# Використання програми (Entry Point)
if __name__ == "__main__":
    # --- БЛОК 1: АНАЛІЗ ТРАНСЦЕНДЕНТНОГО РІВНЯННЯ ---
    print("="*50)
    print(" ЧАСТИНА 1: ТРАНСЦЕНДЕНТНЕ РІВНЯННЯ y' = x - y")
    print("="*50)

    A, B = 0.0, 2.0
    Y0 = 1.0
    H_INITIAL = 1e-2
    EPSILON = 1e-4

    def my_equation(x: float, y: float) -> float:
        return x - y

    def my_exact_solution(x: np.ndarray) -> np.ndarray:
        return x - 1 + 2 * np.exp(-x)

    solver_instance = ODESolver(func=my_equation, a=A, b=B, y0=Y0)

    print(f"Відрізок: [{A}, {B}], Початковий крок: {H_INITIAL}, Точність: {EPSILON}\n")

    # Витягуємо дані для друку
    x_rk, y_rk = solver_instance.solve_rk4_fixed(H_INITIAL)
    x_ad, y_ad, _ = solver_instance.solve_adams_fixed(H_INITIAL)

    print("Метод Рунге-Кутта 4-го порядку (останні 3 точки):")
    for i in range(-3, 0):
        print(f"  x = {x_rk[i]:.4f} | y = {y_rk[i]:.6f}")

    print("\nМетод Адамса 2-го порядку (останні 3 точки):")
    for i in range(-3, 0):
        print(f"  x = {x_ad[i]:.4f} | y = {y_ad[i]:.6f}")


    # --- БЛОК 2: АЛГЕБРАЇЧНЕ РІВНЯННЯ (СТВОРЕННЯ ФАЙЛУ) ---
    print("\n" + "="*50)
    print(" ЧАСТИНА 2: АЛГЕБРАЇЧНЕ РІВНЯННЯ (Робота з файлом)")
    print("="*50)

    # Ті самі коефіцієнти з методички: x^3 - 4x^2 + 6x - 4 = 0
    poly_coeffs = [1.0, -4.0, 6.0, -4.0]
    filename = "coefficients.txt"

    # 1. Створюємо файл і записуємо в нього коефіцієнти
    print(f"1. Створюю файл '{filename}'...")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(" ".join(map(str, poly_coeffs)))
    print("   [Успішно] Файл створено та коефіцієнти записано.")

    # 2. Читаємо з файлу (імітація роботи алгоритму)
    print(f"2. Зчитую дані з файлу '{filename}'...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().split()
    loaded_coeffs = [float(c) for c in data]
    print(f"   [Успішно] Зчитані коефіцієнти: {loaded_coeffs}")

    # --- БЛОК 3: ЗАПУСК ВІЗУАЛІЗАЦІЇ ---
    print("\n" + "="*50)
    print(" ЗАПУСК ВІЗУАЛІЗАЦІЇ ГРАФІКІВ")
    print("="*50)
    print("Відкривається вікно з графіками...")
    print("(Щоб завершити роботу скрипта, закрийте вікно графіка)")

    ODEVisualizer.plot_analysis(
        solver=solver_instance,
        exact_solution_func=my_exact_solution,
        h=H_INITIAL,
        epsilon=EPSILON
    )