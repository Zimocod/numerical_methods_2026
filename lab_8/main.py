import math
import cmath

# ЧАСТИНА 1: Трансцендентне рівняння

# F(x) = sin(x) - x/2
def F(x):
    return math.sin(x) - 0.5 * x


def dF(x):
    return math.cos(x) - 0.5


def d2F(x):
    return -math.sin(x)


def tabulate_function(a, b, h, filename="tabulation.txt"):
    """
    Табулює функцію та записує результати у файл.
    Знаходить інтервали, де функція змінює знак (ознака кореня).
    """
    intervals = []
    x = a

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("x\t\tF(x)\n")
        f.write("-" * 30 + "\n")

        prev_x = x
        prev_F = F(x)

        while x <= b + h / 10:  # h/10 для уникнення проблем з плаваючою комою
            curr_F = F(x)
            f.write(f"{x:.4f}\t{curr_F:.4f}\n")

            # Якщо функція змінила знак, між prev_x та x є корінь
            if prev_F * curr_F < 0:
                behavior = "зростає" if curr_F > prev_F else "спадає"
                intervals.append((prev_x, x, behavior))

            prev_x = x
            prev_F = curr_F
            x += h

    return intervals


# --- Однокрокові методи ---

def simple_iteration(x0, eps=1e-10):
    """Метод простої ітерації (з релаксацією)"""
    # Знаходимо параметр tau: -2 < tau * F'(x0) < 0
    # Якщо F'(x0) > 0, tau має бути від'ємним.
    tau = -0.5 / dF(x0)

    x_prev = x0
    iterations = 0
    while True:
        iterations += 1
        x_next = x_prev + tau * F(x_prev)

        # Критерій зупинки
        if abs(F(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iterations
        x_prev = x_next


def newton_method(x0, eps=1e-10):
    """Метод Ньютона"""
    x_prev = x0
    iterations = 0
    while True:
        iterations += 1
        x_next = x_prev - F(x_prev) / dF(x_prev)

        if abs(F(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iterations
        x_prev = x_next


def chebyshev_method(x0, eps=1e-10):
    """Метод Чебишева (формула другого порядку)"""
    x_prev = x0
    iterations = 0
    while True:
        iterations += 1
        f_val = F(x_prev)
        df_val = dF(x_prev)
        d2f_val = d2F(x_prev)

        # Основна формула
        term1 = f_val / df_val
        term2 = 0.5 * (f_val ** 2 * d2f_val) / (df_val ** 3)
        x_next = x_prev - term1 - term2

        if abs(F(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iterations
        x_prev = x_next


# --- Багатокрокові методи ---

def secant_method(x0, x1, eps=1e-10):
    """Метод хорд (січних)"""
    x_prev_prev = x0
    x_prev = x1
    iterations = 0
    while True:
        iterations += 1
        f_prev = F(x_prev)
        f_prev_prev = F(x_prev_prev)

        # Основна формула
        x_next = x_prev - f_prev * (x_prev - x_prev_prev) / (f_prev - f_prev_prev)

        if abs(F(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iterations

        x_prev_prev = x_prev
        x_prev = x_next


def parabola_method(x0, x1, x2, eps=1e-10):
    """Метод парабол"""
    xn_2, xn_1, xn = x0, x1, x2
    iterations = 0
    while True:
        iterations += 1
        # Розділені різниці
        f_xn_1_xn_2 = (F(xn_1) - F(xn_2)) / (xn_1 - xn_2)
        f_xn_xn_1 = (F(xn) - F(xn_1)) / (xn - xn_1)
        f_xn_xn_1_xn_2 = (f_xn_xn_1 - f_xn_1_xn_2) / (xn - xn_2)

        # Обчислення дискримінанта та дельти
        B = (xn - xn_1) * f_xn_xn_1_xn_2 + f_xn_xn_1
        D = B ** 2 - 4 * f_xn_xn_1_xn_2 * F(xn)

        # Дозволяє знайти комплексні корені, тому використовуємо cmath
        root_D = cmath.sqrt(D)

        denom = 2 * f_xn_xn_1_xn_2
        if denom == 0:
            break

        delta1 = (-B + root_D) / denom
        delta2 = (-B - root_D) / denom

        # Вибираємо найменшу дельту
        delta = delta1 if abs(delta1) < abs(delta2) else delta2
        x_next = xn + delta.real  # Для дійсних коренів залишаємо дійсну частину

        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iterations

        xn_2, xn_1, xn = xn_1, xn, x_next


def inverse_interpolation(x0, x1, x2, eps=1e-10):
    """Метод зворотної інтерполяції (для 3 вузлів)"""
    xn_2, xn_1, xn = x0, x1, x2
    iterations = 0
    while True:
        iterations += 1
        y0, y1, y2 = F(xn_2), F(xn_1), F(xn)

        # Інтерполяційний многочлен Лагранжа для y=0
        term1 = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * xn_2
        term2 = (y0 * y2) / ((y1 - y0) * (y1 - y2)) * xn_1
        term3 = (y0 * y1) / ((y2 - y0) * (y2 - y1)) * xn

        x_next = term1 + term2 + term3

        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iterations

        xn_2, xn_1, xn = xn_1, xn, x_next


# ЧАСТИНА 2: Алгебраїчні рівняння

def write_polynomial_to_file(coeffs, filename="coefficients.txt"):
    """Записує коефіцієнти многочлена у файл."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(" ".join(map(str, coeffs)))


def read_polynomial_from_file(filename="coefficients.txt"):
    """Зчитує коефіцієнти з файлу."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().split()
        return [float(c) for c in data]


def poly_val(coeffs, x):
    """Повертає значення многочлена для даного x (Схема Горнера)."""
    val = coeffs[0]
    for c in coeffs[1:]:
        val = val * x + c
    return val


def newton_horner(coeffs, x0, eps=1e-10):
    """Знаходження дійсного кореня по методу Ньютона + Схема Горнера"""
    x_prev = x0
    iterations = 0
    m = len(coeffs) - 1

    while True:
        iterations += 1

        # Схема Горнера для знаходження F(xn) = b0
        b = [0] * (m + 1)
        b[0] = coeffs[0]  # b_m = a_m
        for i in range(1, m + 1):
            b[i] = coeffs[i] + x_prev * b[i - 1]

        f_val = b[-1]  # b_0

        # Схема Горнера для знаходження F'(xn) = c1
        c = [0] * m
        c[0] = b[0]  # c_m = b_m
        for i in range(1, m):
            c[i] = b[i] + x_prev * c[i - 1]

        df_val = c[-1]  # c_1

        x_next = x_prev - f_val / df_val

        if abs(poly_val(coeffs, x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iterations
        x_prev = x_next


def lin_method(coeffs, alpha0, beta0, eps=1e-10):
    """Метод Ліна для комплексних коренів."""
    iterations = 0
    alpha, beta = alpha0, beta0
    m = len(coeffs) - 1
    a = coeffs[::-1]  # Реверсуємо масив, щоб a[0] був вільним членом

    while True:
        iterations += 1
        p0 = -2 * alpha
        q0 = alpha ** 2 + beta ** 2

        b = [0] * (m + 1)
        b[m] = a[m]
        b[m - 1] = a[m - 1] - p0 * b[m]

        for i in range(m - 2, 1, -1):
            b[i] = a[i] - p0 * b[i + 1] - q0 * b[i + 2]

        if b[2] == 0:
            break  # Уникнення ділення на нуль

        q1 = a[0] / b[2]
        p1 = (a[1] * b[2] - a[0] * b[3]) / (b[2] ** 2)

        alpha1 = -p1 / 2
        beta_squared = q1 - alpha1 ** 2

        # Уникаємо комплексних значень самої бети
        if beta_squared < 0:
            beta1 = math.sqrt(abs(beta_squared))
        else:
            beta1 = math.sqrt(beta_squared)

        if abs(alpha1 - alpha) < eps and abs(beta1 - beta) < eps:
            return complex(alpha1, beta1), iterations

        alpha, beta = alpha1, beta1


# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
if __name__ == "__main__":
    print("--- ЧАСТИНА 1: Трансцендентне рівняння F(x) = sin(x) - x/2 ---")
    eps = 1e-10

    # 1. Табуляція
    print("1. Табуляція функції на відрізку [1, 3] з кроком 0.1...")
    intervals = tabulate_function(1.0, 3.0, 0.1)

    # Вибираємо перший знайдений інтервал для демонстрації
    # (можна розширити табуляцію від -3 до 3, щоб знайти інші корені)
    if intervals:
        a, b, behavior = intervals[0]
        print(f"Знайдено корінь на проміжку [{a:.2f}, {b:.2f}]. Поведінка: {behavior}")

        x0 = (a + b) / 2  # Початкове наближення

        # 2-4. Тестування методів
        print(f"\nПочаткове наближення для всіх методів: x0 = {x0}")
        print("Точність: 1e-10")

        res_simple, it_simple = simple_iteration(x0, eps)
        print(f"Проста ітерація:  x = {res_simple:.10f}, ітерацій: {it_simple}")

        res_newton, it_newton = newton_method(x0, eps)
        print(f"Метод Ньютона:    x = {res_newton:.10f}, ітерацій: {it_newton}")

        res_cheb, it_cheb = chebyshev_method(x0, eps)
        print(f"Метод Чебишева:   x = {res_cheb:.10f}, ітерацій: {it_cheb}")

        res_secant, it_secant = secant_method(a, b, eps)
        print(f"Метод хорд:       x = {res_secant:.10f}, ітерацій: {it_secant}")

        res_para, it_para = parabola_method(a, x0, b, eps)
        print(f"Метод парабол:    x = {res_para:.10f}, ітерацій: {it_para}")

        res_inv, it_inv = inverse_interpolation(a, x0, b, eps)
        print(f"Зворотна інтерп.: x = {res_inv:.10f}, ітерацій: {it_inv}")

    print("\n--- ЧАСТИНА 2: Алгебраїчне рівняння x^3 - 4x^2 + 6x - 4 = 0 ---")
    # Коефіцієнти: x^3 - 4x^2 + 6x - 4 (корені: 2, 1+i, 1-i)
    # Зберігаємо за спаданням степеня
    poly_coeffs = [1, -4, 6, -4]

    # 6-7. Запис та зчитування файлу
    write_polynomial_to_file(poly_coeffs)
    loaded_coeffs = read_polynomial_from_file()
    print(f"Зчитані коефіцієнти з файлу: {loaded_coeffs}")

    # 8. Дійсний корінь по Ньютону+Горнеру
    x0_alg = 3.0  # Початкове наближення для дійсного кореня
    res_horner, it_horner = newton_horner(loaded_coeffs, x0_alg, eps)
    print(f"Дійсний корінь (Ньютон-Горнер): x = {res_horner:.10f}, ітерацій: {it_horner}")

    # 9. Комплексні корені по методу Ліна
    alpha0, beta0 = 0.5, 0.5  # Початкові наближення для комплексної частини
    res_lin, it_lin = lin_method(loaded_coeffs, alpha0, beta0, eps)
    print(f"Комплексний корінь (Метод Ліна): x = {res_lin.real:.10f} ± {res_lin.imag:.10f}i, ітерацій: {it_lin}")