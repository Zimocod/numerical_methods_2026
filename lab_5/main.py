import math
import numpy as np
import matplotlib.pyplot as plt


# 1. ЗАДАНА ФУНКЦІЯ ТА АНАЛІТИЧНЕ РІШЕННЯ

def f(x):
    """Функція інтенсивності навантаження на сервер."""
    return 50 + 20 * np.sin(math.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


def get_exact_integral():
    """
    Повертає точне аналітичне значення означеного інтегралу на відрізку [0, 24].
    Використовує функцію помилок Гаусса (erf) з модуля math.
    """
    term1 = 1200
    term2 = 0
    term3 = 5 * math.sqrt(5 * math.pi) * math.erf(12 * math.sqrt(0.2))
    return term1 + term2 + term3


# 2. МЕТОДИ ЧИСЕЛЬНОГО ІНТЕГРУВАННЯ

def simpson_rule(func, a, b, n):
    """Обчислює інтеграл за складовою формулою Сімпсона для n розбиттів."""
    if n % 2 != 0:
        n += 1  # Кількість розбиттів має бути парною

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([func(val) for val in x])

    # Формула Сімпсона: h/3 * (y[0] + 4*sum(непарні) + 2*sum(парні) + y[n])
    s = y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1]
    return (h / 3) * s


def _adaptive_simpson_recursive(func, a, b, delta, f_a, f_mid, f_b, whole_integral):
    """Внутрішня рекурсивна функція для адаптивного алгоритму."""
    mid = (a + b) / 2
    f_mid_left = func((a + mid) / 2)
    f_mid_right = func((mid + b) / 2)

    # Ми викликали функцію двічі для нових серединних точок
    evaluations_count = 2

    h = (b - a) / 2
    left_integral = (h / 6) * (f_a + 4 * f_mid_left + f_mid)
    right_integral = (h / 6) * (f_mid + 4 * f_mid_right + f_b)

    # Умова збіжності
    if abs(left_integral + right_integral - whole_integral) <= 15 * delta:
        result = left_integral + right_integral + (left_integral + right_integral - whole_integral) / 15
        return result, evaluations_count

    # Якщо точність недостатня, йдемо вглиб дерева (рекурсія)
    left_res, left_evals = _adaptive_simpson_recursive(func, a, mid, delta / 2, f_a, f_mid_left, f_mid, left_integral)
    right_res, right_evals = _adaptive_simpson_recursive(func, mid, b, delta / 2, f_mid, f_mid_right, f_b,
                                                         right_integral)

    return left_res + right_res, evaluations_count + left_evals + right_evals


def adaptive_simpson(func, a, b, delta):
    """
    Обчислює інтеграл за адаптивним алгоритмом Сімпсона.
    Повертає кортеж: (значення_інтегралу, кількість_викликів_функції).
    """
    f_a = func(a)
    f_mid = func((a + b) / 2)
    f_b = func(b)
    h = b - a
    whole_int = (h / 6) * (f_a + 4 * f_mid + f_b)

    result, evals = _adaptive_simpson_recursive(func, a, b, delta, f_a, f_mid, f_b, whole_int)
    return result, evals + 3  # Додаємо 3 початкові виклики


# 3. МЕТОДИ УТОЧНЕННЯ (ЕКСТРАПОЛЯЦІЯ)

def runge_romberg(i_n, i_n_half):
    """Уточнення результату за методом Рунге-Ромберга (для порядку p=4)."""
    return i_n + (i_n - i_n_half) / 15


def aitken_method(i_n, i_n_half, i_n_quarter):
    """
    Уточнення за методом Ейткена та оцінка порядку методу.
    Повертає кортеж: (уточнене_значення, оцінений_порядок_p).
    """
    numerator = (i_n_half) ** 2 - i_n * i_n_quarter
    denominator = 2 * i_n_half - (i_n + i_n_quarter)
    i_eitken = numerator / denominator

    p = (1 / math.log(2)) * math.log(abs((i_n_quarter - i_n_half) / (i_n_half - i_n)))
    return i_eitken, p


# 4. ГОЛОВНА ПРОЦЕДУРА ЛАБОРАТОРНОЇ РОБОТИ

def main():
    A, B = 0, 24
    TARGET_EPS = 1e-12

    print("-" * 50)
    # ПУНКТ 2: Точне значення
    I_exact = get_exact_integral()
    print(f"2. Точне значення інтегралу I0 = {I_exact:.12f}")

    # ПУНКТ 4: Дослідження залежності точності від N
    print("-" * 50)
    print("4. Дослідження залежності точності Сімпсона від N...")
    n_values = np.arange(10, 1002, 2)
    errors = []

    n_opt = None
    eps_opt = None

    for n in n_values:
        i_val = simpson_rule(f, A, B, n)
        err = abs(i_val - I_exact)
        errors.append(err)

        if err <= TARGET_EPS and n_opt is None:
            n_opt = n
            eps_opt = err

    if n_opt is None:
        n_opt = n_values[-1]
        eps_opt = errors[-1]
        print(f"   Точність {TARGET_EPS} не досягнута до N=1000. Беремо N_opt = {n_opt}")
    else:
        print(f"   Оптимальне N_opt = {n_opt}, похибка eps_opt = {eps_opt:.2e}")

    # Побудова графіка
    plt.figure(figsize=(10, 5))
    plt.plot(n_values, errors, label=r'Похибка $\epsilon(N)$')
    plt.axhline(TARGET_EPS, color='r', linestyle='--', label='Задана точність 1e-12')
    plt.yscale('log')
    plt.title('Залежність похибки методу Сімпсона від кількості розбиттів N')
    plt.xlabel('Число розбиттів, N')
    plt.ylabel(r'Абсолютна похибка, $\epsilon$')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show(block=False)  # Графік не блокує подальше виконання програми

    # ПУНКТ 5: Знаходимо N0 (кратне 8)
    print("-" * 50)
    n0 = int(n_opt / 10)
    n0 = max(8, n0 + (8 - n0 % 8) % 8)

    i_n0 = simpson_rule(f, A, B, n0)
    eps0 = abs(i_n0 - I_exact)
    print(f"5. Вибрано N0 = {n0}. Значення = {i_n0:.12f}, Похибка = {eps0:.2e}")

    # Базові обчислення для уточнень
    i_n0_half = simpson_rule(f, A, B, int(n0 / 2))
    i_n0_quarter = simpson_rule(f, A, B, int(n0 / 4))

    # ПУНКТ 6: Метод Рунге-Ромберга
    print("-" * 50)
    i_runge = runge_romberg(i_n0, i_n0_half)
    eps_runge = abs(i_runge - I_exact)
    print(f"6. Метод Рунге-Ромберга: Значення = {i_runge:.12f}, Похибка = {eps_runge:.2e}")

    # ПУНКТ 7: Метод Ейткена
    print("-" * 50)
    i_eitken, p_estimated = aitken_method(i_n0, i_n0_half, i_n0_quarter)
    eps_eitken = abs(i_eitken - I_exact)
    print(f"7. Метод Ейткена: Значення = {i_eitken:.12f}, Похибка = {eps_eitken:.2e}")
    print(f"   Оцінений порядок методу Сімпсона p = {p_estimated:.4f}")

    # ПУНКТ 8: Аналіз похибок
    print("-" * 50)
    print("8. Аналіз результатів (для N0):")
    print(f"   Базовий Сімпсон: {eps0:.2e}")
    print(f"   Рунге-Ромберг:   {eps_runge:.2e} (покращення в {eps0 / eps_runge:.1f} разів)")
    print(f"   Ейткен:          {eps_eitken:.2e} (покращення в {eps0 / eps_eitken:.1f} разів)")

    # ПУНКТ 9: Адаптивний алгоритм
    print("-" * 50)
    print("9. Адаптивний алгоритм:")
    deltas = [1e-3, 1e-6, 1e-9, 1e-12]

    for d in deltas:
        i_adapt, evals = adaptive_simpson(f, A, B, d)
        err_adapt = abs(i_adapt - I_exact)
        print(f"   delta = {d:.0e} | Похибка = {err_adapt:.2e} | Викликів функції = {evals}")

    plt.show()  # Зупиняємо скрипт, щоб подивитись графік в кінці


if __name__ == "__main__":
    main()