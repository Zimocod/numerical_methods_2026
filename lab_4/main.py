import numpy as np
import math
import matplotlib.pyplot as plt


# 1. Математичні функції

def M(t):
    """Функція вологості ґрунту"""
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)


def exact_derivative(t):
    """Аналітична похідна dM/dt"""
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


def central_difference(func, x, h):
    """Центральна різниця для чисельного диференціювання"""
    return (func(x + h) - func(x - h)) / (2 * h)


# 2. Функції для візуалізації

def plot_moisture(t_start=0, t_end=15):
    """
    Побудова графіка функції вологості.
    t_start, t_end - межі часу для спостереження.
    """
    # Генеруємо 500 точок часу від t_start до t_end
    t_values = np.linspace(t_start, t_end, 500)
    moisture_values = M(t_values)

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, moisture_values, label="Вологість $M(t)$", color='green', linewidth=2)

    # Додаємо маркер для нашої точки дослідження t0 = 1.0
    plt.plot(1.0, M(1.0), 'ro', markersize=8, label="Точка дослідження $t_0=1.0$")

    plt.title("Залежність вологості ґрунту від часу", fontsize=14)
    plt.xlabel("Час $t$", fontsize=12)
    plt.ylabel("Рівень вологості $M$", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('moisture_plot.png', dpi=300)
    plt.show()


def plot_error(h_values, errors, optimal_h, min_error):
    """Побудова графіка залежності похибки від кроку h"""
    plt.figure(figsize=(10, 6))

    # plt.loglog робить логарифмічний масштаб по обох осях
    plt.loglog(h_values, errors, label="Залежність похибки $R(h)$", color='b', linewidth=2)

    # Ставимо червону точку в місці мінімуму
    plt.loglog(optimal_h, min_error, 'ro', markersize=8,
               label=f'Оптимальний крок: $h_0 \\approx {optimal_h:.1e}$\nПохибка: $R_0 \\approx {min_error:.1e}$')

    # Оформлення графіка
    plt.title("Залежність похибки чисельного диференціювання від кроку $h$", fontsize=14)
    plt.xlabel("Крок сітки $h$ (логарифмічна шкала)", fontsize=12)
    plt.ylabel("Абсолютна похибка $R$ (логарифмічна шкала)", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('error_plot.png', dpi=300)
    plt.show()


# 3. Головний блок обчислень

def main():
    # Точка дослідження
    t0 = 1.0
    exact_val = exact_derivative(t0)
    print(f"Крок 1. Точне значення похідної в точці t0=1: {exact_val:.8f}")

    # Крок 2. Пошук оптимального кроку h
    # Створюємо масив значень h від 10^-20 до 10^3
    h_values = np.logspace(-20, 3, 1000)
    errors = []
    min_error = float('inf')
    optimal_h = None

    for h in h_values:
        # Запобіжник від ділення на 0 при екстремально малих h
        if h == 0: continue

        num_der = central_difference(M, t0, h)
        err = abs(num_der - exact_val)
        errors.append(err)

        if err < min_error:
            min_error = err
            optimal_h = h

    # Перетворюємо список похибок у numpy масив для графіка
    errors = np.array(errors)

    print(f"\nКрок 2. Оптимальний крок h0: {optimal_h:.2e}")
    print(f"Досягнута точність R0: {min_error:.8e}")

    # Кроки 3-6. Метод Рунге-Ромберга
    h_base = 10 ** (-3)  # Приймаємо h = 10^-3 згідно з методичкою
    print(f"\nКрок 3. Приймаємо h = {h_base}")

    der_h = central_difference(M, t0, h_base)
    der_2h = central_difference(M, t0, 2 * h_base)

    R1 = abs(der_h - exact_val)
    print(f"Крок 4-5. Похідна з кроком h: {der_h:.8f}, Похибка R1: {R1:.8e}")

    # Формула Рунге-Ромберга
    der_RR = der_h + (der_h - der_2h) / 3
    R2 = abs(der_RR - exact_val)
    print(f"Крок 6. Рунге-Ромберг (уточнене): {der_RR:.8f}, Похибка R2: {R2:.8e}")
    print(f"Характер зміни похибки: Похибка зменшилась у {R1 / R2:.2f} разів порівняно з R1.")

    # Крок 7. Метод Ейткена
    der_4h = central_difference(M, t0, 4 * h_base)

    # Формула Ейткена
    numerator_E = der_2h ** 2 - der_4h * der_h
    denominator_E = 2 * der_2h - (der_4h + der_h)
    der_Eitken = numerator_E / denominator_E

    # Оцінка порядку точності p
    numerator_p = der_4h - der_2h
    denominator_p = der_2h - der_h
    p_approx = (1 / math.log(2)) * math.log(abs(numerator_p / denominator_p))

    R3 = abs(der_Eitken - exact_val)

    print(f"\nКрок 7. Метод Ейткена (уточнене): {der_Eitken:.8f}")
    print(f"Оцінка порядку точності p: {p_approx:.2f}")
    print(f"Похибка R3: {R3:.8e}")

    # Висновки щодо режиму поливу
    print("\n--- Висновок щодо режиму поливу ---")
    if exact_val < 0:
        print("Похідна від'ємна. Це означає, що швидкість зміни вологості падає (ґрунт висихає).")
        print(
            "Оскільки швидкість висихання становить близько -1.82 одиниць вологості за одиницю часу, системі автоматичного поливу слід готуватися до увімкнення, якщо поточна вологість наближається до критичного мінімуму.")
    else:
        print("Похідна додатна. Ґрунт накопичує вологу.")

    # Виведення графіків
    print("\nГенерація графіків...")
    plot_moisture(t_start=0, t_end=15)
    plot_error(h_values, errors, optimal_h, min_error)


# Точка запуску програми
if __name__ == "__main__":
    main()