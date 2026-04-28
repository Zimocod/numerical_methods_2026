import random

# Генерація даних та підготовка
def generate_data(n=100, exact_val=2.5, file_A="A.txt", file_B="B.txt"):
    """
    Генерує матрицю А розмірності n x n з діагональним переважанням.
    Задає розв'язок xi = 2.5 та обчислює вектор вільних членів B.
    Зберігає результати у текстові файли.
    """
    A = [[0.0] * n for _ in range(n)]

    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i != j:
                # Генеруємо випадкові недіагональні елементи
                A[i][j] = random.uniform(-10, 10)
                row_sum += abs(A[i][j])
        # Забезпечуємо діагональне переважання: діагональний елемент > суми інших
        A[i][i] = row_sum + random.uniform(1, 10)

    # Обчислення вектора B: b_i = sum(a_ij * x_j)
    B = [0.0] * n
    for i in range(n):
        for j in range(n):
            B[i] += A[i][j] * exact_val

    # Запис у файли
    with open(file_A, 'w') as fa:
        for row in A:
            fa.write(" ".join(f"{val:.6f}" for val in row) + "\n")

    with open(file_B, 'w') as fb:
        for val in B:
            fb.write(f"{val:.6f}\n")

    print("Дані успішно згенеровано та записано у файли A.txt та B.txt.")

# Допоміжні функції

def read_matrix(filename):
    """Зчитування матриці А з текстового файлу."""
    A = []
    with open(filename, 'r') as f:
        for line in f:
            row = [float(x) for x in line.split()]
            A.append(row)
    return A


def read_vector(filename):
    """Зчитування вектора В з текстового файлу."""
    B = []
    with open(filename, 'r') as f:
        for line in f:
            B.append(float(line.strip()))
    return B


def mat_vec_mult(A, X):
    """Обчислення добутку матриці на вектор."""
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * X[j]
    return result


def vec_norm(X1, X2):
    """
    Обчислення норми вектора (різниці між двома векторами).
    Використовуємо нескінченну норму (максимальний модуль різниці),
    щоб перевіряти умову ||X^(k+1) - X^(k)|| < eps.
    """
    return max(abs(x1 - x2) for x1, x2 in zip(X1, X2))


def mat_norm(A):
    """
    Обчислення норми матриці.
    Використовуємо норму ||C||_3 (максимальна сума модулів по рядках).
    """
    n = len(A)
    max_sum = 0
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n))
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum

# Ітераційні методи

def simple_iteration_method(A, B, eps, max_iter=10000):
    """Розв'язок СЛАР методом простої ітерації."""
    n = len(A)
    # Початкове наближення xi = 1.0
    X = [1.0] * n

    # Визначаємо параметр tau з умови 0 < tau < 2/||A||
    norm_A = mat_norm(A)
    tau = 1.0 / norm_A

    iterations = 0
    while iterations < max_iter:
        iterations += 1
        X_new = [0.0] * n

        # Обчислюємо AX
        AX = mat_vec_mult(A, X)

        # Застосовуємо формулу X^(k+1) = X^(k) - tau * (AX - B)
        # Це еквівалентно X^(k+1) = (E - tau*A)X^(k) + tau*B
        for i in range(n):
            X_new[i] = X[i] - tau * (AX[i] - B[i])

        # Перевірка умови закінчення
        if vec_norm(X_new, X) < eps:
            return X_new, iterations

        X = X_new.copy()

    return X, iterations


def jacobi_method(A, B, eps, max_iter=10000):
    """Розв'язок СЛАР методом Якобі."""
    n = len(A)
    # Початкове наближення xi = 1.0
    X = [1.0] * n

    iterations = 0
    while iterations < max_iter:
        iterations += 1
        X_new = [0.0] * n

        for i in range(n):
            sum_ax = 0
            for j in range(n):
                if i != j:
                    sum_ax += A[i][j] * X[j]
            # Формула методу Якобі
            X_new[i] = (B[i] - sum_ax) / A[i][i]

        # Перевірка умови закінчення
        if vec_norm(X_new, X) < eps:
            return X_new, iterations

        X = X_new.copy()

    return X, iterations


def seidel_method(A, B, eps, max_iter=10000):
    """Розв'язок СЛАР методом Зейделя."""
    n = len(A)
    # Початкове наближення xi = 1.0
    X = [1.0] * n

    iterations = 0
    while iterations < max_iter:
        iterations += 1
        X_old = X.copy()

        for i in range(n):
            sum_ax = 0
            # Для j < i використовуємо вже оновлені значення (X_new),
            # а для j > i - старі значення.
            for j in range(n):
                if i != j:
                    sum_ax += A[i][j] * X[j]
            X[i] = (B[i] - sum_ax) / A[i][i]

        # Перевірка умови закінчення
        if vec_norm(X, X_old) < eps:
            return X, iterations

    return X, iterations

# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
if __name__ == "__main__":
    n_size = 100
    eps_0 = 10 ** (-14)  # Задана точність

    # Генерація
    generate_data(n=n_size)

    # Зчитування
    A = read_matrix("A.txt")
    B = read_vector("B.txt")

    print(f"\nРозв'язок СЛАР {n_size}x{n_size} з точністю {eps_0}:")
    print("-" * 50)

    # Знаходження уточненого розв'язку та кількості ітерацій

    # 1. Проста ітерація
    X_simp, iter_simp = simple_iteration_method(A, B, eps_0)
    print(f"Метод простої ітерації:")
    print(f"  Ітерацій: {iter_simp}")
    print(f"  Приклад перших 3 коренів: {X_simp[:3]}")

    # 2. Метод Якобі
    X_jacobi, iter_jacobi = jacobi_method(A, B, eps_0)
    print(f"\nМетод Якобі:")
    print(f"  Ітерацій: {iter_jacobi}")
    print(f"  Приклад перших 3 коренів: {X_jacobi[:3]}")

    # 3. Метод Зейделя
    X_seidel, iter_seidel = seidel_method(A, B, eps_0)
    print(f"\nМетод Зейделя:")
    print(f"  Ітерацій: {iter_seidel}")
    print(f"  Приклад перших 3 коренів: {X_seidel[:3]}")
    print("-" * 50)
    print("Усі методи повинні наближатися до точного розв'язку x_i = 2.5")