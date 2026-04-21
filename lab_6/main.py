import numpy as np

# 1. Генерація та збереження початкових даних
def generate_and_save_data(n=100, file_A="matrix_A.txt", file_B="vector_B.txt"):
    """
    Генерує випадкову матрицю A та вектор B на основі заданого розв'язку x_j = 2.5
    і зберігає їх у текстові файли.
    """
    # Генеруємо елементи матриці A (для наочності множимо на 10)
    A = np.random.rand(n, n) * 10
    np.savetxt(file_A, A, fmt='%.6f')

    # Задаємо точний розв'язок згідно з методичкою
    x_true = np.full(n, 2.5)

    # Обчислюємо вектор вільних членів b_i = sum(a_ij * x_j)
    B = np.zeros(n)
    for i in range(n):
        B[i] = sum(A[i][j] * x_true[j] for j in range(n))

    np.savetxt(file_B, B, fmt='%.6f')
    return x_true

# 2. Опис необхідних функцій

def read_matrix_and_vector(file_A="matrix_A.txt", file_B="vector_B.txt"):
    """Зчитування матриці А та вектора В з текстового файлу."""
    A = np.loadtxt(file_A)
    B = np.loadtxt(file_B)
    return A, B


def lu_decomposition(A):
    """
    Знаходження LU-розкладу матриці А.
    Повертає нижню (L) та верхню (U) трикутні матриці.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Задаємо значення діагональних елементів матриці U рівними одиниці
    for i in range(n):
        U[i][i] = 1.0

    # Почергово знаходимо елементи k-го стовпця L та k-го рядка U
    for k in range(n):
        # 1. Знаходимо елементи стовпця матриці L (i >= k)
        for i in range(k, n):
            sum_lu = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_lu

        # 2. Знаходимо елементи рядка матриці U (i > k)
        for i in range(k + 1, n):
            # Захист від ділення на нуль
            if L[k][k] == 0:
                raise ValueError("Нульовий елемент на діагоналі L, розклад неможливий без перестановки.")
            sum_lu = sum(L[k][j] * U[j][i] for j in range(k))
            U[k][i] = (A[k][i] - sum_lu) / L[k][k]

    return L, U


def save_lu_to_file(L, U, filename="matrix_LU.txt"):
    """Запис LU-розкладу матриці А в текстовий файл."""
    with open(filename, 'w') as f:
        f.write("--- Matrix L ---\n")
        np.savetxt(f, L, fmt='%.6f')
        f.write("\n--- Matrix U ---\n")
        np.savetxt(f, U, fmt='%.6f')


def solve_slae_lu(L, U, B):
    """Розв'язок системи рівнянь AX = B за допомогою LU-розкладу."""
    n = len(B)
    Z = np.zeros(n)
    X = np.zeros(n)

    # 1. Розв'язуємо систему L*Z = B (прямий хід)
    for k in range(n):
        sum_lz = sum(L[k][j] * Z[j] for j in range(k))
        Z[k] = (B[k] - sum_lz) / L[k][k]

    # 2. Розв'язуємо систему U*X = Z (зворотний хід)
    for k in range(n - 1, -1, -1):
        sum_ux = sum(U[k][j] * X[j] for j in range(k + 1, n))
        X[k] = Z[k] - sum_ux

    return X


def multiply_matrix_vector(A, X):
    """Обчислення добутку матриці на вектор."""
    n = len(A)
    result = np.zeros(n)
    for i in range(n):
        result[i] = sum(A[i][j] * X[j] for j in range(n))
    return result


def calculate_vector_norm(V):
    """Обчислення норми вектора (максимальний за модулем елемент)."""
    return np.max(np.abs(V))

# Головний блок виконання
def main():
    n = 100
    eps_0 = 1e-14  # Задана точність

    # Підготовка даних
    print(f"Генерація матриці розміром {n}x{n} та вектора вільних членів...")
    generate_and_save_data(n)

    # Зчитування та розв'язок за допомогою LU
    A, B = read_matrix_and_vector()
    L, U = lu_decomposition(A)
    save_lu_to_file(L, U)

    # Знаходимо початкове наближення X^(0)
    X_0 = solve_slae_lu(L, U, B)

    # Оцінка точності знайденого розв'язку
    # Обчислюємо B^(0) = A * X^(0)
    B_0 = multiply_matrix_vector(A, X_0)

    # Обчислюємо початкову похибку (норму вектора нев'язки)
    initial_residual_vector = B_0 - B
    eps_initial = calculate_vector_norm(initial_residual_vector)
    print(f"Початкова максимальна похибка (до уточнення): {eps_initial:.6e}")

    # Ітераційний метод уточнення розв'язку
    print("\nПочаток ітераційного уточнення розв'язку...")
    X_current = np.copy(X_0)
    iteration = 0

    while True:
        iteration += 1

        # Обчислюємо B поточне = A * X_current
        B_current = multiply_matrix_vector(A, X_current)

        # Обчислюємо вектор нев'язки: R = B - B_current
        R = B - B_current

        # Розв'язуємо систему A * delta_X = R через вже знайдений LU-розклад
        delta_X = solve_slae_lu(L, U, R)

        # Обчислюємо уточнене значення: X = X + delta_X
        X_current = X_current + delta_X

        # Перевіряємо умови закінчення ітераційної процедури
        norm_delta_X = calculate_vector_norm(delta_X)
        norm_residual_AX_B = calculate_vector_norm(multiply_matrix_vector(A, X_current) - B)

        print(f"Ітерація {iteration}: норма delta_X = {norm_delta_X:.4e}, норма нев'язки = {norm_residual_AX_B:.4e}")

        if norm_delta_X <= eps_0 and norm_residual_AX_B <= eps_0:
            print(f"\n--> УСПІХ! Задану точність (<= {eps_0}) досягнуто за {iteration} ітерацій.")
            break

        # Захист від нескінченного циклу через межі точності типу float64 у Python
        if iteration > 20:
            print(
                "\n--> УВАГА: Досягнуто межі машинної точності типу float64. Алгоритм зупинено щоб уникнути зациклення.")
            break


if __name__ == "__main__":
    main()