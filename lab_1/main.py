import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. ВХІДНІ ДАНІ
# Координати маршруту (Lat, Lon)
route_coords = [
    (48.164214, 24.536044), (48.164983, 24.534836), (48.165605, 24.534068),
    (48.166228, 24.532915), (48.166777, 24.531927), (48.167326, 24.530884),
    (48.167011, 24.530061), (48.166053, 24.528039), (48.166655, 24.526064),
    (48.166497, 24.523574), (48.166128, 24.520214), (48.165416, 24.517170),
    (48.164546, 24.514640), (48.163412, 24.512980), (48.162331, 24.511715),
    (48.162015, 24.509462), (48.162147, 24.506932), (48.161751, 24.504244),
    (48.161197, 24.501793), (48.160580, 24.500537), (48.160250, 24.500106)
]


# 2. ОТРИМАННЯ ВИСОТ (API або Резерв)
def get_elevation_data(coords):
    # Формуємо запит
    locations_str = "|".join([f"{lat},{lon}" for lat, lon in coords])
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations_str}"

    try:
        print("Запит до API Open-Elevation...")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            elevations = [r['elevation'] for r in data['results']]
            print("Дані отримано успішно.")
            return elevations
    except Exception as e:
        print(f"Помилка API ({e}). Використовуються резервні дані.")

    # Резервні висоти (якщо сайт не працює)
    return [1285, 1310, 1345, 1380, 1420, 1460, 1490, 1530, 1580,
            1650, 1720, 1780, 1850, 1900, 1950, 1980, 2010, 2030,
            2045, 2055, 2061]


elevations = get_elevation_data(route_coords)


#  3. ОБРОБКА ДАНИХ ТА ЗАПИС У ФАЙЛ
# Формула Гаверсинуса для відстані
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Розрахунок кумулятивної відстані (x_nodes)
x_nodes = [0]
for i in range(1, len(route_coords)):
    dist = haversine(route_coords[i - 1][0], route_coords[i - 1][1],
                     route_coords[i][0], route_coords[i][1])
    x_nodes.append(x_nodes[-1] + dist)

y_nodes = np.array(elevations)
x_nodes = np.array(x_nodes)

# Запис у файл
with open("lab1_results.txt", "w", encoding="utf-8") as f:
    f.write("Index | Latitude | Longitude | Dist(m) | Elev(m)\n")
    for i in range(len(route_coords)):
        f.write(f"{i:2d} | {route_coords[i][0]:.6f} | {route_coords[i][1]:.6f} | "
                f"{x_nodes[i]:.2f} | {y_nodes[i]:.2f}\n")
print("Результати табуляції збережено у 'lab1_results.txt'")


#  4. КУБІЧНИЙ СПЛАЙН (МЕТОД ПРОГОНКИ)
def compute_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    # Ініціалізація масивів для методу прогонки
    alpha = np.zeros(n)
    beta = np.zeros(n)
    c = np.zeros(n + 1)  # коефіцієнти кривизни

    # Пряма прогонка
    # Гранична умова c_0 = 0
    alpha[0] = 0
    beta[0] = 0

    for i in range(1, n):
        A = h[i - 1]
        B = 2 * (h[i - 1] + h[i])
        D = h[i]
        F = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        denom = A * alpha[i - 1] + B
        alpha[i] = -D / denom
        beta[i] = (F - A * beta[i - 1]) / denom

    # Зворотна прогонка
    c[n] = 0  # Гранична умова c_n = 0
    for i in range(n - 1, 0, -1):
        c[i] = alpha[i] * c[i + 1] + beta[i]

    # Обчислення інших коефіцієнтів a, b, d
    a = y[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] * (c[i + 1] + 2 * c[i])) / 3

    return a, b, c[:-1], d


N = 10  #Змінна кількість точок на графіку

# np.linspace рівномірно генерує індекси (номери) точок від першої до останньої.
# astype(int) робить ці номери цілими числами (бо точка не може бути під номером 2.5)
indices = np.linspace(0, len(x_nodes) - 1, N).astype(int)

# Вирізаємо з наших повних даних тільки вибрані N точок
x_nodes_selected = np.array(x_nodes)[indices]
y_nodes_selected = np.array(y_nodes)[indices]

a, b, c, d = compute_spline(x_nodes_selected, y_nodes_selected)

# Вивід коефіцієнтів у консоль
print("\nКоефіцієнти сплайнів (фрагмент):")
print(f"{'i':<3} | {'a':<8} | {'b':<8} | {'c':<8} | {'d':<8}")
print("-" * 45)
for i in range(min(10, len(a))):  # Виводимо перші 10 для прикладу
    print(f"{i + 1:<3} | {a[i]:<8.2f} | {b[i]:<8.4f} | {c[i]:<8.6f} | {d[i]:<8.8f}")


# Функція S(x) для побудови графіка
def spline_eval(x_val, x_nodes_selected, a, b, c, d):
    if x_val < x_nodes_selected[0] or x_val > x_nodes_selected[-1]: return 0
    idx = np.searchsorted(x_nodes_selected, x_val) - 1
    idx = max(0, min(idx, len(a) - 1))
    dx = x_val - x_nodes_selected[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


# 5. ВІЗУАЛІЗАЦІЯ ТА АНАЛІЗ
# Генерація точок для гладкої лінії
x_smooth = np.linspace(x_nodes_selected[0], x_nodes_selected[-1], 500)
y_smooth = [spline_eval(xi, x_nodes_selected, a, b, c, d) for xi in x_smooth]

# Розрахунок характеристик
total_dist = x_nodes_selected[-1]
total_ascent = sum(max(y_nodes_selected[i] - y_nodes_selected[i - 1], 0) for i in range(1, len(y_nodes_selected)))
total_descent = sum(max(y_nodes_selected[i - 1] - y_nodes_selected[i], 0) for i in range(1, len(y_nodes_selected)))

# Розрахунок енергії та калорій
mass = 80  # кг
g = 9.81
work_joules = mass * g * total_ascent
kcal = work_joules / 4184

print("\n--- Аналіз маршруту ---")
print(f"Довжина маршруту: {total_dist:.2f} м")
print(f"Набір висоти: {total_ascent:.2f} м")
print(f"Спуск: {total_descent:.2f} м")
print(f"Механічна робота: {work_joules / 1000:.2f} кДж")
print(f"Енерговитрати: {kcal:.2f} ккал")

# --- РОЗРАХУНОК ПОХИБКИ ---
# Обчислюємо значення сплайна для ВСІХ початкових точок x_nodes
y_pred = np.array([spline_eval(xi, x_nodes_selected, a, b, c, d) for xi in x_nodes])

# Знаходимо абсолютну різницю (похибку) між реальними та передбаченими висотами
errors = np.abs(y_nodes - y_pred)

# Рахуємо максимальну та середню абсолютну похибку (MAE - Mean Absolute Error)
max_error = np.max(errors)
mean_error = np.mean(errors)

# --- ВІЗУАЛІЗАЦІЯ: МАРШРУТ ТА ПОХИБКИ ---

# Створюємо одне вікно, розділене на два графіки (ax1 - верхній, ax2 - нижній)
# gridspec_kw={'height_ratios': [3, 1]} робить верхній графік у 3 рази вищим за нижній
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]})

# 1. ВЕРХНІЙ ГРАФІК: Профіль висоти
ax1.plot(x_nodes_selected, y_nodes_selected, 'ro', label='GPS Вузли (опорні)', markersize=6)
ax1.plot(x_smooth, y_smooth, 'b-', label='Кубічний сплайн', linewidth=2)
ax1.fill_between(x_smooth, y_smooth, alpha=0.1, color='blue')

ax1.set_title(f'Профіль висоти маршруту "Заросляк - Говерла"\nЕнерговитрати: ~{kcal:.0f} ккал | Середня похибка: {mean_error:.2f} м')
ax1.set_ylabel('Висота (м)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# 2. НИЖНІЙ ГРАФІК: Похибки
# Будуємо графік похибок червоною лінією з крапками
ax2.plot(x_nodes, errors, color='red', marker='o', linestyle='-', linewidth=1.5, markersize=4, label='Абсолютна похибка')

# Заливаємо площу під графіком похибок для наочності
ax2.fill_between(x_nodes, errors, alpha=0.2, color='red')

ax2.set_xlabel('Відстань (м)')
ax2.set_ylabel('Похибка (м)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# Вирівнюємо графіки, щоб вони не налізали один на одного
plt.tight_layout()
plt.show()

print("\n--- Аналіз похибки інтерполяції ---")
print(f"Сплайн побудовано по {N} точках з {len(x_nodes)} доступних.")
print(f"Максимальна похибка: {max_error:.2f} м")
print(f"Середня похибка (MAE): {mean_error:.2f} м")

#  АНАЛІЗ ГРАДІЄНТА
# Рахуємо градієнт (похідну) для згладжених точок і множимо на 100 для відсотків
grad_full = np.gradient(y_smooth, x_smooth) * 100

max_ascent_grad = np.max(grad_full)
max_descent_grad = np.min(grad_full)
avg_grad = np.mean(np.abs(grad_full))

print("\n--- Аналіз градієнта (нахилу стежки) ---")
print(f"Максимальний підйом: {max_ascent_grad:.2f} %")
print(f"Максимальний спуск: {max_descent_grad:.2f} %")
print(f"Середній градієнт: {avg_grad:.2f} %")

# Шукаємо екстремальні ділянки з крутизною > 15%
steep_sections = grad_full > 15

# Якщо такі ділянки є, знайдемо, на яких саме метрах маршруту вони знаходяться
if np.any(steep_sections):
    print("Увага: Знайдено дуже стрімкі ділянки з крутизною > 15%!")
    # Витягуємо відстані (x_smooth), де умова steep_sections є істинною (True)
    steep_distances = np.array(x_smooth)[steep_sections]

    # Виводимо початок і кінець найважчої зони для прикладу
    print(f"Найважчі відрізки починаються приблизно на {steep_distances[0]:.0f} м "
          f"і трапляються аж до {steep_distances[-1]:.0f} м маршруту.")
else:
    print("Ділянок з крутизною > 15% не виявлено. Підйом відносно пологий.")

# Графіки
plt.figure(figsize=(12, 6))
plt.plot(x_nodes_selected, y_nodes_selected, 'ro', label='GPS Вузли', markersize=6)
plt.plot(x_smooth, y_smooth, 'b-', label='Кубічний сплайн', linewidth=2)
plt.fill_between(x_smooth, y_smooth, alpha=0.1, color='blue')
plt.title(f'Профіль висоти маршруту "Заросляк - Говерла"\nЕнерговитрати: ~{kcal:.0f} ккал')
plt.xlabel('Відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
