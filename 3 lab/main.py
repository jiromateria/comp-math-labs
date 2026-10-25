import numpy as np

# =========================================================
# БЛОК 1: УНИВЕРСАЛЬНЫЕ ЧИСЛЕННЫЕ МЕТОДЫ
# =========================================================

def find_intervals_1d(f, x_start: float, x_end: float, step: float) -> list:
    """Численный поиск отрезков [a, b], где функция f(x) меняет знак."""
    intervals = []
    x = x_start
    while x < x_end:
        x_next = x + step
        # Если функция меняет знак или попадает точно в 0
        if f(x) * f(x_next) < 0 or f(x) == 0:
            intervals.append((x, x_next))
        x = x_next
    return intervals


def find_initial_guess_2d(F, x_range: tuple, y_range: tuple, grid_steps: int = 100) -> list:
    """Численный поиск начального приближения для системы F(x,y) = 0."""
    best_point = None
    min_residual = float('inf')
    
    x_vals = np.linspace(x_range[0], x_range[1], grid_steps)
    y_vals = np.linspace(y_range[0], y_range[1], grid_steps)
    
    for x in x_vals:
        for y in y_vals:
            try:
                f1, f2 = F([x, y])
                residual = f1**2 + f2**2 # Сумма квадратов невязок
                if residual < min_residual:
                    min_residual = residual
                    best_point = [x, y]
            except (ValueError, ZeroDivisionError):
                # Игнорируем точки, где функция не определена (например, log(x) при x <= 0)
                continue
                
    return best_point


def bisection(f, a: float, b: float, tol: float = 1e-5, max_iter: int = 100) -> tuple:
    """Метод половинного деления (бисекции)."""
    # ИСПРАВЛЕНИЕ: Если один из концов уже является корнем с заданной точностью
    if abs(f(a)) < tol:
        return a, 0
    if abs(f(b)) < tol:
        return b, 0
        
    if f(a) * f(b) > 0:
        raise ValueError("Функция должна иметь разные знаки на концах отрезка")
    
    c = a
    for i in range(max_iter):
        c = (a + b) / 2.0
        if abs(f(c)) < tol or (b - a) / 2.0 < tol:
            return c, i + 1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, max_iter


def newton_1d(f, df, x0: float, tol: float = 1e-5, max_iter: int = 100) -> tuple:
    """Метод Ньютона для одного уравнения f(x) = 0."""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, i + 1
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Производная равна нулю, метод Ньютона не применим")
        x = x - fx / dfx
    return x, max_iter


def newton_system(F, J, x0: list, tol: float = 1e-5, max_iter: int = 100) -> tuple:
    """Метод Ньютона для системы уравнений F(X) = 0."""
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        Fx = np.array(F(x))
        if np.linalg.norm(Fx) < tol: 
            return x, i + 1
        
        Jx = np.array(J(x))
        try:
            delta = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            print("  [!] Ошибка: Матрица Якоби вырождена, метод не сходится.")
            break
            
        x = x + delta
    return x, max_iter


# =========================================================
# БЛОК 2: РЕШЕНИЕ ЗАДАНИЯ IV.12.3 (Уравнения)
# =========================================================
print("="*60)
print("ЗАДАНИЕ IV.12.3 (Нелинейные уравнения)")
print("="*60)

# --- Задача а) 2x^2 + 5x - 3 = 0 ---
f_a = lambda x: 2*x**2 + 5*x - 3
df_a = lambda x: 4*x + 5

print("\nЗадача а): 2x^2 + 5x - 3 = 0")
intervals_a = find_intervals_1d(f_a, -10, 10, 0.5)
print(f"  Найденные отрезки с корнями: {intervals_a}")

if intervals_a:
    # Берем первый отрезок
    a, b = intervals_a[0]
    root_bis, it_bis = bisection(f_a, a, b, tol=1e-5)
    root_new, it_new = newton_1d(f_a, df_a, (a+b)/2, tol=1e-5)
    print(f"  Уточнение на отрезке [{a}, {b}]:")
    print(f"    Метод бисекции: x = {root_bis:.6f} (итераций: {it_bis})")
    print(f"    Метод Ньютона:  x = {root_new:.6f} (итераций: {it_new})")

# --- Задача е) x^2 - 20*sin(x) = 0 ---
f_e = lambda x: x**2 - 20*np.sin(x)
df_e = lambda x: 2*x - 20*np.cos(x)

print("\nЗадача е): x^2 - 20*sin(x) = 0")
intervals_e = find_intervals_1d(f_e, -10, 10, 0.5)
print(f"  Найденные отрезки с корнями: {intervals_e}")

if intervals_e:
    # Берем отрезок, где корень ненулевой (второй по счету)
    a, b = intervals_e[1] if len(intervals_e) > 1 else intervals_e[0]
    root_bis, it_bis = bisection(f_e, a, b, tol=1e-5)
    root_new, it_new = newton_1d(f_e, df_e, (a+b)/2, tol=1e-5)
    print(f"  Уточнение на отрезке [{a}, {b}]:")
    print(f"    Метод бисекции: x = {root_bis:.6f} (итераций: {it_bis})")
    print(f"    Метод Ньютона:  x = {root_new:.6f} (итераций: {it_new})")


# =========================================================
# БЛОК 3: РЕШЕНИЕ СИСТЕМ УРАВНЕНИЙ (IV.12.4 и IV.12.6)
# =========================================================
print("\n" + "="*60)
print("ЗАДАНИЕ IV.12.4 и IV.12.6 (Системы уравнений)")
print("="*60)

# --- IV.12.4 а) ---
def F_12_4a(X):
    x, y = X
    return [np.sin(x + 1) - y - 1.2, 2*x + np.cos(y) - 2]

def J_12_4a(X):
    x, y = X
    return [[np.cos(x + 1), -1],
            [2, -np.sin(y)]]

print("\nЗадача IV.12.4 а) (Точность 10^-3)")
guess_12_4 = find_initial_guess_2d(F_12_4a, (-5, 5), (-5, 5), grid_steps=50)
print(f"  Численный поиск нач. приближения: x={guess_12_4[0]:.2f}, y={guess_12_4[1]:.2f}")

res_12_4, it_12_4 = newton_system(F_12_4a, J_12_4a, guess_12_4, tol=1e-3)
print(f"  Ответ: x = {res_12_4[0]:.5f}, y = {res_12_4[1]:.5f} (итераций: {it_12_4})")


# --- IV.12.6 а) ---
def F_12_6a(X):
    x, y = X
    return [2*x**2 - x*y - 5*x + 1, x + 3*np.log10(x) - y**2]

def J_12_6a(X):
    x, y = X
    return [[4*x - y - 5, -x],
            [1 + 3 / (x * np.log(10)), -2*y]]

print("\nЗадача IV.12.6 а) (Точность 10^-5)")
# Поиск в области x > 0, так как есть логарифм
guess_12_6 = find_initial_guess_2d(F_12_6a, (0.1, 5), (-5, 5), grid_steps=50)
print(f"  Численный поиск нач. приближения: x={guess_12_6[0]:.2f}, y={guess_12_6[1]:.2f}")

res_12_6, it_12_6 = newton_system(F_12_6a, J_12_6a, guess_12_6, tol=1e-5)
print(f"  Ответ: x = {res_12_6[0]:.6f}, y = {res_12_6[1]:.6f} (итераций: {it_12_6})")