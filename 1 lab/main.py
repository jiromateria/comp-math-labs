import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. ФУНКЦИИ И ИХ АНАЛИТИЧЕСКИЕ ПРОИЗВОДНЫЕ ---
# Обратите внимание: здесь мы используем обычные строки, 
# символы $ для формул добавим уже при построении графиков.
def f1(x): return np.sin(x**2)
def df1(x): return 2 * x * np.cos(x**2)

def f2(x): return np.cos(np.sin(x))
def df2(x): return -np.sin(np.sin(x)) * np.cos(x)

def f3(x): return np.exp(np.sin(np.cos(x)))
def df3(x): return np.exp(np.sin(np.cos(x))) * np.cos(np.cos(x)) * (-np.sin(x))

def f4(x): return np.log(x + 3)
def df4(x): return 1 / (x + 3)

def f5(x): return (x + 3)**0.5
def df5(x): return 0.5 / np.sqrt(x + 3)

# --- 2. МЕТОДЫ ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ ---
def method1(f, x, h): 
    return (f(x + h) - f(x)) / h

def method2(f, x, h): 
    return (f(x) - f(x - h)) / h

def method3(f, x, h): 
    return (f(x + h) - f(x - h)) / (2 * h)

def method4(f, x, h): 
    return (4/3) * (f(x + h) - f(x - h)) / (2 * h) - (1/3) * (f(x + 2*h) - f(x - 2*h)) / (4 * h)

def method5(f, x, h): 
    return (3/2) * (f(x + h) - f(x - h)) / (2 * h) - (3/5) * (f(x + 2*h) - f(x - 2*h)) / (4 * h) + (1/10) * (f(x + 3*h) - f(x - 3*h)) / (6 * h)

# --- 3. ОСНОВНАЯ ЛОГИКА ---
def main():
    if not os.path.exists('pictures'):
        os.makedirs('pictures')

    x0 = 1.0 
    
    # ИЗМЕНЕНИЕ 1: Делаем больше точек (60 штук вместо 21) для плавности графика.
    # np.linspace(1, 21, 60) создает 60 равномерных значений от 1 до 21.
    # Если преподаватель строго требует ровно 21 точку, замените эту строку на: n_values = np.arange(1, 22)
    n_values = np.linspace(1, 21, 60) 
    
    h_values = 2.0 / (2.0 ** n_values)

    # ИЗМЕНЕНИЕ 2: Оборачиваем формулы в символы $ для правильного отображения
    functions = [
        (f1, df1, r'$f(x) = \sin(x^2)$', 1),
        (f2, df2, r'$f(x) = \cos(\sin(x))$', 2),
        (f3, df3, r'$f(x) = \exp(\sin(\cos(x)))$', 3),
        (f4, df4, r'$f(x) = \ln(x+3)$', 4),
        (f5, df5, r'$f(x) = (x+3)^{0.5}$', 5)
    ]

    methods = [
        (method1, 'Метод 1 (Вперед)'),
        (method2, 'Метод 2 (Назад)'),
        (method3, 'Метод 3 (Центральный)'),
        (method4, 'Метод 4 (4-й порядок)'),
        (method5, 'Метод 5 (6-й порядок)')
    ]

    # Настройки шрифтов
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # Чтобы русские буквы не квадратиками были
    plt.rcParams['mathtext.fontset'] = 'cm' 

    for func, dfunc, title_formula, idx in functions:
        plt.figure(figsize=(10, 6))
        exact_val = dfunc(x0)

        for method, method_name in methods:
            errors = []
            for h in h_values:
                num_val = method(func, x0, h)
                error = np.abs(num_val - exact_val)
                errors.append(error)
            
            # ИЗМЕНЕНИЕ 3: Увеличили размер маркеров (ms) и толщину линий (lw)
            # marker='o' (кружочки) смотрится аккуратнее точек
            plt.loglog(h_values, errors, marker='o', markersize=4, linewidth=2, label=method_name, alpha=0.8)

        # ИЗМЕНЕНИЕ 4: Правильные символы $ для заголовка и осей
        plt.title(f'Абсолютная погрешность {title_formula}', fontsize=14)
        plt.xlabel(r'Шаг дифференцирования $h$', fontsize=12)
        plt.ylabel(r'Абсолютная погрешность $|f^{\prime}_{num} - f^{\prime}_{exact}|$', fontsize=12)
        
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.legend(fontsize=10, loc='best')
        
        filename = f'pictures/plot_{idx}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f'Сохранен график: {filename}')

if __name__ == '__main__':
    main()
    print('Все графики успешно перестроены!')