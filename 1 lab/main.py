import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. ФУНКЦИИ И ИХ АНАЛИТИЧЕСКИЕ ПРОИЗВОДНЫЕ ---
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
# Формулы строго по заданию
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
    # Создаем папку для картинок, если ее нет
    if not os.path.exists('pictures'):
        os.makedirs('pictures')

    # Точка, в которой считаем производную (можно выбрать любую, например 1.0)
    x0 = 1.0 
    
    # Список шагов h_n = 2 / 2^n, от n=1 до 21
    n_values = np.arange(1, 22)
    h_values = 2.0 / (2.0 ** n_values)

    # Список функций: (функция, производная, название, индекс)
    functions = [
        (f1, df1, r'\(f(x) = \sin(x^2)\)', 1),
        (f2, df2, r'\(f(x) = \cos(\sin(x))\)', 2),
        (f3, df3, r'\(f(x) = \exp(\sin(\cos(x)))\)', 3),
        (f4, df4, r'\(f(x) = \ln(x+3)\)', 4),
        (f5, df5, r'\(f(x) = (x+3)^{0.5}\)', 5)
    ]

    methods = [
        (method1, 'Метод 1 (Вперед)'),
        (method2, 'Метод 2 (Назад)'),
        (method3, 'Метод 3 (Центральный)'),
        (method4, 'Метод 4 (4-й порядок)'),
        (method5, 'Метод 5 (6-й порядок)')
    ]

    # Включаем красивый шрифт для графиков
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'cm' # Шрифт как в LaTeX для формул

    for func, dfunc, title, idx in functions:
        plt.figure(figsize=(10, 6))
        
        # Идеальное (точное) значение производной
        exact_val = dfunc(x0)

        # Считаем и строим графики для каждого метода
        for method, method_name in methods:
            errors = []
            for h in h_values:
                # Численное значение
                num_val = method(func, x0, h)
                # Абсолютная погрешность
                error = np.abs(num_val - exact_val)
                errors.append(error)
            
            # Строим график в логарифмическом масштабе по обеим осям
            plt.loglog(h_values, errors, marker='.', label=method_name, linewidth=2)

        # --- Красивое оформление графика ---
        plt.title(f'Абсолютная погрешность {title}', fontsize=14)
        plt.xlabel(r'Шаг дифференцирования \(h\)', fontsize=12)
        plt.ylabel(r'Абсолютная погрешность \(|f\'_{num} - f\'_{exact}|\)', fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.7) # Сетка для лог. шкалы
        plt.legend(fontsize=10)
        
        # Сохранение графика
        filename = f'pictures/plot_{idx}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300) # dpi=300 для хорошего качества
        plt.close() # Закрываем окно, чтобы не забивать память
        print(f'Сохранен график: {filename}')

if __name__ == '__main__':
    main()
    print('Все графики успешно построены и сохранены в папку pictures!')