"""
Лабораторная по вычислительной математике: решение СЛАУ  A x = b.

Вариант "о" (выбран генератором случайных чисел, seed = 473354).
Методы (случайные 3 из списка + прямой метод для сверки):
    * gauss   -- метод Гаусса с выбором главного элемента (прямой, эталон)
    * jacobi  -- метод Якоби
    * mr      -- метод минимальных невязок
    * cg      -- метод сопряжённых градиентов

Архитектура
-----------
Каждый итерационный метод -- это *генератор итераций*: функция, которая по (A, b, x0, **параметры)
отдаёт очередное приближение x_k. Методы регистрируются декоратором @register в словаре METHODS.
Прямые методы -- обычные функции (A, b) -> x, они регистрируются через @register_direct в DIRECT.
Единая функция solve(A, b, method, **params) работает с обоими видами: для итерационных сама считает
невязку, строит историю и проверяет критерий остановки. Чтобы добавить метод (Зейдель, SOR,
градиентный спуск, BiCGStab, LU...), достаточно написать функцию и повесить на неё декоратор.
Решение Гаусса используется как эталон: ошибка итерационных методов считается относительно него,
а сам Гаусс проверяется независимо -- по решению numpy.linalg.solve (LAPACK).

Запуск
------
    python main.py                                   # система из systems/variant_o.json
    python main.py --system poisson:n=50             # одномерная задача Пуассона
    python main.py --system random:n=30,seed=1       # случайная СПД диагонально доминирующая
    python main.py --system my_system.json           # своя СЛАУ из JSON {"A": [[...]], "b": [...]}
    python main.py --methods gauss,cg,jacobi --tol 1e-12
    python main.py --param jacobi.omega=0.8          # параметр метода: имя.параметр=значение
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
#  Задание СЛАУ
# --------------------------------------------------------------------------- #


def random_spd_system(n: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Случайная симметричная система со строгим диагональным преобладанием (=> СПД)."""
    rng = np.random.default_rng(seed)
    M = rng.uniform(-1, 1, size=(n, n))
    A = (M + M.T) / 2
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(n)] = np.abs(A).sum(axis=1) + rng.uniform(0.5, 2.0, size=n)
    x_true = rng.uniform(-3, 3, size=n)
    return A, A @ x_true


def poisson1d_system(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Трёхдиагональная матрица (2, -1): дискретизация -u'' = f. Обусловленность ~ n^2."""
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.ones(n)
    return A, b


DEFAULT_SYSTEM = "systems/variant_o.json"


def variant_o_placeholder() -> tuple[np.ndarray, np.ndarray]:
    """Система-заглушка для варианта «о» (8x8, СПД, диагональное преобладание)."""
    A, b = random_spd_system(8, seed=473354)
    return np.round(A, 4), np.round(b, 4)


def load_system(spec: str) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Загрузка СЛАУ по строке-описанию:
        путь/к/файлу.json         {"A": [[..]], "b": [..], "name": "..."}
        random:n=10,seed=0
        poisson:n=50
    """
    if spec.startswith(("random:", "poisson:")):
        kind, _, args = spec.partition(":")
        kw = dict(item.split("=") for item in args.split(",") if item)
        kw = {k: int(v) for k, v in kw.items()}
        A, b = random_spd_system(**kw) if kind == "random" else poisson1d_system(**kw)
        return A, b, spec
    path = Path(spec)
    if not path.is_absolute() and not path.exists():
        path = HERE / spec
    if not path.exists():
        if spec.replace("\\", "/") == DEFAULT_SYSTEM:
            # первый запуск без папки systems/: создаём файл-заглушку, который можно отредактировать
            A, b = variant_o_placeholder()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"name": "variant_o (ЗАГЛУШКА: замените A и b данными из методички)",
                 "A": A.tolist(), "b": b.tolist()}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[!] Файл {path} не найден -- создан с системой-заглушкой.\n"
                  f"    Замените в нём A и b на данные своего варианта и запустите снова.\n")
        else:
            raise SystemExit(
                f"Файл системы не найден: {path}\n"
                f"Ожидается JSON вида {{\"A\": [[...], ...], \"b\": [...]}}, либо используйте "
                f"random:n=10,seed=0 / poisson:n=50")
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        np.array(data["A"], dtype=float),
        np.array(data["b"], dtype=float),
        data.get("name", path.stem),
    )


# --------------------------------------------------------------------------- #
#  Методы: генераторы итераций
# --------------------------------------------------------------------------- #

METHODS: dict[str, Callable[..., Iterator[np.ndarray]]] = {}  # итерационные: генераторы x_k
DIRECT: dict[str, Callable[..., np.ndarray]] = {}  # прямые: (A, b) -> x
TITLES: dict[str, str] = {}


def register(name: str, title: str):
    def deco(fn):
        METHODS[name] = fn
        TITLES[name] = title
        return fn

    return deco


def register_direct(name: str, title: str):
    def deco(fn):
        DIRECT[name] = fn
        TITLES[name] = title
        return fn

    return deco


@register_direct("gauss", "Метод Гаусса (прямой)")
def gauss(A, b):
    """
    Метод Гаусса с выбором главного элемента по столбцу.
    Прямой ход: на шаге k берём строку с наибольшим |a_ik| (i >= k), переставляем её наверх
    и обнуляем элементы столбца под диагональю. Обратный ход -- подстановка снизу вверх.
    """
    n = len(b)
    M = np.hstack([np.asarray(A, dtype=float), np.asarray(b, dtype=float).reshape(-1, 1)])
    scale = np.abs(M[:, :n]).max()
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) <= 1e-14 * scale:
            raise np.linalg.LinAlgError("Метод Гаусса: матрица вырождена (нулевой главный элемент).")
        if p != k:
            M[[k, p]] = M[[p, k]]
        factors = M[k + 1:, k] / M[k, k]
        M[k + 1:, k:] -= factors[:, None] * M[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x


@register("jacobi", "Метод Якоби")
def jacobi(A, b, x0, omega: float = 1.0):
    """x_{k+1} = (1-w) x_k + w D^{-1} (b - (A - D) x_k). При w = 1 -- классический Якоби."""
    D = np.diag(A).copy()
    if np.any(D == 0):
        raise ValueError("Метод Якоби: на диагонали есть нули (переставьте строки).")
    R = A - np.diagflat(D)
    x = x0.copy()
    while True:
        x = (1 - omega) * x + omega * (b - R @ x) / D
        yield x


@register("mr", "Метод минимальных невязок")
def minimal_residual(A, b, x0):
    """
    r_k = b - A x_k,  tau_k = (r_k, A r_k) / (A r_k, A r_k),  x_{k+1} = x_k + tau_k r_k.
    Шаг tau минимизирует ||r_{k+1}||_2 вдоль направления r_k.
    Сходится, если симметричная часть A положительно определена.
    """
    x = x0.copy()
    r = b - A @ x
    while True:
        Ar = A @ r
        denom = Ar @ Ar
        if denom == 0.0:  # r = 0 -- точное решение
            yield x
            return
        tau = (r @ Ar) / denom
        x = x + tau * r
        r = r - tau * Ar
        yield x


@register("cg", "Метод сопряжённых градиентов")
def conjugate_gradient(A, b, x0):
    """Классический CG для симметричной положительно определённой A."""
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs = r @ r
    while True:
        if rs == 0.0:
            yield x
            return
        Ap = A @ p
        alpha = rs / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        p = r + (rs_new / rs) * p
        rs = rs_new
        yield x


# --------------------------------------------------------------------------- #
#  Единая функция solve
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    method: str
    x: np.ndarray
    residuals: list[float]  # относительная невязка ||b - A x_k|| / ||b||, k = 0..K
    errors: list[float] | None  # ||x_k - x*|| / ||x*||, если известно точное решение
    converged: bool
    reason: str
    params: dict = field(default_factory=dict)
    direct: bool = False  # прямой метод: итераций нет, residuals/errors содержат одно значение

    @property
    def iterations(self) -> int:
        return 0 if self.direct else len(self.residuals) - 1

    @property
    def mean_rate(self) -> float:
        """Средний геометрический множитель убывания невязки за итерацию."""
        k = self.iterations
        r0, rk = self.residuals[0], self.residuals[-1]
        if k == 0 or r0 == 0 or rk == 0:
            return float("nan")
        return (rk / r0) ** (1.0 / k)


def solve(
    A: np.ndarray,
    b: np.ndarray,
    method: str,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    x_exact: np.ndarray | None = None,
    **params,
) -> Result:
    """
    Единая точка входа: решает A x = b указанным методом.

    method   -- ключ из METHODS (итерационные: "jacobi", "mr", "cg") или DIRECT (прямые: "gauss")
    tol      -- останов по относительной невязке ||b - A x|| / ||b|| < tol (только итерационные)
    **params -- параметры конкретного метода (например, omega=0.8 для Якоби)
    """
    if method not in METHODS and method not in DIRECT:
        raise KeyError(f"Неизвестный метод '{method}'. Доступны: {', '.join([*DIRECT, *METHODS])}")
    nb = np.linalg.norm(b)
    nb = nb if nb > 0 else 1.0
    nx = np.linalg.norm(x_exact) if x_exact is not None else 1.0

    def rel_res(x):
        return np.linalg.norm(b - A @ x) / nb

    if method in DIRECT:
        x = DIRECT[method](A, b, **params)
        err = [np.linalg.norm(x - x_exact) / nx] if x_exact is not None else None
        return Result(method, x, [rel_res(x)], err, True, "прямой метод", params, direct=True)

    x0 = np.zeros_like(b, dtype=float) if x0 is None else np.asarray(x0, dtype=float)

    x = x0
    residuals = [rel_res(x)]
    errors = [np.linalg.norm(x - x_exact) / nx] if x_exact is not None else None
    converged, reason = residuals[0] < tol, "достигнута точность"
    if not converged:
        reason = f"исчерпан лимит {max_iter} итераций"
        for x in METHODS[method](A, b, x0, **params):
            res = rel_res(x)
            residuals.append(res)
            if errors is not None:
                errors.append(np.linalg.norm(x - x_exact) / nx)
            if not np.isfinite(res) or res > 1e12 * max(residuals[0], 1e-300):
                reason = "расходится"
                break
            if res < tol:
                converged, reason = True, "достигнута точность"
                break
            if len(residuals) - 1 >= max_iter:
                break
    return Result(method, x, residuals, errors, converged, reason, params)


# --------------------------------------------------------------------------- #
#  Диагностика системы
# --------------------------------------------------------------------------- #


def analyze(A: np.ndarray) -> dict:
    """Свойства матрицы, которые определяют применимость методов."""
    n = A.shape[0]
    sym = bool(np.allclose(A, A.T))
    eig_sym = np.linalg.eigvalsh((A + A.T) / 2)
    spd = sym and bool(eig_sym.min() > 0)
    off = np.abs(A).sum(axis=1) - np.abs(np.diag(A))
    D = np.diag(A)
    rho_j = float("nan")
    if np.all(D != 0):
        B = -(A - np.diagflat(D)) / D[:, None]
        rho_j = float(np.max(np.abs(np.linalg.eigvals(B))))
    kappa = float(np.linalg.cond(A))
    out = {
        "n": n,
        "symmetric": sym,
        "positive_definite_sym_part": bool(eig_sym.min() > 0),
        "spd": spd,
        "strict_diag_dominance": bool(np.all(np.abs(D) > off)),
        "cond2": kappa,
        "rho_jacobi": rho_j,
        "lambda_min": float(eig_sym.min()),
        "lambda_max": float(eig_sym.max()),
    }
    if spd:
        out["theory_mr"] = (kappa - 1) / (kappa + 1)
        out["theory_cg"] = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    return out


# --------------------------------------------------------------------------- #
#  Графики
# --------------------------------------------------------------------------- #

# Первые три слота проверенной категориальной палитры (валидны для всех пар, светлая тема)
COLORS = {"jacobi": "#2a78d6", "mr": "#eb6834", "cg": "#1baf7a"}
MARKERS = {"jacobi": "o", "mr": "s", "cg": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, xlabel, ylabel, title):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())  # без частокола малых делений
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


# Нижняя граница для значений, которые на лог-шкале нельзя показать (точный ноль)
FLOOR = 1e-17
# Якоби и MR на плохо обусловленных задачах почти совпадают -- Якоби рисуем толще и ниже
LINEWIDTH = {"jacobi": 3.4, "mr": 2.0, "cg": 2.0}
ZORDER = {"jacobi": 2, "mr": 3, "cg": 4}


def _marker_idx(n: int, logx: bool) -> list[int]:
    """Одинаковое число маркеров (~10) на каждой кривой; на лог-оси -- равномерно по логарифму."""
    if n <= 12:
        return list(range(n))
    if logx:
        idx = np.unique(np.geomspace(1, n - 1, 10).round().astype(int))
        return [0, *idx.tolist()]
    return np.unique(np.linspace(0, n - 1, 10).round().astype(int)).tolist()


def _plot_curve(ax, res: Result, values, lw=None) -> bool:
    """Рисует кривую. Точный ноль поднимается до FLOOR и подписывается. Возвращает True, если был ноль."""
    c = COLORS.get(res.method, "#6250d6")
    values = np.asarray(values, dtype=float)
    has_zero = bool(np.any(values <= 0))
    shown = np.maximum(values, FLOOR)
    logx = ax.get_xscale() != "linear"
    ax.plot(
        np.arange(len(shown)), shown, color=c, linewidth=lw or LINEWIDTH.get(res.method, 2),
        zorder=ZORDER.get(res.method, 3), label=TITLES[res.method],
        marker=MARKERS.get(res.method, "D"), markevery=_marker_idx(len(shown), logx),
        markersize=7, markeredgecolor="white", markeredgewidth=1.0,
    )
    if has_zero:
        i = int(np.argmax(values <= 0))
        ax.annotate("ровно 0", xy=(i, FLOOR), xytext=(-8, 12), textcoords="offset points",
                    ha="right", color=INK2, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    return has_zero


def _maybe_log_x(ax, results: list[Result]):
    """Если методам нужно сильно разное число итераций, ось X: линейная до 10, дальше логарифмическая."""
    its = [max(r.iterations, 1) for r in results]
    if max(its) / min(its) > 20:
        ax.set_xscale("symlog", linthresh=10, linscale=0.6)
        ax.set_xlabel("номер итерации k (шкала: линейная до 10, далее логарифмическая)", color=INK2)


def _tol_line(ax, tol: float):
    ax.axhline(tol, color=INK2, linestyle="--", linewidth=1, zorder=1)
    ax.text(0.015, tol, f"tol = {tol:g}", color=INK2, va="bottom", ha="left", fontsize=9,
            transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def _fit_axes(ax, low: float | None = None):
    """Пределы осей после построения: X начинается с 0; низ Y опускается, чтобы вошло значение `low`."""
    ax.set_xlim(left=0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, low / 5) if low else lo, hi * 3)  # запас сверху: старт не липнет к рамке


def _legend_below(ax):
    n = len(ax.get_legend_handles_labels()[0])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3 if n <= 3 else 2)


def _direct_lines(ax, direct: list[Result]) -> float | None:
    """Прямой метод итераций не имеет: рисуем его невязку горизонтальной линией (достижимая точность)."""
    low = None
    for r in direct:
        v = r.residuals[0]
        shown = max(v, FLOOR)
        ax.axhline(shown, color=INK2, linestyle=":", linewidth=1.8, zorder=1,
                   label=f"{TITLES[r.method]}: невязка {v:.1e}")
        low = shown if low is None else min(low, shown)
    return low


def make_plots(results: list[Result], tol: float, out_dir: Path, tag: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    direct = [r for r in results if r.direct]
    results = [r for r in results if not r.direct]  # графики-кривые только для итерационных
    if not results:
        return files

    # 1. по отдельному графику невязки на метод
    for res in results:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
        _style(ax, "номер итерации k", r"$\|b - Ax_k\| / \|b\|$",
               f"{TITLES[res.method]}: убывание невязки")
        z = _plot_curve(ax, res, res.residuals, lw=2.2)  # одна кривая: легенда не нужна, имя в заголовке
        _tol_line(ax, tol)
        _fit_axes(ax, FLOOR if z else None)
        fig.tight_layout()
        p = out_dir / f"{tag}_residual_{res.method}.png"
        fig.savefig(p, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    # 2. сравнение невязок: общая ось X в итерациях
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=140)
    _style(ax, "номер итерации k", r"$\|b - Ax_k\| / \|b\|$", "Сравнение методов: невязка")
    _maybe_log_x(ax, results)  # до построения кривых: от шкалы зависит расстановка маркеров
    z = any([_plot_curve(ax, res, res.residuals) for res in results])
    _tol_line(ax, tol)
    lows = [v for v in (FLOOR if z else None, _direct_lines(ax, direct)) if v]
    _fit_axes(ax, min(lows) if lows else None)
    _legend_below(ax)
    fig.tight_layout()
    p = out_dir / f"{tag}_residual_compare.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    files.append(p)

    # 3. сравнение ошибки относительно точного решения
    if all(r.errors is not None for r in results):
        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=140)
        _style(ax, "номер итерации k", r"$\|x_k - x^*\| / \|x^*\|$",
               "Сравнение методов: ошибка решения")
        _maybe_log_x(ax, results)
        z = any([_plot_curve(ax, res, res.errors) for res in results])
        _fit_axes(ax, FLOOR if z else None)
        _legend_below(ax)
        fig.tight_layout()
        p = out_dir / f"{tag}_error_compare.png"
        fig.savefig(p, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        files.append(p)
    return files


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_params(items: list[str]) -> dict[str, dict]:
    """--param jacobi.omega=0.8  ->  {"jacobi": {"omega": 0.8}}"""
    out: dict[str, dict] = {}
    for it in items or []:
        key, _, val = it.partition("=")
        meth, _, name = key.partition(".")
        out.setdefault(meth, {})[name] = float(val)
    return out


def run(system: str, methods: list[str], tol: float, max_iter: int,
        method_params: dict[str, dict], out_dir: Path, tag: str) -> dict:
    A, b, name = load_system(system)
    info = analyze(A)

    print(f"Система: {name}   n = {info['n']}")
    for k, v in info.items():
        print(f"  {k:28s} {v}")

    # Эталон -- наш метод Гаусса; сам Гаусс проверяем независимо по LAPACK (numpy)
    x_star = solve(A, b, "gauss").x
    x_lapack = np.linalg.solve(A, b)
    diff = np.linalg.norm(x_star - x_lapack) / np.linalg.norm(x_lapack)
    print(f"\nЭталон: метод Гаусса. Сверка с numpy.linalg.solve: ‖x_gauss − x_numpy‖/‖x_numpy‖ = {diff:.2e}")
    print("Ошибка итерационных методов считается относительно решения Гаусса,")
    print("ошибка самого Гаусса -- относительно numpy.linalg.solve.")

    results = [
        solve(A, b, m, tol=tol, max_iter=max_iter,
              x_exact=x_lapack if m in DIRECT else x_star, **method_params.get(m, {}))
        for m in methods
    ]
    print(f"\n{'метод':32s}{'итераций':>10s}{'невязка':>13s}{'ошибка':>13s}{'rate':>9s}  статус")
    for r in results:
        its = "прямой" if r.direct else str(r.iterations)
        rate = f"{r.mean_rate:9.4f}" if np.isfinite(r.mean_rate) else "        —"
        print(f"{TITLES[r.method]:32s}{its:>10s}{r.residuals[-1]:13.2e}{r.errors[-1]:13.2e}{rate}  {r.reason}")

    files = make_plots(results, tol, out_dir, tag)
    print("\nГрафики:")
    for f in files:
        print("  ", f.relative_to(HERE) if f.is_relative_to(HERE) else f)
    return {"name": name, "info": info, "results": results, "files": files, "x_star": x_star}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--methods", default="gauss,jacobi,mr,cg",
                    help=f"из: {', '.join([*DIRECT, *METHODS])}")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=10_000)
    ap.add_argument("--param", action="append", help="параметры метода, например jacobi.omega=0.8")
    ap.add_argument("--out", default=str(HERE / "pictures"))
    ap.add_argument("--tag", default=None, help="префикс имён картинок")
    a = ap.parse_args()
    # на Windows консоль с узкой кодировкой не должна падать на символах вроде «‖» и «−»
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    tag = a.tag or Path(a.system).stem.replace(":", "_").replace("=", "").replace(",", "_")
    run(a.system, a.methods.split(","), a.tol, a.max_iter, parse_params(a.param), Path(a.out), tag)


if __name__ == "__main__":
    main()
