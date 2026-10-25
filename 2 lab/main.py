"""
Лабораторная 2: решение СЛАУ  A x = b.

Методы:  gauss (прямой, эталон) · jacobi · mr (мин. невязки) · cg (сопряжённые градиенты)

    python main.py                          # все встроенные системы -> pictures/*.png
    python main.py --system poisson         # одна система (easy | poisson | hard)
    python main.py --system my.json         # своя СЛАУ: {"A": [[...]], "b": [...]}
    python main.py --methods gauss,cg --tol 1e-12 --param jacobi.omega=0.8

Архитектура: итерационный метод -- генератор приближений x_k (@register), прямой -- функция
(A, b) -> x (@register_direct). Единая solve(A, b, method, **params) работает с обоими видами.
Новый метод = одна функция с декоратором, остальной код не меняется.
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
#  Системы
# --------------------------------------------------------------------------- #


def gen_random(n: int = 8, seed: int = 0):
    """Симметричная матрица со строгим диагональным преобладанием (=> СПД)."""
    rng = np.random.default_rng(seed)
    M = rng.uniform(-1, 1, (n, n))
    A = (M + M.T) / 2
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(n)] = np.abs(A).sum(axis=1) + rng.uniform(0.5, 2.0, n)
    return A, A @ rng.uniform(-3, 3, n)


def gen_poisson(n: int = 50):
    """Трёхдиагональная (2, -1): дискретизация -u'' = f. Обусловленность ~ n^2."""
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1), np.ones(n)


def gen_spd(n: int = 30, cond: float = 100.0, seed: int = 0):
    """Плотная СПД-матрица с заданным числом обусловленности (спектр равномерно в [1, cond])."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = Q @ np.diag(np.linspace(1.0, cond, n)) @ Q.T
    A = (A + A.T) / 2
    return A, A @ rng.uniform(-3, 3, n)


GENERATORS = {"random": gen_random, "poisson": gen_poisson, "spd": gen_spd}

# имя -> (описание, генератор:параметры, tol)
SYSTEMS = {
    "easy": ("Диагональное преобладание, n=8", "random:n=8,seed=1", 1e-10),
    "poisson": ("Трёхдиагональная (2, −1), n=50", "poisson:n=50", 1e-6),
    "hard": ("Плотная СПД, n=30, κ=100", "spd:n=30,cond=100,seed=3", 1e-10),
}


def load_system(spec: str):
    """spec: имя из SYSTEMS, 'генератор:k=v,...' или путь к JSON. Возвращает (A, b, описание, tol)."""
    tol = 1e-10
    desc = spec
    if spec in SYSTEMS:
        desc, gen, tol = SYSTEMS[spec]
        spec = gen
    kind, _, args = spec.partition(":")
    if kind in GENERATORS:
        kw = {k: float(v) if "." in v else int(v) for k, v in (i.split("=") for i in args.split(",") if i)}
        A, b = GENERATORS[kind](**kw)
        return A, b, desc, tol
    path = Path(spec)
    if not path.exists() and (HERE / spec).exists():
        path = HERE / spec
    if not path.exists():
        raise SystemExit(f"Система не найдена: {spec}\n"
                         f"Доступны: {', '.join(SYSTEMS)}, либо JSON {{\"A\": [[...]], \"b\": [...]}}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["A"], float), np.array(data["b"], float), data.get("name", path.stem), tol


# --------------------------------------------------------------------------- #
#  Методы
# --------------------------------------------------------------------------- #

METHODS: dict[str, Callable[..., Iterator[np.ndarray]]] = {}  # итерационные
DIRECT: dict[str, Callable[..., np.ndarray]] = {}  # прямые
TITLES: dict[str, str] = {}


def register(name: str, title: str):
    def deco(fn):
        METHODS[name], TITLES[name] = fn, title
        return fn
    return deco


def register_direct(name: str, title: str):
    def deco(fn):
        DIRECT[name], TITLES[name] = fn, title
        return fn
    return deco


@register_direct("gauss", "Гаусс (прямой)")
def gauss(A, b):
    """Метод Гаусса с выбором главного элемента по столбцу + обратный ход."""
    n = len(b)
    M = np.hstack([np.asarray(A, float), np.asarray(b, float).reshape(-1, 1)])
    scale = np.abs(M[:, :n]).max()
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) <= 1e-14 * scale:
            raise np.linalg.LinAlgError("Матрица вырождена (нулевой главный элемент).")
        if p != k:
            M[[k, p]] = M[[p, k]]
        M[k + 1:, k:] -= (M[k + 1:, k] / M[k, k])[:, None] * M[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x


@register("jacobi", "Якоби")
def jacobi(A, b, x0, omega: float = 1.0):
    """x_{k+1} = (1-w) x_k + w D^-1 (b - (A-D) x_k); w=1 -- классический Якоби."""
    D = np.diag(A).copy()
    if np.any(D == 0):
        raise ValueError("Якоби: на диагонали есть нули.")
    R = A - np.diagflat(D)
    x = x0.copy()
    while True:
        x = (1 - omega) * x + omega * (b - R @ x) / D
        yield x


@register("mr", "Минимальные невязки")
def minimal_residual(A, b, x0):
    """x_{k+1} = x_k + tau r_k, tau = (r, Ar)/(Ar, Ar) -- минимизирует ||r_{k+1}||."""
    x = x0.copy()
    r = b - A @ x
    while True:
        Ar = A @ r
        den = Ar @ Ar
        if den == 0.0:
            yield x
            return
        tau = (r @ Ar) / den
        x = x + tau * r
        r = r - tau * Ar
        yield x


@register("cg", "Сопряжённые градиенты")
def conjugate_gradient(A, b, x0):
    """Классический CG для симметричной положительно определённой A."""
    x = x0.copy()
    r = b - A @ x
    p, rs = r.copy(), r @ r
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
    residuals: list[float]  # ||b - A x_k|| / ||b||, k = 0..K
    errors: list[float]  # ||x_k - x*|| / ||x*||
    converged: bool
    reason: str
    direct: bool = False
    params: dict = field(default_factory=dict)

    @property
    def iterations(self) -> int:
        return 0 if self.direct else len(self.residuals) - 1


def solve(A, b, method: str, x0=None, tol: float = 1e-10, max_iter: int = 10_000,
          x_exact=None, **params) -> Result:
    """
    Решает A x = b методом `method` (ключ из DIRECT или METHODS); **params -- параметры метода.
    Итерационные: останов по ||b - A x|| / ||b|| < tol, защита от расходимости и лимит итераций.
    x_exact -- точное решение для истории ошибки (если не задано, ошибка = nan).
    """
    if method not in METHODS and method not in DIRECT:
        raise KeyError(f"Неизвестный метод '{method}'. Доступны: {', '.join([*DIRECT, *METHODS])}")
    nb = np.linalg.norm(b) or 1.0
    nx = np.linalg.norm(x_exact) if x_exact is not None else np.nan

    def res(x):
        return np.linalg.norm(b - A @ x) / nb

    def err(x):
        return np.linalg.norm(x - x_exact) / nx if x_exact is not None else np.nan

    if method in DIRECT:
        x = DIRECT[method](A, b, **params)
        return Result(method, x, [res(x)], [err(x)], True, "прямой", True, params)

    x = np.zeros_like(b, float) if x0 is None else np.asarray(x0, float)
    residuals, errors = [res(x)], [err(x)]
    converged, reason = residuals[0] < tol, "точность достигнута"
    if not converged:
        reason = f"лимит {max_iter} итераций"
        for x in METHODS[method](A, b, x, **params):
            residuals.append(res(x))
            errors.append(err(x))
            if not np.isfinite(residuals[-1]) or residuals[-1] > 1e12 * max(residuals[0], 1e-300):
                reason = "расходится"
                break
            if residuals[-1] < tol:
                converged, reason = True, "точность достигнута"
                break
            if len(residuals) - 1 >= max_iter:
                break
    return Result(method, x, residuals, errors, converged, reason, False, params)


def analyze(A) -> dict:
    """Свойства матрицы, от которых зависит применимость методов."""
    D = np.diag(A)
    off = np.abs(A).sum(axis=1) - np.abs(D)
    eig = np.linalg.eigvalsh((A + A.T) / 2)
    B = -(A - np.diagflat(D)) / D[:, None]
    return {
        "spd": bool(np.allclose(A, A.T) and eig.min() > 0),
        "diag_dom": bool(np.all(np.abs(D) > off)),
        "cond": float(np.linalg.cond(A)),
        "rho_jacobi": float(np.max(np.abs(np.linalg.eigvals(B)))),
    }


# --------------------------------------------------------------------------- #
#  График: слева невязка, справа ошибка; все методы на одной оси
# --------------------------------------------------------------------------- #

COLORS = {"jacobi": "#2a78d6", "mr": "#eb6834", "cg": "#1baf7a"}  # проверенная палитра
MARKERS = {"jacobi": "o", "mr": "s", "cg": "^"}
LW = {"jacobi": 3.4, "mr": 2.0, "cg": 2.0}  # Якоби толще и ниже: на плохих системах он ≈ MR
ZO = {"jacobi": 2, "mr": 3, "cg": 4}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FLOOR = 1e-17  # куда «поднимаем» точный ноль, чтобы показать его на лог-шкале


def _markers(n: int, logx: bool) -> list[int]:
    if n <= 12:
        return list(range(n))
    if logx:
        return [0, *np.unique(np.geomspace(1, n - 1, 10).round().astype(int)).tolist()]
    return np.unique(np.linspace(0, n - 1, 10).round().astype(int)).tolist()


def _panel(ax, results, direct, key, ylabel, tol=None):
    ax.set_yscale("log")
    its = [max(r.iterations, 1) for r in results]
    logx = max(its) / min(its) > 20  # CG и Якоби на разных масштабах: линейно до 10, дальше лог.
    if logx:
        ax.set_xscale("symlog", linthresh=10, linscale=0.6)
    zero = False
    for r in results:
        v = np.asarray(getattr(r, key), float)
        zero |= bool(np.any(v <= 0))
        shown = np.maximum(v, FLOOR)
        ax.plot(range(len(shown)), shown, color=COLORS[r.method], lw=LW[r.method], zorder=ZO[r.method],
                marker=MARKERS[r.method], markevery=_markers(len(shown), logx), ms=6.5,
                mec="white", mew=1, label=TITLES[r.method])
        if np.any(v <= 0):
            i = int(np.argmax(v <= 0))
            ax.annotate("ровно 0", (i, FLOOR), xytext=(-6, 10), textcoords="offset points",
                        ha="right", fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    low = FLOOR if zero else None
    if tol:
        ax.axhline(tol, color=INK2, ls="--", lw=1, zorder=1)
        ax.text(0.02, tol, f"tol = {tol:g}", transform=ax.get_yaxis_transform(), fontsize=8,
                color=INK2, va="bottom", bbox=dict(fc="white", ec="none", pad=1))
    for r in direct:  # прямой метод: итераций нет -- показываем достижимую точность линией
        if key == "residuals":
            v = max(r.residuals[0], FLOOR)
            ax.axhline(v, color=INK2, ls=":", lw=1.8, zorder=1, label=f"{TITLES[r.method]}: {r.residuals[0]:.0e}")
            low = v if low is None else min(low, v)
    ax.set_xlim(left=0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, low / 5) if low else lo, hi * 3)
    ax.set_xlabel("итерация k" + (" (линейно до 10, далее лог.)" if logx else ""), color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def make_plot(results: list[Result], title: str, tol: float, path: Path):
    it = [r for r in results if not r.direct]
    direct = [r for r in results if r.direct]
    if not it:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=130)
    _panel(axes[0], it, direct, "residuals", r"невязка $\|b-Ax_k\|/\|b\|$", tol)
    _panel(axes[1], it, [], "errors", r"ошибка $\|x_k-x^*\|/\|x^*\|$")
    axes[0].set_title("Невязка", loc="left", color=INK, fontweight="bold", fontsize=11)
    axes[1].set_title("Ошибка относительно Гаусса", loc="left", color=INK, fontweight="bold", fontsize=11)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_params(items):
    """--param jacobi.omega=0.8 -> {"jacobi": {"omega": 0.8}}"""
    out: dict[str, dict] = {}
    for it in items or []:
        key, _, val = it.partition("=")
        meth, _, name = key.partition(".")
        out.setdefault(meth, {})[name] = float(val)
    return out


def run(system: str, methods: list[str], tol: float | None, max_iter: int,
        params: dict, out_dir: Path) -> list[Result]:
    A, b, desc, sys_tol = load_system(system)
    tol = tol or sys_tol
    info = analyze(A)
    x_star = solve(A, b, "gauss").x  # эталон -- наш Гаусс
    x_lapack = np.linalg.solve(A, b)  # независимая проверка Гаусса
    gap = np.linalg.norm(x_star - x_lapack) / np.linalg.norm(x_lapack)

    print(f"\n=== {system}: {desc} ===")
    print(f"κ(A) = {info['cond']:.3g}   ρ(Якоби) = {info['rho_jacobi']:.4f}   "
          f"СПД: {info['spd']}   диаг. преобладание: {info['diag_dom']}   tol = {tol:g}")
    print(f"Сверка Гаусса с numpy.linalg.solve: {gap:.1e}")
    results = [solve(A, b, m, tol=tol, max_iter=max_iter,
                     x_exact=x_lapack if m in DIRECT else x_star, **params.get(m, {})) for m in methods]
    print(f"{'метод':24s}{'итераций':>10s}{'невязка':>11s}{'ошибка':>11s}  статус")
    for r in results:
        its = "—" if r.direct else str(r.iterations)
        print(f"{TITLES[r.method]:24s}{its:>10s}{r.residuals[-1]:11.1e}{r.errors[-1]:11.1e}  {r.reason}")

    name = Path(system).stem if system not in SYSTEMS else system
    path = out_dir / f"{name.replace(':', '_').replace('=', '').replace(',', '_')}.png"
    make_plot(results, f"{system}: {desc}", tol, path)
    print(f"график: {path.relative_to(HERE) if path.is_relative_to(HERE) else path}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default=",".join(SYSTEMS), help="имена через запятую, JSON-файл или генератор")
    ap.add_argument("--methods", default="gauss,jacobi,mr,cg", help=f"из: {', '.join([*DIRECT, *METHODS])}")
    ap.add_argument("--tol", type=float, default=None, help="по умолчанию свой для каждой системы")
    ap.add_argument("--max-iter", type=int, default=10_000)
    ap.add_argument("--param", action="append", help="параметр метода: jacobi.omega=0.8")
    ap.add_argument("--out", default=str(HERE / "pictures"))
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # узкая кодировка консоли Windows не должна ронять вывод
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    for s in a.system.split(","):
        run(s.strip(), a.methods.split(","), a.tol, a.max_iter, parse_params(a.param), Path(a.out))


if __name__ == "__main__":
    main()
