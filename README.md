# Quantitative Risk Engine: Stochastic Portfolio Optimization & Risk Modeling

![Language](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Data Source](https://img.shields.io/badge/Data-MOEX%20ISS-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance-blueviolet)

> **Advanced framework for stochastic portfolio modeling, volatility regime analysis, and optimal control (HJB) strategy implementation for Russian financial markets.**

---

##  Table of Contents
1.  [Executive Summary](#-executive-summary)
2.  [Theoretical Framework](#-theoretical-framework)
    *   [Stochastic Dynamics](#stochastic-dynamics)
    *   [Volatility Modeling (GARCH)](#volatility-modeling-garch)
    *   [Optimal Control (HJB Equation)](#optimal-control-hjb-equation)
3.  [Technical Architecture](#-technical-architecture)
4.  [Risk Analytics & Metrics](#-risk-analytics--metrics)
5.  [Project Structure](#-project-structure)
6.  [Installation & Usage](#-installation--usage)
7.  [Disclaimer](#-disclaimer)

**Executive Summary**

Этот проект представляет собой исследовательский движок (Research Engine) для количественного анализа рыночных рисков и управления портфелем. Он решает задачу **динамического распределения активов** в условиях неопределенности, используя методы стохастического исчисления.

---

##  Ключевые возможности

*   **ETL & Data Pipeline**: Автоматическая выгрузка истории котировок через MOEX ISS API, очистка данных, детекция сплитов и синхронизация временных рядов.
*   **Volatility Modeling**: Оценка рыночных режимов с помощью **GARCH(1,1)**. Анализ стационарности, персистентности волатильности и тяжелых хвостов распределения.
*   **Quantitative Modeling**: Решение задачи стохастического оптимального управления (задача Мертона) для максимизации CRRA-полезности. Динамическое перераспределение весов.
*   **Monte Carlo Simulation**: Генерация 10,000+ сценариев эволюции портфеля. Поддержка процессов:
    *   Геометрическое броуновское движение (GBM).
    *   *(В разработке)* Скачкообразная диффузия Мертона (MJD) и процессы Леви.
*   **Risk Metrics**: Расчет VaR (95/99), CVaR (Expected Shortfall), Maximum Drawdown и вероятности разорения.

---

##  Theoretical Framework

### Stochastic Dynamics
Мы моделируем эволюцию цены рискового актива $S_t$ как геометрическое броуновское движение (GBM), расширенное для учета стохастической волатильности:

$$ \frac{dS_t}{S_t} = \mu dt + \sigma_t dW_t $$

Где:
*   $\mu$ — коэффициент сноса (drift), отражающий ожидаемую доходность.
*   $\sigma_t$ — зависящая от времени волатильность.
*   $W_t$ — стандартный винеровский процесс (Brownian motion) с корреляцией $\rho_{ij}$ между активами.

### Volatility Modeling (GARCH)
Финансовые временные ряды демонстрируют кластеризацию волатильности и "тяжелые хвосты". Для корректной оценки $\sigma_t$ используется модель **GARCH(1,1)**:

$$ r_t = \mu + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1) $$

$$ \sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1} $$

Условия стационарности: $\alpha + \beta < 1$. Если $\alpha + \beta \approx 1$, процесс проявляет свойства IGARCH (Integrated GARCH), что учитывается при фильтрации активов.

### Optimal Control (HJB Equation)
Рассматривается задача стохастического оптимального управления на конечном горизонте $T$. Цель — максимизация ожидаемой полезности терминального богатства $U(X_T)$ для инвестора с функцией полезности CRRA (Constant Relative Risk Aversion):

$$ U(x) = \frac{x^{1-\gamma}}{1-\gamma}, \quad \gamma > 0, \gamma \neq 1 $$

Функция стоимости $V(t, x)$ определяется как:
$$ V(t, x) = \sup_{w} \mathbb{E} \left[ U(X_T) \mid X_t = x \right] $$

Она удовлетворяет уравнению в частных производных **Гамильтона — Якоби — Беллмана (HJB)**:

$$ \frac{\partial V}{\partial t} + \sup_{w \in \mathbb{R}^n} \left[ (rx + w^T(\mu - r\mathbf{1})x) \frac{\partial V}{\partial x} + \frac{1}{2} x^2 w^T \Sigma w \frac{\partial^2 V}{\partial x^2} \right] = 0 $$

Аналитическое решение для оптимальных весов портфеля $w^*$:

$$ w^* = \frac{1}{\gamma} \Sigma^{-1} (\mu - r\mathbf{1}) $$

Где $\Sigma$ — ковариационная матрица, $r$ — безрисковая ставка, $\gamma$ — коэффициент неприятия риска.

---

##  Technical Architecture

Проект построен по модульному принципу, где каждый этап анализа изолирован в отдельный компонент (notebook/module):

1.  **Data Layer (`data_loader`, `filtration`)**:
    *   Загрузка исторических данных (OHLCV).
    *   Кэширование в формат **Parquet** для высокой производительности I/O.
    *   Фильтрация неликвидных активов и детекция аномалий (outliers detection).

2.  **Modeling Layer (`GARCH`, `HJB_strategy`)**:
    *   Калибровка параметров ($\omega, \alpha, \beta$) методом максимального правдоподобия (MLE).
    *   Расчет ковариационных матриц с применением shrinkage-методов (Ledoit-Wolf) для борьбы с шумом.
    *   Формирование вектора оптимальных весов.

3.  **Simulation Layer (`VaR_backtest`)**:
    *   Векторизированная реализация Монте-Карло на `NumPy`.
    *   Генерация коррелированных многомерных винеровских процессов через разложение Холецкого ($\mathbf{L}\mathbf{L}^T = \Sigma$).

4.  **Analytics Layer (`results_analitics`)**:
    *   Агрегация результатов симуляций.
    *   Визуализация распределений доходности и траекторий капитала.

---

##  Risk Analytics & Metrics

Система рассчитывает полный спектр метрик риска согласно стандартам Basel III / FRTB:

| Metric | Описание | Формула / Метод |
| :--- | :--- | :--- |
| **VaR (Value at Risk)** | Максимальный ожидаемый убыток с уверенностью $(1-\alpha)$. | $P(L > \text{VaR}_{\alpha}) \le 1-\alpha$ |
| **CVaR (Expected Shortfall)** | Средний убыток в хвосте распределения (за пределами VaR). Более когерентная мера риска. | $\text{ES}_{\alpha} = \frac{1}{1-\alpha} \int_{\alpha}^{1} \text{VaR}_{u} du$ |
| **Maximum Drawdown** | Максимальное падение стоимости портфеля от пика до дна. | $\text{MDD} = \sup_{t} (\sup_{s < t} X_s - X_t)$ |
| **Sharpe Ratio** | Отношение риск/доходность. | $S = \frac{E[R_p] - R_f}{\sigma_p}$ |

---

## Требования
*   Python 3.8+
*   Библиотеки: `numpy`, `pandas`, `scipy`, `arch`, `matplotlib`, `seaborn`

---

##  Disclaimer

Данный проект разработан в исследовательских целях для демонстрации методов количественного анализа (Quantitative Finance). 

*   Код не является торговым роботом или платформой для исполнения ордеров.
*   Результаты моделирования (Backtesting) не гарантируют будущих доходов.
*   Автор не несет ответственности за финансовые убытки, возникшие в результате использования представленных моделей.

---

**Author:** Egor Galkin  
**Profile:** Quantitative Researcher / Risk Analyst  
**Focus:** Stochastic Calculus, Derivatives Pricing, Market Risk Management
