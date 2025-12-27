# Stochastic Portfolio Optimization via HJB Strategy with Stress-Testing

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

## Executive Summary

Полнофункциональная реализация портфельной оптимизации на основе **уравнения Гамильтона-Якоби-Беллмана (HJB)** с коэффициентом неприятия риска γ = 3.0, включающая загрузку данных с MOEX ISS API, GARCH-моделирование волатильности, Монте-Карло симуляцию с 5000 траектории и комплексный стресс-тестинг с 8 сценариями. 

**Ключевые особенности:**
- Загрузка исторических данных с MOEX ISS для произвольных тикеров и периодов
- Оценка параметров (µ, Σ) с фильтрацией активов по критериям GARCH-стационарности
- Аналитическое решение задачи Мертона с проекцией на ограничения (no short sales, full investment)
- Расчет полной доходности (прирост капитала + дивиденды) с учетом корпоративных действий
- Симуляция 5000 траекторий капитала с дискретизацией Эйлера-Маруямы
- Расчет метрик риска: VaR, CVaR, Sharpe Ratio, Maximum Drawdown
- Стресс-тестирование по 8 сценариям (базовый, кризис, волатильность, корреляционный шок и т.д.)
- Анализ чувствительности к γ, временным горизонтам и параметрам рынка

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

## Математическая основа

### 1. Задача Мертона и HJB-уравнение

Инвестор с функцией полезности **CRRA** (Constant Relative Risk Aversion):

$$U(C) = \frac{C^{1-\gamma}}{1-\gamma}, \quad \gamma > 0, \gamma \neq 1$$

максимизирует ожидаемую дисконтированную полезность:

$$\max_{\{\pi_t\}_{t \geq 0}} \mathbb{E}\left[\int_0^{\infty} e^{-\rho t} U(C_t) \, dt\right]$$

Динамика капитала (стохастическое дифференциальное уравнение):

$$dX_t = X_t \left(w_t^T \mu \, dt + w_t^T \Sigma^{1/2} dW_t\right)$$

где:
- $X_t$ — капитал портфеля в момент $t$
- $w_t = \pi_t / X_t$ — вектор весов (доли капитала)
- $\mu \in \mathbb{R}^n$ — вектор ожидаемых годовых доходностей
- $\Sigma \in \mathbb{R}^{n \times n}$ — ковариационная матрица доходностей
- $W_t$ — $n$-мерное броуновское движение
- $\rho > 0$ — субъективная норма дисконтирования

**HJB-уравнение** (при отсутствии потребления):

$$\rho V(X) = \max_{\pi} \\{ \pi^T\mu \cdot V'(X) + \frac{1}{2}\pi^T\Sigma\pi \cdot V''(X) \\}$$

**Аналитическое решение** (Мертон, 1969) при $V(X) = \frac{X^{1-\gamma}}{1-\gamma}$:

$$w^* = \frac{1}{\gamma} \Sigma^{-1} \mu$$

### 2. Применение ограничений

**Ограничение 1: Запрет коротких продаж** ($w_i \geq 0$)

$$w_{\text{clipped}} = \max(w^*, 0)$$

**Ограничение 2: Полное инвестирование** ($\sum_{i=1}^n w_i = 1$)

$$w_{\text{final}, i} = \frac{w_{\text{clipped}, i}}{\sum_{j=1}^n w_{\text{clipped}, j}}$$

### 3. GARCH(1,1) моделирование волатильности

Условная волатильность актива $i$:

$$\sigma_i^2(t) = \omega_i + \alpha_i \varepsilon_i^2(t-1) + \beta_i \sigma_i^2(t-1)$$

где $\varepsilon_i(t) = \sigma_i(t) z_t$, $z_t \sim N(0, 1)$.

**Условие стационарности:** $\alpha_i + \beta_i < 1$

**Долгосрочная волатильность:**

$$\sigma_i^2(\infty) = \frac{\omega_i}{1 - \alpha_i - \beta_i}$$

Параметры $(\omega_i, \alpha_i, \beta_i)$ оцениваются методом максимального правдоподобия (MLE).

**Критерии фильтрации активов:**
1. Стационарность: $\alpha_i + \beta_i < 0.995$
2. Сходимость оптимизации MLE
3. Положительность параметров: $\omega_i > 0, \alpha_i \geq 0, \beta_i \geq 0$
4. Разумная годовая волатильность: $5\% < \sigma_{\text{annual}}^i < 100\%$

### 4. Ковариационная матрица

Выборочная ковариационная матрица доходностей:

$$\hat{\Sigma} = \frac{1}{T-1} \sum_{t=1}^T (r(t) - \bar{r})(r(t) - \bar{r})^T$$

**Проверка положительной определённости:** вычисляются собственные числа $\lambda_1, \ldots, \lambda_n$. Если $\lambda_{\min} \leq 0$, применяется регуляризация Тихонова:

$$\Sigma_{\text{reg}} = \hat{\Sigma} + \epsilon I, \quad \epsilon = \max(10^{-6}, 10|\lambda_{\min}|)$$

Матрица корреляций:

$$\rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}}$$

### 5. Расчет доходностей

**Логарифмическая доходность за день:**

$$r_i(t) = \ln\left(\frac{P_i(t)}{P_i(t-1)}\right)$$

**Средняя годовая доходность от прироста капитала:**

$$\mu_i^{\text{capital}} = \bar{\mu}_i^{\text{daily}} \times 252$$

где $\bar{\mu}_i^{\text{daily}} = \frac{1}{T-1} \sum_{t=2}^T r_i(t)$

**Дивидендная доходность:**

$$y_i^{\text{div}} = \frac{\sum_{k=1}^{N_{\text{div}}} D_i(t_k)}{P_i(t_{\text{current}})}$$

**Полная ожидаемая доходность:**

$$\mu_i = \mu_i^{\text{capital}} + y_i^{\text{div}}$$

**Корректировка за корпоративные действия** (сплиты акций):

$$P_{\text{adj}}^i(t) = \begin{cases} P^i(t) \cdot m, & t < t_{\text{split}} \\ P^i(t), & t \geq t_{\text{split}} \end{cases}$$

где $m$ — коэффициент сплита.

### 6. Монте-Карло симуляция

**Дискретизация Эйлера-Маруямы:**

$$X_{t+\Delta t} = X_t \left(1 + w^T\mu \cdot \Delta t + w^T L Z_t \sqrt{\Delta t}\right)$$

где:
- $\Delta t = 1/252$ (шаг = 1 торговый день)
- $L$ — нижняя треугольная матрица разложения Cholesky ($\Sigma = LL^T$)
- $Z_t \sim N(0, I_n)$ — стандартный нормальный вектор

**Параметры симуляции:**
- Горизонт: $T = 1$ год (настраивается)
- Число шагов: $N = 252$ (торговые дни)
- Число траекторий: $M = 5000$ (настраивается)
- Начальный капитал: $X_0 = 1\,000\,000$ (произвольно)

**Вывод:** матрица траекторий размер $M \times (N+1)$

### 7. Метрики риска и доходности

**Доходность портфеля (траектория $j$):**

$$r_j = \frac{X_j(T) - X_0}{X_0}$$

**Средняя доходность:**

$$\bar{r} = \frac{1}{M} \sum_{j=1}^M r_j$$

**Волатильность (стандартное отклонение):**

$$\hat{\sigma}_{\text{port}} = \sqrt{\frac{1}{M-1} \sum_{j=1}^M (r_j - \bar{r})^2}$$

**Коэффициент Шарпа:**

$$\text{Sharpe} = \frac{\bar{r} - r_f}{\hat{\sigma}_{\text{port}}}$$

где $r_f = 16.5\%$ (ставка ЦБ РФ на дату анализа, задается вручную).

**Value at Risk (VaR) на уровне α:**

$$\text{VaR}_\alpha = Q_\alpha(\{r_j\}_{j=1}^M)$$

где $Q_\alpha$ — квантиль уровня $\alpha$. Для $\alpha = 0.05$ (95% доверительный уровень).

**Conditional Value at Risk (CVaR):**

$$\text{CVaR}_\alpha = \mathbb{E}[r | r \leq \text{VaR}_\alpha] = \frac{1}{|\{j : r_j \leq \text{VaR}_\alpha\}|} \sum_{j: r_j \leq \text{VaR}_\alpha} r_j$$

**Maximum Drawdown (MDD) для траектории $j$:**

$$\text{MDD}_j = \max_{0 \leq s \leq t \leq T} \frac{X_j(s) - X_j(t)}{X_j(s)} = \max_{0 \leq t \leq T} \left(1 - \frac{X_j(t)}{\max_{0 \leq s \leq t} X_j(s)}\right)$$

**Средний MDD:**

$$\text{MDD} = \frac{1}{M} \sum_{j=1}^M \text{MDD}_j$$

---

## Ограничения и чувствительность

### Ограничение 1: Запрет коротких продаж

При наложении ограничения $w_i \geq 0$ решение Мертона после проекции становится менее эффективным по Парето. Для некоторых портфелей весь вес может сконцентрироваться на нескольких активах.

### Ограничение 2: Инвариантность w* к γ

При наличии ограничений на короткие продажи оптимальные веса в диапазоне γ ∈ [1.0, 10.0] практически инвариантны к γ. Это связано с тем, что безограниченное решение Мертона содержит отрицательные компоненты, которые обнуляются при проекции.

**Для проявления влияния γ требуется:**
- Разрешение коротких продаж
- Или использование других ограничений (например, лимиты концентрации)

### Ограничение 3: Параметрический риск

Оценки µ и Σ подвержены **sampling error**, особенно при малых выборках T.

**Рекомендации:**
- Использовать исторический период минимум 2-3 года (500+ торговых дней)
- Регулярно пересчитывать параметры (ежемесячно или ежеквартально)
- Применять сжатие (shrinkage) оценок Ledoit-Wolf для матрицы Σ

### Ограничение 4: GARCH(1,1) стационарность

Условие $\alpha + \beta < 1$ необходимо для стационарности условной дисперсии. Активы с $\alpha + \beta \geq 0.995$ автоматически исключаются.

### Чувствительность к ключевым параметрам

| Параметр | Влияние | Рекомендация |
|----------|--------|--------------|
| **r_f** (безрисковая ставка) | Высокое на Sharpe Ratio | Обновлять для текущей даты |
| **γ (CRRA)** | Низкое при no short-sales | Использовать 3.0 по умолчанию |
| **M (траектории)** | Сходимость улучшается | ≥ 5000 для надежности |
| **Коррелирующие активы** | Концентрация портфеля | Фильтровать ρ > 0.95 |
| **Волатильность σ** | Очень высокое | Пересчитывать ежемесячно |

---

## Стресс-тестирование

### Базовые сценарии

| Сценарий | µ | Σ | Описание |
|----------|---|---|---------|
| **Baseline** | µ | Σ | Текущие рыночные условия |
| **Crisis** | 0.7µ | Σ | Снижение доходностей на 30% (кризис 2008) |
| **High Vol** | µ | 2.25Σ | Увеличение волатильности на 50% (COVID-19) |
| **Corr Shock** | µ | D·P*·D | Увеличение корреляций на 50% (потеря диверсификации) |
| **Individual** | µ* → 0.5µ* | Σ | Худший актив теряет 50% доходности |

### Экстремальные сценарии

| Сценарий | µ | Σ | Описание |
|----------|---|---|---------|
| **Black Swan** | 0.5µ | Σ | Экстремальный кризис (~1% вероятность в год) |
| **Bull Market** | 1.5µ | Σ | Продолжительный период роста |
| **Low Vol** | µ | 0.25Σ | Снижение волатильности на 50% |

### Анализ временных горизонтов

Симуляция проводится для горизонтов: **1 месяц, 3 месяца, 1 год, 3 года**

$T_h \in \{1/12, 1/4, 1, 3\}$ года, $N_h = \lfloor T_h \times 252 \rfloor$

### Анализ чувствительности к γ

Γ ∈ {1.0, 2.0, 3.0, 5.0, 10.0}

**Замечание:** При ограничениях на короткие продажи и полное инвестирование оптимальные веса в диапазоне [1.0, 10.0] остаются практически инвариантны к γ, так как решение Мертона содержит много отрицательных компонент. После проекции на положительный ортант эффект γ становится пренебрежимо мал.

---

## Структура репозитория

```
stochastic-portfolio-optimization/
├── README.md                          # Этот файл
├── requirements.txt                   # Зависимости Python
├── .gitignore                         # Git исключения
├── setup.py                           # Установка пакета (опционально)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Загрузка данных с MOEX ISS
│   ├── parameter_estimation.py        # GARCH, ковариация, доходности
│   ├── portfolio_optimization.py      # HJB решение, проекции
│   ├── monte_carlo_simulator.py       # Монте-Карло симуляция
│   ├── risk_metrics.py                # Расчет VaR, CVaR, MDD, Sharpe
│   ├── stress_testing.py              # Стресс-сценарии
│   └── utils.py                       # Вспомогательные функции
│
├── notebooks/
│   └── full_analysis.ipynb            # Полный анализ с примерами
│
├── examples/
│   ├── basic_example.py               # Простой пример
│   ├── moex_portfolio.py              # Пример с MOEX тикерами
│   └── stress_test_report.py          # Подробный стресс-тест
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_optimization.py
│   ├── test_metrics.py
│   └── test_stress_testing.py
│
├── data/
│   └── .gitkeep                       # Для локальных данных (не в git)
│
└── outputs/
    └── .gitkeep                       # Для результатов (не в git)
```

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

## Установка

### Требования

- Python 3.8+
- pip или conda

### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/yourusername/stochastic-portfolio-optimization.git
cd stochastic-portfolio-optimization
```

### Шаг 2: Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows
```

### Шаг 3: Установка зависимостей

```bash
pip install -r requirements.txt
```

### Требуемые пакеты

```txt
numpy>=1.24.0
pandas>=1.5.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
arch>=5.3.0           # GARCH моделирование
requests>=2.28.0      # Загрузка данных из API
openpyxl>=3.10.0      # Excel экспорт
```

---

## Лицензия

MIT License. See LICENSE file for details.

---

## Контакты и поддержка

- **Issues:** Используйте GitHub Issues для отчетов об ошибках
- **Discussions:** Используйте GitHub Discussions для вопросов
- **Email:** [lifeofwhitedesiigner@gmail.com]

---

## Цитирование

Если вы используете этот код в научной работе, пожалуйста цитируйте:

```bibtex
@software{stochastic_portfolio_hjb,
  author = {Your Name},
  title = {Stochastic Portfolio Optimization via HJB Strategy with Stress-Testing},
  year = {2025},
  url = {https://github.com/yourusername/stochastic-portfolio-optimization}
}
```

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
