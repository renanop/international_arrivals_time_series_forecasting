import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import numpy as np
from typing import Callable
from tqdm import tqdm


def plot_stl_decomposition(df, value_col, date_col, log=False, period=12, title="STL Decomposition"):
    """
    Realiza a decomposição STL de uma série temporal e plota seus componentes.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo a série temporal.
    value_col : str
        Nome da coluna com os valores da série temporal (ex: 'arrivals').
    date_col : str
        Nome da coluna com as datas (ex: 'date').
    log : bool
        Se True, calcula-se a decomposicão de forma multiplicativa através da transformacão logarítmica da série original
    period : int, opcional
        Período sazonal da decomposição STL (padrão = 12 para dados mensais).
    title : str, opcional
        Título principal do gráfico.

    Retorna:
    --------
    dcmp : pd.DataFrame
        DataFrame com colunas: data original, tendência, sazonalidade e resíduo.
    """

    df = df.copy()
    # Se escolhe-se a escala logaritmica, calcula-se a decomposicao com log e as componentes na escala original
    if log:
        # Transforma para a escala log
        df[value_col] = df[value_col].apply(np.log1p)

        # Ajusta o STL
        stl = STL(df[value_col], period=period)
        res_stl = stl.fit()

        # cria o dataframe com os componentes
        dcmp = pd.DataFrame({
            "ds": df[date_col],
            "data": df[value_col],
            "trend": res_stl.trend,
            "seasonal": res_stl.seasonal,
            "remainder": res_stl.resid,
            "data_original_scale": np.exp(df[value_col]),
            "trend_original_scale": np.exp(res_stl.trend),
            "seasonal_original_scale": np.exp(res_stl.seasonal),
            "remainder_original_scale": np.exp(res_stl.resid),
            "detrend_original_scale": np.exp( df[value_col] - res_stl.trend )
        }).reset_index(drop=True)

    else:
        # aplica a decomposição STL
        stl = STL(df[value_col], period=period)
        res_stl = stl.fit()

        # cria o dataframe com os componentes
        dcmp = pd.DataFrame({
            "ds": df[date_col],
            "data": df[value_col],
            "trend": res_stl.trend,
            "seasonal": res_stl.seasonal,
            "remainder": res_stl.resid
        }).reset_index(drop=True)

    # plota os componentes
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 9))
    sns.lineplot(data=dcmp, x=dcmp.index, y="data", ax=axes[0])
    sns.lineplot(data=dcmp, x=dcmp.index, y="trend", ax=axes[1])
    sns.lineplot(data=dcmp, x=dcmp.index, y="seasonal", ax=axes[2])
    sns.lineplot(data=dcmp, x=dcmp.index, y="remainder", ax=axes[3])

    axes[0].set_ylabel(value_col)
    axes[1].set_ylabel("Trend")
    axes[2].set_ylabel("Seasonal")
    axes[3].set_ylabel("Remainder")

    fig.suptitle(title)
    fig.subplots_adjust(top=0.90)
    plt.xlabel("")

    # ajusta titulo de acordo com forma de decompor
    if log:
        fig.text(0.5, 0.92, "log(Observed) = log(Trend) + log(Seasonal) + log(Remainder)", ha="center")
    else:
        fig.text(0.5, 0.92, "Observed = Trend + Seasonal + Remainder", ha="center")

    plt.show()

    return dcmp


def process_df(df:pd.DataFrame, state:str, agg_cols:list, value_cols:list, agg_func:Callable) -> pd.DataFrame:
    """A function that filters the dataframe for a specific state and uses a aggregation function
    to aggregate the data given a set of aggregation columns and a value column."""
    # Copying dataframe to avoid problems
    df = df.copy()

    # Filtering state
    df = df.loc[df["state"] == state, :]

    # Group by desired columns
    return df.groupby(agg_cols)[value_cols].agg(agg_func).reset_index()



def block_bootstrap_stl_residuals(
    dcmp: pd.DataFrame,
    block_size: int,
    simulations: int,
    remainder_col: str = "remainder",
    trend_col: str = "trend",
    seasonal_col: str = "seasonal",
    date_col: str = "ds",
    log: bool = False,
    return_original_scale: bool = True,
    random_state: int | None = None
) -> pd.DataFrame:
    """
    Realiza block bootstrapping dos resíduos da decomposição STL e reconstrói
    séries simuladas.

    Parâmetros
    ----------
    dcmp : pd.DataFrame
        DataFrame retornado pela decomposição STL. Deve conter pelo menos:
        [date_col, trend_col, seasonal_col, remainder_col].
    block_size : int
        Tamanho de cada bloco de resíduos (ex.: 8 para dados trimestrais,
        12 para mensais, etc.).
    simulations : int
        Número de séries simuladas a gerar.
    remainder_col, trend_col, seasonal_col, date_col : str
        Nomes das colunas no dcmp.
    log : bool
        Se True, assume que trend/seasonal/remainder estão na escala log
        (como no seu plot_stl_decomposition(log=True)).
    return_original_scale : bool
        Quando log=True, se True devolve as séries simuladas na escala original
        (exp). Se False, devolve na escala log.
    random_state : int | None
        Semente para reprodutibilidade.

    Retorna
    -------
    sim_df : pd.DataFrame
        DataFrame wide com colunas:
        [date_col, sim_0, sim_1, ..., sim_{simulations-1}]
    """

    if random_state is not None:
        np.random.seed(random_state)

    n = len(dcmp)
    if block_size <= 0:
        raise ValueError("block_size deve ser >= 1.")
    if block_size > n:
        raise ValueError("block_size não pode ser maior que o tamanho da série.")

    # número de blocos para cobrir a série toda
    num_blocks = int(np.ceil(n / block_size))

    sims = np.empty((n, simulations), dtype=float)

    remainder_series = dcmp[remainder_col].reset_index(drop=True)
    base_level = (dcmp[trend_col] + dcmp[seasonal_col]).reset_index(drop=True)

    for s in range(simulations):
        blocks = []
        for _ in range(num_blocks):
            start = np.random.randint(0, n - block_size + 1)
            end = start + block_size
            blocks.append(remainder_series.iloc[start:end])

        boot_remainder = pd.concat(blocks, ignore_index=True).iloc[:n].to_numpy()
        sims[:, s] = base_level.to_numpy() + boot_remainder

    sim_df = pd.DataFrame(
        sims,
        columns=[f"sim_{i}" for i in range(simulations)]
    )
    sim_df.insert(0, date_col, dcmp[date_col].values)

    # Se STL foi feita no log, opcionalmente volta para a escala original
    if log and return_original_scale:
        for c in sim_df.columns:
            if c != date_col:
                sim_df[c] = np.exp(sim_df[c])

    return sim_df



def stl_baseline_sem_covid(df, value_col, date_col, covid_mask, period=12, log=False):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # série para STL só fora da COVID
    df_nocovid = df.loc[~covid_mask, [date_col, value_col]].copy()

    # roda sua STL existente
    dcmp_nocovid = plot_stl_decomposition(df_nocovid, value_col, date_col, log=log, period=period)

    # baseline fora da covid
    baseline_nocovid = dcmp_nocovid["trend"] + dcmp_nocovid["seasonal"]

    # reindexa para série completa
    baseline_full = (
        pd.Series(baseline_nocovid.values, index=dcmp_nocovid[date_col])
          .reindex(df[date_col])
          .interpolate("time")   # preenche COVID “na continuidade”
          .bfill().ffill()
    )

    df["baseline"] = baseline_full.values

    # intervenção = diferença no período COVID
    df["intervention"] = df[value_col] - df["baseline"]

    # resíduos “normais” = fora COVID
    df["remainder_normal"] = df[value_col] - df["baseline"] - df["intervention"].where(covid_mask, 0)

    return df



import numpy as np
import pandas as pd

# precisa ter pmdarima instalado
# pip install pmdarima
from pmdarima import auto_arima


def fit_sarima_on_simulations_long(
    sim_df: pd.DataFrame,
    h: int,
    date_col: str = "ds",
    simulation_prefix: str = "sim_",
    seasonal: bool = True,
    m: int | None = None,
    freq: str | None = None,
    auto_arima_kwargs: dict | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Lê um DataFrame wide de simulações (saída do block_bootstrap_stl_residuals),
    ajusta um SARIMA ótimo em cada simulação (auto_arima) e gera previsões h passos
    à frente. Retorna dataset long com labels train/forecast.

    Parâmetros
    ----------
    sim_df : pd.DataFrame
        Saída wide com colunas [date_col, sim_0, sim_1, ...].
    h : int
        Horizonte de previsão (X períodos à frente).
    date_col : str
        Nome da coluna de datas.
    simulation_prefix : str
        Prefixo que identifica colunas de simulação.
    seasonal : bool
        Se True, permite componente sazonal no auto_arima.
    m : int | None
        Período sazonal (ex.: 12 mensal, 4 trimestral). Se None, tenta inferir.
    freq : str | None
        Frequência do índice temporal para criar datas futuras (ex.: "MS", "M", "Q").
        Se None, tenta inferir pela série de datas.
    auto_arima_kwargs : dict | None
        Dicionário extra de parâmetros para auto_arima.
    random_state : int | None
        Semente para reprodutibilidade em auto_arima quando aplicável.
    verbose : bool
        Imprime progresso se True.

    Retorna
    -------
    long_df : pd.DataFrame
        Colunas:
        - date_col: datas (treino + futuro)
        - simulation: label da simulação (sim_0, sim_1, ...)
        - set: "train" ou "forecast"
        - value: valor observado/treino ou previsto
    """

    if h <= 0:
        raise ValueError("h deve ser >= 1")

    df = sim_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # identificar colunas de simulação
    sim_cols = [c for c in df.columns if c.startswith(simulation_prefix)]
    if len(sim_cols) == 0:
        raise ValueError(f"Nenhuma coluna começando com '{simulation_prefix}' encontrada.")

    # inferir frequência (para datas futuras)
    if freq is None:
        freq = pd.infer_freq(df[date_col])
        if freq is None:
            # fallback: usa mediana dos deltas
            deltas = df[date_col].diff().dropna()
            median_delta = deltas.median()
            # tenta converter em freq aproximada (dia)
            if pd.isna(median_delta):
                raise ValueError("Não consegui inferir freq; passe freq manualmente.")
            freq = median_delta  # DateOffset/Timedelta ok para date_range

    # inferir m se não veio
    if m is None and seasonal:
        # regra simples: tenta mapear freq para um m típico
        # mensal -> 12, trimestral -> 4, semanal -> 52, diário -> 7 (se sazonal semanal)
        if isinstance(freq, str):
            f = freq.upper()
            if "M" in f:      # M, MS, BM, BMS...
                m = 12
            elif "Q" in f:    # Q, QS...
                m = 4
            elif "W" in f:
                m = 52
            elif "D" in f:
                m = 7
            else:
                m = 1
        else:
            # freq como Timedelta/DateOffset -> não dá pra mapear robusto
            m = 1

    # kwargs padrão do auto_arima
    base_kwargs = dict(
        seasonal=seasonal,
        m=m if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=verbose,
        random_state=random_state
    )
    if auto_arima_kwargs:
        base_kwargs.update(auto_arima_kwargs)

    last_date = df[date_col].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=h+1, freq=freq)[1:]

    long_parts = []

    for col in tqdm(sim_cols):
        y = df[col].astype(float).to_numpy()

        if verbose:
            print(f"Ajustando auto_arima para {col} (n={len(y)})...")

        # auto_arima -> melhor SARIMA
        model = auto_arima(y, **base_kwargs)

        y_fore = model.predict(n_periods=h)

        # treino em long
        train_part = pd.DataFrame({
            date_col: df[date_col].values,
            "simulation": col,
            "set": "train",
            "value": y
        })

        # forecast em long
        fore_part = pd.DataFrame({
            date_col: future_dates,
            "simulation": col,
            "set": "forecast",
            "value": y_fore
        })

        long_parts.append(train_part)
        long_parts.append(fore_part)

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = long_df.sort_values(["simulation", date_col]).reset_index(drop=True)

    return long_df
