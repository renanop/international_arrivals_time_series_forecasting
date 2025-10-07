import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import numpy as np
from typing import Callable


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
        df[value_col] = df[value_col].apply(np.log)

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