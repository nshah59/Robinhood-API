import robin_stocks as rb
import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt


def test_for_cointegration(X1, X2):
    X1_historical = rb.get_stock_historicals(X1,
                                             interval="hour",
                                             span="month",
                                             info="close_price")
    X2_historical = rb.get_stock_historicals(X2,
                                             interval="hour",
                                             span="month",
                                             info="close_price")
    p = coint(X1_historical, X2_historical)[1]
    if p < 0.01:
        return True
    else:
        return False


def check_spread(symbols):
    X1 = symbols[0]
    X2 = symbols[1]
    X1_historical = pd.Series(rb.get_stock_historicals(X1,
                                                       interval="hour",
                                                       span="week",
                                                       info="close_price"))

    X2_historical = pd.Series(rb.get_stock_historicals(X2,
                                                       interval="hour",
                                                       span="week",
                                                       info="close_price"))
    X1_historical = sm.add_constant(X1_historical)
    X1_historical = np.array(X1_historical, dtype=float)
    X2_historical = np.array(X2_historical, dtype=float)
    results = sm.OLS(X2_historical, X1_historical).fit()
    b = results.params[1]
    X1_historical = X1_historical[:, 1]
    length = len(X1_historical)
    spread = X2_historical - b * X1_historical
    spread_mean = np.mean(spread)
    spread_sd = np.std(spread)
    low, high = spread_mean - 20 * spread_sd / length, spread_mean + 20 * spread_sd / length
    lastX1 = X1_historical[-1]
    lastX2 = X2_historical[-1]
    curr_spread = lastX2 - lastX1 * b
    if curr_spread < low:
        print("Sell ", X1, ", Buy ", X2)
        print("Close when ", X2, " - ", b, " * ", X1, " >= ", low)
    elif curr_spread > high:
        print("Sell ", X2, ", Buy ", X1)
        print("Close when ", X2, " - ", b, " * ", X1, " <= ", high)
    else:
        print(X1, " and ", X2, " are in reasonable range.")
    return True


def main():
    rb.login(username, password)
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]

    df.to_csv('S&P500-Info.csv')
    df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])
    cols = df.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
    df.columns = cols
    industries = set(df["GICS_Sub_Industry"])
    coint_symbols = []

    for industry in industries:
        print(industry)
        curr_sector = df.loc[df["GICS_Sub_Industry"] == industry].Symbol
        for x in curr_sector:
            for y in curr_sector:
                if x != y:
                    if test_for_cointegration(x, y):
                        coint_symbols.append([x, y])
                        print(coint_symbols)

    for symbol in coint_symbols:
        check_spread(symbol)


if __name__ == "__main__":
    main()
