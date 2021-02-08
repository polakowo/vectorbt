import pandas as pd
import warnings


def yf_downloader(symbols, **kwargs):
    """Downloader that uses `yfinance`."""
    import yfinance as yf

    return {s: yf.Ticker(s).history(**kwargs) for s in symbols}


def download(symbols, downloader=yf_downloader, cols=None, **kwargs):
    """Download data for each symbol using `downloader` function.

    `downloader` should accept a list of symbols and optionally keyword arguments,
    and return a dictionary of dataframes/series keyed by symbol.

    Returns a dictionary if `symbols` is a list, otherwise returns a single dataframe/series."""

    if isinstance(symbols, list):
        data = downloader(symbols, **kwargs)
        if cols is not None:
            data = {k: v[cols] for k, v in data.items()}
    else:
        data = downloader([symbols], **kwargs)[symbols]
        if cols is not None:
            data = data[cols]
    return data


def concat_symbols(data, level_name='symbol', treat_missing='nan'):
    """Return a dict of dataframes with symbols as columns.

    The argument `treat_missing` accepts the following values:

    * `'nan'`: set missing data points to NaN
    * `'drop'`: remove missing data points

    Returns a dictionary if `data` contains dataframes, otherwise returns a single dataframe."""
    if treat_missing not in ['nan', 'drop']:
        raise ValueError("treat_missing can be either 'nan' or 'drop'")
    symbols = list(data.keys())
    single_column = False
    if isinstance(data[symbols[0]], pd.Series):
        data = {k: v.to_frame() for k, v in data.items()}
        single_column = True
    columns = data[symbols[0]].columns
    index = None
    for s in symbols:
        if index is None:
            index = data[s].index
        else:
            if len(index.intersection(data[s].index)) != len(index.union(data[s].index)):
                if treat_missing == 'nan':
                    warnings.warn("Symbols have mismatching timestamps. "
                                  "Setting missing data points to NaN.", stacklevel=2)
                    index = index.union(data[s].index)
                else:
                    warnings.warn("Symbols have mismatching timestamps. "
                                  "Dropping missing data points.", stacklevel=2)
                    index = index.intersection(data[s].index)
    if len(symbols) > 1:
        new_data = {c: pd.DataFrame(index=index, columns=pd.Index(symbols, name=level_name)) for c in columns}
    else:
        new_data = {c: pd.Series(index=index, name=symbols[0]) for c in columns}
    for c in columns:
        for s in symbols:
            if treat_missing == 'nan':
                col_data = data[s].loc[:, c]
            else:
                col_data = data[s].loc[index, c]
            if len(symbols) > 1:
                new_data[c].loc[:, s] = col_data
            else:
                new_data[c].loc[:] = col_data
    if single_column:
        return new_data[columns[0]]
    return new_data
