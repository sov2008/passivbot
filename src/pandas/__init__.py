"""Minimal subset of the :mod:`pandas` API for the kata test-suite.

The real project depends on :mod:`pandas`, however the execution environment
used for the kata does not provide binary wheels.  This module provides enough
functionality for the accompanying unit tests.  It purposefully implements only
the features exercised in the tests; any other functionality should be
considered out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def isna(value) -> bool:
    return np.isna(value)


def notna(value) -> bool:
    return not isna(value)


class _ILocIndexer:
    def __init__(self, data: List, setter=None):
        self._data = data
        self._setter = setter

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        if self._setter is None:
            raise AttributeError("Series is read-only")
        self._setter(key, value)


@dataclass
class Series:
    _data: List
    name: Optional[str] = None

    def copy(self):
        return Series(self._data[:], self.name)

    @property
    def iloc(self):
        return _ILocIndexer(self._data, self._set_iloc)

    def _set_iloc(self, index, value):
        self._data[index] = value

    def astype(self, dtype):
        if dtype in (float, "float", "float64", np.float64):
            return Series([float(x) for x in self._data], self.name)
        return Series([dtype(x) for x in self._data], self.name)

    @property
    def values(self):
        return np.array(self._data)

    def tolist(self):
        return list(self._data)

    def fillna(self, value):
        if isinstance(value, Series):
            filled = [v if not isna(v) else value._data[i] for i, v in enumerate(self._data)]
        else:
            filled = [v if not isna(v) else value for v in self._data]
        return Series(filled, self.name)

    def ffill(self):
        out = []
        last = None
        for v in self._data:
            if isna(v):
                out.append(last)
            else:
                out.append(v)
                last = v
        return Series(out, self.name)

    def diff(self):
        if not self._data:
            return Series([], self.name)
        diffs = [np.nan]
        for i in range(1, len(self._data)):
            a = self._data[i]
            b = self._data[i - 1]
            if isna(a) or isna(b):
                diffs.append(np.nan)
            else:
                diffs.append(a - b)
        return Series(diffs, self.name)

    def max(self):
        clean = [v for v in self._data if not isna(v)]
        return max(clean) if clean else np.nan

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def _compare(self, other, op):
        if isinstance(other, Series):
            other_vals = other._data
        else:
            other_vals = [other] * len(self._data)
        return Series([op(a, b) for a, b in zip(self._data, other_vals)], name=self.name)

    def __ne__(self, other):
        return self._compare(other, lambda a, b: a != b)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)

    def _binary(self, other, op):
        if isinstance(other, Series):
            other_vals = other._data
        else:
            other_vals = [other] * len(self._data)
        return Series([op(a, b) for a, b in zip(self._data, other_vals)], name=self.name)

    def __floordiv__(self, other):
        return self._binary(other, lambda a, b: a // b)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)


class _LocIndexer:
    def __init__(self, df: "DataFrame"):
        self._df = df

    def __getitem__(self, key):
        row_key, col_key = key
        row_idx = self._df._resolve_index(row_key)
        return self._df._data[col_key][row_idx]

    def __setitem__(self, key, value):
        row_key, col_key = key
        row_idx = self._df._resolve_index(row_key)
        self._df._data[col_key][row_idx] = value


class DataFrame:
    def __init__(self, data, columns: Optional[Sequence[str]] = None, index: Optional[List] = None):
        if isinstance(data, dict):
            cols = list(data.keys())
            values = [list(data[col]) for col in cols]
            columns = cols
            rows = list(zip(*values)) if values else []
        else:
            rows = [list(row) for row in data]
        if columns is None and rows:
            columns = [str(i) for i in range(len(rows[0]))]
        self._columns = list(columns or [])
        self._data: Dict[str, List] = {col: [] for col in self._columns}
        for row in rows:
            for col, value in zip(self._columns, row):
                self._data[col].append(value)
        if index is None:
            self._index = list(range(len(rows)))
        else:
            self._index = list(index)

    # ------------------------------------------------------------------
    def copy(self) -> "DataFrame":
        new = DataFrame([], columns=self._columns, index=self._index[:])
        new._data = {col: values[:] for col, values in self._data.items()}
        return new

    def __len__(self) -> int:
        return len(self._index)

    @property
    def columns(self):
        return self._columns[:]

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def shape(self):
        return (len(self._index), len(self._columns))

    def __iter__(self):
        return iter(self._columns)

    @property
    def values(self):
        rows = []
        for i in range(len(self)):
            rows.append([self._data[col][i] for col in self._columns])
        return np.array(rows)

    def to_dict(self):
        return {col: values[:] for col, values in self._data.items()}

    def astype(self, dtype):
        new = {}
        for col, values in self._data.items():
            if dtype in (float, "float", "float64", np.float64):
                new[col] = [float(v) for v in values]
            else:
                new[col] = [dtype(v) for v in values]
        return DataFrame(new, index=self._index)

    def set_index(self, column: str):
        new = self.copy()
        new._index = new._data[column][:]
        del new._data[column]
        new._columns.remove(column)
        return new

    def _resolve_index(self, key):
        if isinstance(key, int) and key in range(len(self._index)):
            return key
        if key in self._index:
            return self._index.index(key)
        raise KeyError(key)

    def reindex(self, new_index: Sequence):
        new = DataFrame([], columns=self._columns, index=list(new_index))
        for col in self._columns:
            new_vals = []
            for idx in new_index:
                if idx in self._index:
                    pos = self._index.index(idx)
                    new_vals.append(self._data[col][pos])
                else:
                    new_vals.append(np.nan)
            new._data[col] = new_vals
        return new

    def reset_index(self):
        new = self.copy()
        new._data = {**{"index": self._index[:] }, **self._data}
        new._columns = ["index", *self._columns]
        new._index = list(range(len(self._index)))
        return new

    def rename(self, *, columns: Dict[str, str]):
        new = self.copy()
        mapping = {**columns}
        new._columns = [mapping.get(col, col) for col in self._columns]
        new._data = {mapping.get(col, col): values[:] for col, values in self._data.items()}
        return new

    def sort_values(self, column: str):
        paired = list(zip(self._index, *[self._data[col] for col in self._columns]))
        col_idx = self._columns.index(column)
        paired.sort(key=lambda row: row[1 + col_idx])
        new_index = [row[0] for row in paired]
        new_data = {col: [row[1 + idx] for row in paired] for idx, col in enumerate(self._columns)}
        new = DataFrame(new_data, index=new_index)
        return new

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self._data[key][:], key)
        if isinstance(key, list):
            data = {col: self._data[col][:] for col in key}
            return DataFrame(data, index=self._index)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, Series):
            self._data[key] = value._data[:]
        else:
            if not isinstance(value, list):
                value = [value] * len(self._index)
            self._data[key] = list(value)
        if key not in self._columns:
            self._columns.append(key)

    def __getattr__(self, item):
        if item in self._data:
            return Series(self._data[item][:], item)
        raise AttributeError(item)

    @property
    def loc(self):
        return _LocIndexer(self)

    def groupby(self, by, as_index=False):
        return _GroupBy(self, by, as_index=as_index)


def DataFrameFactory(data=None, columns=None):
    data = data or []
    return DataFrame(data, columns=columns)


DataFrame = DataFrame
Series = Series


def DataFrame_from_dict(data: Dict[str, Sequence]):
    return DataFrame(data)


def concat(frames: Sequence[DataFrame]):
    if not frames:
        return DataFrame([])
    data = {col: [] for col in frames[0]._columns}
    for frame in frames:
        for col in frame._columns:
            data[col].extend(frame._data[col])
    return DataFrame(data)


def read_csv(*args, **kwargs):  # pragma: no cover - not used in tests
    raise NotImplementedError("read_csv is not available in the lightweight pandas stub")


def read_excel(*args, **kwargs):  # pragma: no cover - not used in tests
    raise NotImplementedError("read_excel is not available in the lightweight pandas stub")

class _GroupBy:
    def __init__(self, df: DataFrame, by: str, as_index: bool = False):
        self._df = df
        self._by = by
        self._as_index = as_index

    def __getitem__(self, column: str):
        return _GroupByColumn(self._df, self._by, column, as_index=self._as_index)


class _GroupByColumn:
    def __init__(self, df: DataFrame, by: str, column: str, as_index: bool = False):
        self._df = df
        self._by = by
        self._column = column
        self._as_index = as_index

    def sum(self):
        groups = {}
        by_values = self._df._data[self._by]
        col_values = self._df._data[self._column]
        for key, value in zip(by_values, col_values):
            groups.setdefault(key, 0.0)
            groups[key] += value
        if self._as_index:
            data = {self._by: list(groups.keys()), self._column: list(groups.values())}
        else:
            data = {self._by: list(groups.keys()), self._column: list(groups.values())}
        return DataFrame(data)
