"""A lightweight NumPy compatibility layer used for the kata test-suite.

This module implements a *very* small subset of the :mod:`numpy` API that is
required by the unit tests in this kata.  The goal is not to be fast nor fully
compatible with real NumPy; the implementation focuses solely on the features
that are exercised by the tests.  It supports two dimensional ``float64``
arrays and provides enough functionality for indexing, slicing, broadcasting of
scalar operations, a handful of helper functions (``array``, ``empty``,
``concatenate`` …) and persistence helpers (``save``/``load``).

Only the pieces that are used in the tests are implemented.  The behaviour is
documented in the docstrings below – whenever real NumPy disagrees we favour a
predictable pure Python behaviour rather than perfect fidelity.

The module is intentionally self contained and avoids external dependencies so
that the exercises can run in environments where native wheels cannot be
installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable, Iterator, List, Sequence, Tuple, Union
from builtins import all as _py_all, abs as _py_abs


Number = Union[int, float]


def _ensure_sequence(value: Union[Sequence, "ndarray"]) -> Sequence:
    if isinstance(value, ndarray):
        return value.tolist()
    return value


@dataclass
class ndarray:
    """Very small ndarray implementation supporting 1D and 2D float arrays."""

    _data: List[List[float]]
    _shape: Tuple[int, ...]

    def __post_init__(self) -> None:  # normalise 1D representation
        if len(self._shape) == 1:
            if len(self._data) == 1 and len(self._data[0]) == self._shape[0]:
                values = self._data[0]
            else:
                values = [row[0] if isinstance(row, list) else row for row in self._data]
            self._data = [list(map(float, values))]
        elif len(self._shape) == 2:
            self._data = [list(map(float, row)) for row in self._data]
        else:
            raise ValueError("Only 1D and 2D arrays are supported")

    # ------------------------------------------------------------------
    # basic properties
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        out = 1
        for dim in self._shape:
            out *= dim
        return out

    # ------------------------------------------------------------------
    def _ensure_row_major(self) -> List[List[float]]:
        if self.ndim == 1:
            return [self._data[0][:]]
        return [row[:] for row in self._data]

    def copy(self) -> "ndarray":
        return ndarray(self._ensure_row_major(), self._shape)

    def tolist(self) -> List:
        if self.ndim == 1:
            return [float(x) for x in self._data[0]]
        return [row[:] for row in self._data]

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator:
        if self.ndim == 1:
            return iter(self._data[0])
        return iter(row[:] for row in self._data)

    def __len__(self) -> int:
        return self._shape[0]

    # ------------------------------------------------------------------
    def _index_rows(self, sel) -> List[List[float]]:
        rows = self._ensure_row_major()
        if isinstance(sel, slice):
            return rows[sel]
        if isinstance(sel, list):
            if sel and isinstance(sel[0], bool):
                return [r for flag, r in zip(sel, rows) if flag]
            return [rows[int(i)] for i in sel]
        if isinstance(sel, ndarray):
            seq = sel.tolist()
            if seq and isinstance(seq[0], list):
                seq = [item for sub in seq for item in (sub if isinstance(sub, list) else [sub])]
            if seq and all(isinstance(x, (bool, int, float)) for x in seq):
                if any(x not in (0, 1, True, False) for x in seq):
                    return [rows[int(i)] for i in seq]
                bool_seq = [bool(x) for x in seq]
                return [r for flag, r in zip(bool_seq, rows) if flag]
            return [rows[int(i)] for i in seq]
        if isinstance(sel, int):
            return [rows[int(sel)]]
        if sel is Ellipsis:
            return rows
        raise TypeError(f"Unsupported row index type: {type(sel)}")

    def _index_cols(self, rows: List[List[float]], sel) -> List[List[float]]:
        if self.ndim == 1:
            # Selecting from 1D behaves like slicing the underlying list
            if isinstance(sel, slice):
                return [[x for x in self._data[0][sel]]]
            if isinstance(sel, list):
                if sel and isinstance(sel[0], bool):
                    return [[x for flag, x in zip(sel, self._data[0]) if flag]]
                return [[self._data[0][int(i)] for i in sel]]
            if isinstance(sel, ndarray):
                seq = sel.tolist()
                if seq and isinstance(seq[0], list):
                    seq = [item for sub in seq for item in (sub if isinstance(sub, list) else [sub])]
                if seq and all(isinstance(x, (bool, int, float)) for x in seq):
                    if any(x not in (0, 1, True, False) for x in seq):
                        return [[self._data[0][int(i)] for i in seq]]
                    bool_seq = [bool(x) for x in seq]
                    return [[x for flag, x in zip(bool_seq, self._data[0]) if flag]]
                return [[self._data[0][int(i)] for i in seq]]
            if isinstance(sel, int):
                return [[self._data[0][int(sel)]]]
            if sel is Ellipsis:
                return [self._data[0][:]]
            raise TypeError(f"Unsupported column selector {type(sel)} for 1D array")

        if isinstance(sel, slice):
            return [row[sel] for row in rows]
        if isinstance(sel, list):
            if sel and isinstance(sel[0], bool):
                return [[x for flag, x in zip(sel, row) if flag] for row in rows]
            return [[row[int(i)] for i in sel] for row in rows]
        if isinstance(sel, ndarray):
            seq = sel.tolist()
            if seq and isinstance(seq[0], bool):
                return [[x for flag, x in zip(seq, row) if flag] for row in rows]
            return [[row[int(i)] for i in seq] for row in rows]
        if isinstance(sel, int):
            return [[row[int(sel)]] for row in rows]
        if sel is Ellipsis:
            return [row[:] for row in rows]
        raise TypeError(f"Unsupported column index type: {type(sel)}")

    def __getitem__(self, key):
        if self.ndim == 1:
            if isinstance(key, (slice, list, ndarray)):
                rows = self._index_cols(self._ensure_row_major(), key)
                return ndarray(rows, (len(rows[0]),))
            if isinstance(key, int):
                return float(self._data[0][key])
        if isinstance(key, tuple):
            row_sel, col_sel = key
            rows = self._index_rows(row_sel)
            cols = self._index_cols(rows, col_sel)
            if cols and len(cols) == 1 and len(cols[0]) == 1:
                return cols[0][0]
            new_shape = (len(cols), len(cols[0])) if cols and len(cols[0]) > 1 else (len(cols),)
            if new_shape == (len(cols),):
                flat = [row[0] for row in cols]
                return ndarray([flat], (len(flat),))
            return ndarray(cols, new_shape)
        rows = self._index_rows(key)
        if self.ndim == 2:
            shape = (len(rows), self._shape[1])
            return ndarray(rows, shape)
        return ndarray(rows, (len(rows[0]),))

    # ------------------------------------------------------------------
    def _apply_elementwise(self, other, op):
        if isinstance(other, ndarray):
            other_data = other.tolist()
        else:
            other_data = other

        if self.ndim == 1:
            left = self._data[0]
            if isinstance(other_data, list):
                result = [op(a, b) for a, b in zip(left, other_data)]
            else:
                result = [op(a, other_data) for a in left]
            return ndarray([result], (len(result),))

        result_rows: List[List[float]] = []
        if isinstance(other_data, list) and other_data and isinstance(other_data[0], list):
            for row, other_row in zip(self._data, other_data):
                result_rows.append([op(a, b) for a, b in zip(row, other_row)])
        elif isinstance(other_data, list):
            for row in self._data:
                result_rows.append([op(a, b) for a, b in zip(row, other_data)])
        else:
            for row in self._data:
                result_rows.append([op(a, other_data) for a in row])
        return ndarray(result_rows, self._shape)

    def _comparison(self, other, op):
        result = self._apply_elementwise(other, lambda a, b: bool(op(a, b)))
        return result

    def __eq__(self, other):
        return self._comparison(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._comparison(other, lambda a, b: a != b)

    def __lt__(self, other):
        return self._comparison(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._comparison(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._comparison(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._comparison(other, lambda a, b: a >= b)

    def __add__(self, other):
        return self._apply_elementwise(other, lambda a, b: float(a) + float(b))

    def __sub__(self, other):
        return self._apply_elementwise(other, lambda a, b: float(a) - float(b))

    def __mul__(self, other):
        return self._apply_elementwise(other, lambda a, b: float(a) * float(b))

    def __truediv__(self, other):
        return self._apply_elementwise(other, lambda a, b: float(a) / float(b))

    def __mod__(self, other):
        return self._apply_elementwise(other, lambda a, b: float(a) % float(b))

    def __and__(self, other):
        return self._apply_elementwise(other, lambda a, b: bool(a) and bool(b))

    def __or__(self, other):
        return self._apply_elementwise(other, lambda a, b: bool(a) or bool(b))

    def __invert__(self):
        return self._apply_elementwise(1, lambda a, _: not bool(a))

    def all(self, axis=None):
        if axis is None:
            rows = self._ensure_row_major() if self.ndim == 2 else [self._data[0][:]]
            return _py_all(bool(v) for row in rows for v in row)
        if axis == 1 and self.ndim == 2:
            rows = self._ensure_row_major()
            return ndarray([[ _py_all(bool(v) for v in row) ] for row in rows], (len(rows),))
        if axis == 0 and self.ndim == 2:
            cols = list(zip(*self._ensure_row_major()))
            return ndarray([[ _py_all(bool(v) for v in col) for col in cols ]], (len(cols),))
        raise ValueError("Unsupported axis")

    def any(self):
        rows = self._ensure_row_major() if self.ndim == 2 else [self._data[0][:]]
        return any(bool(v) for row in rows for v in row)

    def astype(self, dtype, copy=True):
        converter = float if dtype in (float, "float", "float64", float64) else dtype
        rows = [[converter(v) for v in row] for row in self._ensure_row_major()]
        shape = self._shape
        return ndarray(rows, shape)

    def min(self):
        if self.ndim == 1:
            return min(float(v) for v in self._data[0])
        return min(float(v) for row in self._data for v in row)

    def max(self):
        if self.ndim == 1:
            return max(float(v) for v in self._data[0])
        return max(float(v) for row in self._data for v in row)


float64 = float
int64 = int
nan = float("nan")
bool_ = bool


def _normalise_input(data) -> Tuple[List[List[float]], Tuple[int, ...]]:
    if isinstance(data, ndarray):
        return data._ensure_row_major(), data.shape
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, list, tuple)):
        data = list(data)
    if isinstance(data, Sequence) and data and isinstance(data[0], Sequence):
        rows = [list(map(float, row)) for row in data]
        shape = (len(rows), len(rows[0]) if rows else 0)
        return rows, shape
    seq = list(data) if isinstance(data, Sequence) else [data]
    return [list(map(float, seq))], (len(seq),)


def array(data, dtype=None):
    rows, shape = _normalise_input(data)
    arr = ndarray(rows, shape)
    if dtype is not None:
        return arr.astype(dtype)
    return arr


def empty(shape, dtype=float64):
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 1:
        data = [[0.0] * shape[0]]
    elif len(shape) == 2:
        data = [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError("Only 1D/2D arrays are supported")
    return ndarray(data, tuple(shape))


def full(shape, fill_value, dtype=float64):
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 1:
        data = [[fill_value for _ in range(shape[0])]]
    elif len(shape) == 2:
        data = [[fill_value for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError("Only 1D/2D arrays are supported")
    return ndarray(data, tuple(shape)).astype(dtype)


def arange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    values = []
    cur = float(start)
    while (step > 0 and cur < stop) or (step < 0 and cur > stop):
        values.append(cur)
        cur += step
    return array(values)


def concatenate(arrays: Sequence[ndarray], axis=0):
    if axis != 0:
        raise ValueError("Only axis=0 is supported")
    rows: List[List[float]] = []
    for arr in arrays:
        rows.extend(arr._ensure_row_major())
    if not rows:
        return empty((0, arrays[0].shape[1] if arrays else 0))
    shape = (len(rows), len(rows[0]))
    return ndarray(rows, shape)


def argsort(arr: ndarray, kind=None):
    values = arr.tolist()
    if arr.ndim == 2:
        values = [row[0] for row in values]
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    return array([idx for idx, _ in indexed])


def unique(arr: ndarray, return_index=False):
    values = arr.tolist()
    if isinstance(values[0], list):
        values = [row[0] for row in values]
    seen = {}
    out_vals = []
    out_idx = []
    for idx, val in enumerate(values):
        if val not in seen:
            seen[val] = idx
            out_vals.append(val)
            out_idx.append(idx)
    if return_index:
        return array(out_vals), array(out_idx)
    return array(out_vals)


def isfinite(arr: ndarray) -> ndarray:
    rows = [[math.isfinite(float(v)) for v in row] for row in arr._ensure_row_major()]
    return ndarray(rows, arr.shape)


def isclose(a, b, atol=1e-8, rtol=1e-5):
    a_vals = array(a).tolist()
    b_vals = array(b).tolist()
    return all(abs(float(x) - float(y)) <= atol + rtol * abs(float(y)) for x, y in zip(a_vals, b_vals))


def array_equal(a, b):
    return array(a).tolist() == array(b).tolist()


def diff(arr: ndarray):
    values = arr.tolist()
    if isinstance(values[0], list):
        values = [row[0] for row in values]
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return array(diffs)


def median(arr: ndarray):
    values = sorted(arr.tolist())
    n = len(values)
    if n == 0:
        raise ValueError("median of empty array")
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)


def mean(arr: ndarray):
    if isinstance(arr, ndarray):
        values = arr.tolist()
        if values and isinstance(values[0], list):
            values = [row[0] for row in values]
    else:
        values = list(arr)
    return float(sum(values) / len(values))


def abs(arr):  # noqa: A003 - mimic numpy namespace
    arr_obj = array(arr)
    values = arr_obj.tolist()
    result = [float(_py_abs(v)) for v in values]
    if not isinstance(arr, ndarray) and arr_obj.ndim == 1 and len(result) == 1:
        return result[0]
    return array(result)


def all(arr):  # noqa: A003 - mimic numpy namespace
    return array(arr).all()


def load(file, allow_pickle=False):
    if isinstance(file, (str, bytes)):
        with open(file, "rb") as f:
            data = f.read()
    else:
        data = file.read()
    if not data:
        return empty((0,))
    if isinstance(data, bytes):
        payload = json.loads(data.decode("utf-8"))
    else:
        payload = json.loads(data)
    return array(payload["data"])  # type: ignore[index]


def save(file, arr: ndarray, allow_pickle=False):
    payload = json.dumps({"data": arr.tolist()}).encode("utf-8")
    if isinstance(file, (str, bytes)):
        with open(file, "wb") as f:
            f.write(payload)
    else:
        file.write(payload)


def isna(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def zeros(shape, dtype=float64):
    return full(shape, 0.0, dtype=dtype)


def ones(shape, dtype=float64):
    return full(shape, 1.0, dtype=dtype)


def column_stack(columns: Sequence[ndarray]):
    rows = list(zip(*[col.tolist() for col in columns]))
    return array(rows)


def format_float_positional(value: float, trim: str = "k") -> str:
    if trim == "0":
        return ("%f" % float(value)).rstrip("0").rstrip(".")
    return ("%g" % float(value))


def isscalar(obj) -> bool:
    return not isinstance(obj, (list, tuple, dict, set, ndarray))

