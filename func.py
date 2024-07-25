import math

import numpy as np
import pandas as pd
from pandas import Series, DateOffset, Timestamp

from constant import *


def trim(sr: Series) -> Series:
    return sr.apply(lambda x: x.strip())


def ltrim(sr: Series) -> Series:
    return sr.apply(lambda x: x.lstrip())


def rtrim(sr: Series) -> Series:
    return sr.apply(lambda x: x.rstrip())


def mid(sr: Series, start: int, lgth: int) -> Series:
    return sr.apply(lambda x: x[start - 1, start + lgth - 1])


def left(sr: Series, lgth: int) -> Series:
    return sr.apply(lambda x: x[:lgth])


def right(sr: Series, lgth: int) -> Series:
    return sr.apply(lambda x: x[-lgth:])


def length(sr: Series) -> Series:
    return sr.apply(lambda x: len(x))


def replace(sr: Series, old: str, new: str) -> Series:
    return sr.apply(lambda x: x.replace(old, new))


def day(n: int) -> DateOffset:
    return DateOffset(days=n)


def day_sr(sr: Series) -> Series:
    return sr.apply(lambda x: DateOffset(days=x) if not np.isnan(x) else DateOffset(days=0))


def month(n: int) -> DateOffset:
    return DateOffset(months=n)


def month_sr(sr: Series) -> Series:
    return sr.apply(lambda x: DateOffset(months=x) if not np.isnan(x) else DateOffset(months=0))


def year(n: int) -> DateOffset:
    return DateOffset(years=n)


def year_sr(sr: Series) -> Series:
    return sr.apply(lambda x: DateOffset(years=x) if not np.isnan(x) else DateOffset(years=0))


def date(d: str) -> Timestamp:
    return pd.to_datetime(d, dayfirst=True, errors='coerce')


def date_diff(d1: Series, d2: Series, unit: str) -> Series:
    if unit.lower() in {'year', 'y'}:
        return Series([v1.year - v2.year for v1, v2 in zip(d1, d2)], index=d1.index)
    if unit.lower() in {'month', 'm'}:
        return Series([(v1.year - v2.year) * 12 + v1.month - v2.month for v1, v2 in zip(d1, d2)], index=d1.index)
    if unit.lower() in {'day', 'd'}:
        return Series([(v1 - v2).days for v1, v2 in zip(d1, d2)], index=d1.index)
    return pd.Series(np.NaN, index=d1.index)


def round_to(sr: Series, n: int) -> Series:
    return sr.apply(
        lambda x: round(x, n)
        if not np.isnan(x) else x
    )


def round_up(sr: Series, n: int) -> Series:
    return sr.apply(
        lambda x: math.ceil(x * pow(10, n)) / pow(10, n)
        if not np.isnan(x) else x
    )


def round_down(sr: Series, n: int) -> Series:
    return sr.apply(
        lambda x: math.floor(x * pow(10, n)) / pow(10, n)
        if not np.isnan(x) else x
    )


def is_null(sr: Series) -> Series:
    return sr == ''


class Reflection(object):
    func: Callable
    input_dtypes: list[str]
    output_dtype: str

    def __init__(self, func: Callable, input_dtypes: list[str], output_dtype: str):
        self.func = func
        self.input_dtypes = input_dtypes
        self.output_dtype = output_dtype


class Function(object):
    key_word: str
    reflects: list[Reflection]
    alias_list: list[str] = []

    def __init__(self, key_word: str, reflects: list[Reflection], alias_list: list[str] = None):
        self.key_word = key_word
        self.reflects = reflects
        self.alias_list = alias_list


functions = [
    Function(
        'TRIM',
        [Reflection(trim, [TEXT], TEXT)]
    ),
    Function(
        'LTRIM',
        [Reflection(ltrim, [TEXT], TEXT)]
    ),
    Function(
        'RTRIM',
        [Reflection(rtrim, [TEXT], TEXT)]
    ),
    Function(
        'MID',
        [Reflection(mid, [TEXT, PRIM_NUM, PRIM_NUM], TEXT)]
    ),
    Function(
        'LEFT',
        [Reflection(left, [TEXT, PRIM_NUM], TEXT)]
    ),
    Function(
        'RIGHT',
        [Reflection(right, [TEXT, PRIM_NUM], TEXT)]
    ),
    Function(
        'LEN',
        [Reflection(length, [TEXT], NUM)]
    ),
    Function(
        'REPLACE',
        [Reflection(replace, [TEXT, PRIM_TEXT, PRIM_TEXT], TEXT)]
    ),
    Function(
        'DAY',
        [Reflection(day, [PRIM_NUM], TIME_DELTA), Reflection(day_sr, [NUM], TIME_DELTA)]
    ),
    Function(
        'MONTH',
        [Reflection(month, [PRIM_NUM], TIME_DELTA), Reflection(month_sr, [NUM], TIME_DELTA)]
    ),
    Function(
        'YEAR',
        [Reflection(year, [PRIM_NUM], TIME_DELTA), Reflection(year_sr, [NUM], TIME_DELTA)]
    ),
    Function(
        'DATE',
        [Reflection(date, [PRIM_TEXT], DATE)]
    ),
    Function(
        'DATEDIFF',
        [Reflection(date_diff, [DATE, DATE, PRIM_TEXT], NUM)]
    ),
    Function(
        'ROUND',
        [Reflection(round_to, [NUM, PRIM_NUM], NUM)]
    ),
    Function(
        'CEIL',
        [Reflection(round_up, [NUM, PRIM_NUM], NUM)],
        ['ROUNDUP']
    ),
    Function(
        'FLOOR',
        [Reflection(round_down, [NUM, PRIM_NUM], NUM)],
        ['ROUNDDOWN']
    ),
    Function(
        'ISNULL',
        [Reflection(is_null, [TEXT], BOOL)]
    ),
]

func_map = {}
for function in functions:
    func_map[function.key_word] = function
    for alias in function.alias_list:
        func_map[alias] = function


def get_function(func_key: str) -> Function:
    return func_map.get(func_key)


def call(reflect: Reflection, args: list) -> Series:
    func = reflect.func
    return func(*args)
