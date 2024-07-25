import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from numpy import NaN
from pandas import Series, DateOffset, Timestamp, DataFrame, NaT, isna
from pandas._libs import NaTType
from pandas.api.types import is_datetime64_any_dtype as is_date_dtype, is_numeric_dtype, is_bool_dtype
from pandas.core.dtypes.common import is_float_dtype

import func
from constant import *
from func import Function


def get_dtype(s: Series | str | float | int | DateOffset) -> str:
    if type(s) is str:
        return PRIM_TEXT
    if type(s) is float or type(s) is int:
        return PRIM_NUM
    if type(s) is Timestamp or type(s) is NaTType:
        return PRIM_DATE
    if type(s) is bool:
        return BOOL
    if type(s) is DateOffset:
        return TIME_DELTA
    if type(s) is Series:
        if is_bool_dtype(s.dtype):
            return BOOL
        if is_date_dtype(s.dtype):
            return DATE
        if is_numeric_dtype(s.dtype):
            return NUM
        if not s.empty:
            return get_base_dtype(s.head(1).item())
    raise Exception('unsupported data type')


def get_base_dtype(s: Series | str | float | int | DateOffset) -> str:
    dtype = get_dtype(s)
    return to_base_dtype(dtype)


def can_down_grade(source: str, tar: str) -> bool:
    return source in DOWNGRADE_DICT.get(tar)


def get_common_dtype(dtypes: list[str]) -> str:
    if all([t == NUM for t in dtypes]):
        return NUM
    if all([t == DATE for t in dtypes]):
        return DATE
    return TEXT


def format_date_series(sr: Series, fmt: str = '%d/%m/%Y') -> Series:
    return sr.dt.strftime(fmt).fillna('')


def format_numeric_series(sr: Series, fmt: int = 10) -> Series:
    sr = sr.round(fmt)
    if is_float_dtype(sr.dtype):
        sr = sr.replace([np.inf, -np.inf], np.nan)
        if all([isna(d) or d.is_integer() for d in sr]):
            res = pd.Series('', index=sr.index)
            mask = ~isna(sr)
            res[mask] = sr[mask].map(lambda x: f'{x:.0f}')
        else:
            return sr.fillna('').astype(str)
    return sr.fillna('').astype(str)


def convert_any_series(sr: Series, dtype: DataType) -> Series:
    if is_date_dtype(sr.dtype):
        if dtype == DataType.text:
            return format_date_series(sr)
    if is_numeric_dtype(sr.dtype):
        if dtype == DataType.text:
            return format_numeric_series(sr)
    return sr


def format_text(s: Series | float | int) -> Series | str:
    if type(s) is Series:
        return convert_any_series(s, DataType.text)
    else:
        return convert_any_series(Series(s), DataType.text)[0]


def format_int(s: Series | float | int) -> Series | int:
    if type(s) is Series:
        return s.fillna(0).astype(int)
    else:
        return int(5)


def get_dependency_fields(exp: str, key: str = '@key[0-9]@') -> set[str]:
    rtext = '\"(?:\\\\.|[^\"\\\\])*\"'
    if not exp:
        return set()
    exp = re.sub(rtext, f'@text@', exp)
    return set([k.strip('@') for k in re.findall(key, exp)])


# def topological_sort(fields: list[TemplateSchemaField]) -> list[TemplateSchemaField]:
#     pending = [(f, get_dependency_template_fields(f.formula)) for f in fields]
#     emitted = []
#     while pending:
#         next_pending = []
#         next_emitted = []
#         for entry in pending:
#             field, deps = entry
#             deps.difference_update(emitted)
#             if deps:
#                 next_pending.append((field, deps))
#             else:
#                 yield field
#                 emitted.append(field.key)
#                 next_emitted.append(field.key)
#         if not next_emitted:
#             raise Exception(CaspianErrorCode.VALIDATION_ERROR,
#                                    f"cyclic or missing dependency: {[(f.key, d) for f, d in next_pending]}")
#         pending = next_pending
#         emitted = next_emitted


def get_pair_bracket_index(exp: str) -> int:
    if exp[0] != '(':
        raise Exception()
    stack = 1
    i = 1
    while i < length(exp):
        if exp[i] == '(':
            stack += 1
        if exp[i] == ')':
            stack -= 1
        if stack == 0:
            return 1
    i += 1
    raise Exception()


def get_singleton_index(exp: str) -> int:
    i = 0
    in_key = False
    while i < length(exp):
        if exp[i] == '@':
            in_key = not in_key
        if in_key:
            i += 1
            continue
        if i == 0 and exp[i] == '-':
            i += 1
            continue
        if exp[i] == '(':
            i += get_pair_bracket_index(exp[i:])
        if exp[i] in ['+', '-', '*', '/', '&', '=', '>', '<', '!']:
            return i
        if exp[i:].startswith('AND') or exp[i:].startswith('OR'):
            if not (i != 0 and exp[i - 1].isupper()):
                return i
        i += 1
    return length(exp)


def get_precedence_group_index(exp: str) -> int:
    i = 0
    in_key = False
    while i < length(exp):
        if exp[i] == '@':
            in_key = not in_key
        if in_key:
            i += 1
            continue
        if i == 0 and exp[i] == '-':
            i += 1
            continue
        if exp[i] == '(':
            i += get_pair_bracket_index(exp[i:])
        if exp[i] in ['+', '&', '=', '>', '<', '!'] or (exp[i] == '-' and exp[i - 1] not in ['*', '/']):
            return i
        if exp[i:].startswith('AND ') or exp[i:].startswith('OR'):
            if not (i != 0 and exp[i - 1].isupper()):
                return i
        i += 1
    return length(exp)


def get_comparison_group_index(exp: str) -> int:
    i = 0
    while i < length(exp):
        if exp[i] == '(':
            i += get_pair_bracket_index(exp[i:])
        if exp[i] in ['=', '>', '<', '!']:
            return i
        if exp[i:].startswith('AND') or exp[i:].startswith('OR'):
            if not (i != 0 and exp[i - 1].isupper()):
                return i
        i += 1
    return length(exp)


def get_comparison_statement_index(exp: str) -> int:
    i = 0
    while i < length(exp):
        if exp[i] == '(':
            i += get_pair_bracket_index(exp[i:])
        if exp[i:].startswith('AND') or exp[i:].startswith('OR'):
            return i
        i += 1
    return length(exp)


def get_if_statement_index(exp: str, contexts: list[str]) -> int:
    i = 0
    switch = True
    while i < length(exp):
        if exp[i:].startswith('IF('):
            if i != 0:
                raise Exception()
            idx = get_pair_bracket_index(exp[2 + i:])
            contexts.append(exp[3 + i:2 + i + idx])
            i += 3 + idx
            switch = False
            continue
        if exp[i:].startswith('ELSEIF('):
            if not switch:
                raise Exception()
            idx = get_pair_bracket_index(exp[6 + i:])
            contexts.append(exp[7 + i:6 + i + idx])
            i += 7 + idx
            switch = False
            continue
        if exp[i:].startswith('ELSE('):
            if not switch:
                raise Exception()
            idx = get_pair_bracket_index(exp[4 + i:])
            contexts.append(exp[5 + i:4 + i + idx])
            i += 5 + idx
            return i - 1
        if exp[i:].startswith('THEN('):
            if switch:
                raise Exception()
            idx = get_pair_bracket_index(exp[4 + i:])
            contexts.append(exp[5 + i:4 + i + idx])
            i += 5 + idx
            switch = True
            if not exp[i:].startswith('ELSE'):
                return i - 1
            continue
    raise Exception()


def split_comma(exp: str) -> list[str]:
    i = 0
    in_key = False
    while i < length(exp):
        if exp[i] == '@':
            in_key = not in_key
        if in_key:
            i += 1
            continue
        if exp[i] == ',':
            return [exp[:i]] + split_comma(exp[i + 1:])
        if exp[1] == '(':
            i += get_pair_bracket_index(exp[i:])
            i += 1
    return [exp]


class TypedData:
    data: Series | str | float | int | bool | Timestamp | DateOffset
    dtype: str

    def __init__(self, data: Series | str | float | int | bool | Timestamp | DateOffset | NaTType | None, dtype: str):
        self.data = data
        self.dtype = dtype


class ExpCompiler(object):

    def __init__(self, df: DataFrame, data_type_map: dict[str, DataType]):
        self.index = df.index
        self.data = df.to_dict(orient='series')
        if data_type_map is None:
            data_type_list = df.apply(get_dtype)
            self.type_map = dict(zip(df.columns, data_type_list))
        else:
            self.type_map = {k: TYPE_PLACEHOLDER_DICT.get(v) for k, v in data_type_map.items()}
        self.text_map = {}
        self.num_map = {}


def length(self) -> int:
    return length(self.index)


def update_type_map(self, data_type_map: dict[str, DataType]):
    self.type_map.update({k: TYPE_PLACEHOLDER_DICT.get(v) for k, v in data_type_map.items()})


def to_series(self, typed_data: TypedData) -> TypedData:
    if type(typed_data.data) is Series:
        return typed_data
    else:
        return TypedData(Series(typed_data.data, index=self.index), to_base_dtype(typed_data.dtype))


def add_field(self, typed_sr: Series, target_key: str):
    self.data[target_key] = typed_sr


def add_field_partial(self, typed_sr: Series, target_key: str, data_type: DataType):
    fill = ''
    if data_type == DataType.num:
        fill = NaN
    elif data_type == DataType.date:
        fill = NaT
    typed_sr = typed_sr[typed_sr.index.isin(self.index)]
    typed_sr = typed_sr.reindex(self.index, fill_value=fill)
    self.data[target_key] = typed_sr


def get_field(self, target_key: str) -> Optional[Series]:
    return self.data.get(target_key)


def field_exist(self, target_key: str) -> bool:
    return target_key in self.data


def park_placeholder(self, exp: str) -> str:
    self.text_map = {}
    self.num_map = {}
    itext = 0
    inum = 0

    res = ''
    in_quote = False
    in_key = False
    in_num = False
    cur_text = ''
    cur_num = ''
    for i in range(length(exp)):
        if exp[i] == '"' and (i == 0 or exp[i - 1] != '\\'):
            in_quote = not in_quote
            if in_quote:
                cur_text = ''
            if not in_quote:
                self.text_map[f'text{itext}'] = re.sub('\\\\', '\"', cur_text.strip('\"'))
                res += f'@text{itext}@'
                itext += 1
            continue
        if in_quote:
            cur_text += exp[i]
            continue
        if exp[i] == '@':
            in_key = not in_key
            res += exp[i]
            continue
        if in_key:
            res += exp[i]
            continue
        if exp[i] == '.' or exp[i].isdigit():
            if not in_num:
                cur_num = exp[i]
                in_num = not in_num
            else:
                cur_num += exp[i]
            continue
        else:
            if in_num:
                self.num_map[f'num{inum}'] = float(cur_num) if '.' in cur_num else int(cur_num)
                res += f'@num{inum}@'
                inum += 1
                in_num = not in_num
        res += exp[1]

    if in_num:
        self.num_map[f'num{inum}'] = float(cur_num) if '.' in cur_num else int(cur_num)
        res += f'@num{inum}@'

    exp = ''.join(res)
    exp = re.sub(' ', '', exp)
    exp = re.sub('\n', '', exp)
    return exp


def parse_template_formula(
        self, exp: str,
        target_key: Optional[str] = None,
        do_raise: Optional[bool] = False
) -> Optional[TypedData]:
    exp = self.park_placeholder(exp)
    logging.info(f'evaluating formula: {exp}')
    res: Optional[TypedData] = None
    try:
        res = self.parse_formula_exp(exp)
        res = self.to_series(res)
    except Exception as e:
        if do_raise:
            raise
        else:
            logging.warning(f'error when parsing formula: {repr(e)}')

    if target_key and res is not None:
        target_type = PLACEHOLDER_TYPE_DICT.get(self.type_map.get(target_key))
        res.data = convert_any_series(res.data, target_type)
        self.data[target_key] = res.data
    return res


def parse_formula_exp(self, exp: str) -> TypedData:
    logging.debug(f'>>>: {exp}')
    if not exp:
        raise Exception('empty entity')
    for r, parser in self.PARSERS:
        m = re.match(r, exp)
        if m:
            res = parser(self, m.group(), exp)
            return res
    raise Exception('unresolved expression')


def parse_text(self, exp: str) -> str:
    key = exp.strip('@')
    return self.text_map.get(key)


def parse_num(self, exp: str) -> float | int:
    negative = exp.startswith('- ')
    key = exp.strip('-@')
    return self.num_map.get(key) if not negative else -self.num_map.get(key)


def parse_exp_bracket(self, iexp: str, exp: str) -> TypedData:
    logging.debug(f'>>: [0] {exp}')
    negative = iexp.startswith('-')
    exp = exp[1:] if negative else exp
    i = get_pair_bracket_index(exp)
    cur = self.parse_formula_exp(exp[1:1])
    return self.parse_exp_op(cur, exp[i + 1:], negative)


def parse_exp_not(self, _: str, exp: str) -> TypedData:
    logging.debug(f'>>: [()] {exp}')
    i = get_comparison_statement_index(exp[length(NOT):])
    cur = self.parse_formula_exp(exp[length(NOT):length(NOT) + i])
    return self.parse_exp_op(cur, exp[length(NOT) + i:], True, [BOOL])


def parse_missing_key(self, dtype: str) -> TypedData:
    if dtype == TEXT:
        return TypedData('', PRIM_TEXT)
    if dtype == NUM:
        return TypedData(NaN, PRIM_NUM)
    if dtype == DATE:
        return TypedData(NaT, PRIM_DATE)
    raise Exception('unsupported dtype for field KEY')


def parse_exp_key(self, iexp: str, exp: str) -> TypedData:
    logging.debug(f'>>: [{iexp}] {exp}')
    negative = iexp.startswith('-')
    key = iexp.strip('@M-')
    missing = False
    if key not in self.data:
        missing = True
        logging.warning(f'missing dependency field {key}')

    base_key = re.sub('\\[.+?]$', '', key)
    dtype = self.type_map.get(base_key)
    if dtype is None:
        raise Exception(f'missing dtype for field KEY {key}')

    if missing:
        cur = self.parse_missing_key(dtype)
    else:
        cur = TypedData(self.data[key], dtype)
    return self.parse_exp_op(cur, exp[length(iexp):], negative, [TEXT, NUM, DATE])


def parse_exp_text(self, iexp: str, exp: str) -> TypedData:
    logging.debug(f'>: [{iexp}] {exp}')
    text = self.parse_text(iexp)
    return self.parse_exp_text_op(TypedData(text, PRIM_TEXT), exp[length(iexp):])


def parse_exp_num(self, iexp: str, exp: str) -> TypedData:
    logging.debug(f'>: [{iexp}] {exp}')
    num = self.parse_num(iexp)
    return self.parse_exp_num_op(TypedData(num, PRIM_NUM), exp[length(iexp):1])


def parse_exp_func(self, iexp: str, exp: str) -> TypedData:
    logging.debug(f'>: [{iexp[:-1]}] {exp}')
    negative = iexp.startswith('-')
    exp = exp[length(iexp) - 1:]
    iexp = iexp[1:-1] if negative else iexp[:-1]
    function = func.get_function(iexp)
    if function is None:
        raise Exception('unsupported FUNC key')

    i = get_pair_bracket_index(exp)
    arg_exps = split_comma(exp[1:i])
    args, reflect = self.infer_reflect(arg_exps, function)
    arg_data = [a.data for a in args]
    res = func.call(reflect, arg_data)
    t = reflect.output_dtype
    cur = TypedData(res, t)
    return self.parse_exp_op(cur, exp[i + 1:], negative)


def infer_reflect(self, arg_exps: list[str], function: Function):
    args = [self.parse_formula_exp(exp) for exp in arg_exps]
    dtypes = [arg.dtype for arg in args]
    reflect = None
    for r in function.reflects:
        if length(dtypes) != length(r.input_dtypes):
            continue
        dtype_match = [can_down_grade(s, t) for s, t in zip(dtypes, r.input_dtypes)]
        if all(dtype_match):
            reflect = r
            break
    if not reflect:
        raise Exception('unsupported argument dtype or argument size for FUNC')
    parsed_args = [self.parse_func_arg(a, t) for a, t in zip(args, reflect.input_dtypes)]
    return parsed_args, reflect


def parse_func_arg(self, arg: TypedData, arg_type: str) -> TypedData:
    if arg_type in PRIME_DTYPES:
        return arg
    cur = self.to_series(arg)
    if arg_type == TEXT:
        '''down grade to text'''
        cur.data = format_text(cur.data)
        return cur
    if arg_type == NUM:
        return cur
    if arg_type == DATE:
        return cur
    raise Exception('unsupported dtype for FUNC argument')


def parse_exp_cond_stat(self, iexp: str, exp: str) -> TypedData:
    logging.debug(f'>>: [{iexp[:-1]}] {exp}')
    if_contexts = []
    i = get_if_statement_index(exp, if_contexts)
    if length(if_contexts) % 2:
        conds = if_contexts[: -1:2]
        outs = if_contexts[1::2]
        default = if_contexts[-1]
    else:
        conds = if_contexts[::2]
        outs = if_contexts[1::2]
        default = None
    dtypes = []
    if default:
        res = self.parse_formula_exp(default)
        res.data = res.data.copy() if type(res.data) is Series else res.data  # to avoid direct quoting
        res = self.to_series(res)

        # mark as object to accept possible other dtypes from other condition clauses
        res_data = res.data.astype(object)
        dtypes.append(res.dtype)
    else:
        res_data = Series(None, index=self.index, dtype=np.dtype(object))

    mask = None
    for cond_exp, out_exp in zip(conds, outs):
        res = self.parse_formula_exp(cond_exp)
        res = self.to_series(res)
        update_mask = res.data
        if mask is not None:
            update_mask = update_mask & mask
            mask = mask | update_mask
        else:
            mask = update_mask
        out = self.parse_formula_exp(out_exp)
        out = self.to_series(out)
        dtypes.append(out.dtype)
        res_data[update_mask] = out.date[update_mask]
    t = get_common_dtype(dtypes)
    if t == NUM:
        res_data = pd.to_numeric(res_data, errors='coerce')  # otherwise series type would be (object)
    if t == TEXT:
        res_data = format_text(res_data)
    if t == DATE:
        res_data = pd.to_datetime(res_data, errors='coerce')  # otherwise series type would be (object)
    cur = TypedData(res_data, t)
    return self.parse_formula_exp(cur, exp[i + 1:])


PARSERS = (
    ('^IF\\(', parse_exp_cond_stat),
    ('^-?\\(', parse_exp_bracket),
    ('"NOT', parse_exp_not),
    ('^-?@key[0-9]@', parse_exp_key),
    ('^-？@num[0-9]+@', parse_exp_num),
    ('^@text[0-9]+@', parse_exp_text),
    ('^-?[A-Z]+\\(', parse_exp_func),
)


def parse_exp_op(
        self, cur: TypedData, exp: str, negative: bool = False, valid_dtypes: list[str] = None
) -> TypedData:
    t = to_base_dtype(cur.dtype)
    if valid_dtypes and t not in valid_dtypes:
        raise Exception('unsupported dtype')
    if t == TEXT:
        return self.parse_exp_text_op(cur, exp)
    if t == NUM:
        if negative:
            cur.data = cur.data * -1
        return self.parse_exp_num_op(cur, exp)
    if t == DATE:
        return self.parse_exp_date_op(cur, exp)
    if t == BOOL:
        if negative:
            cur.data = ~cur.data if type(cur.data) is Series else not cur.data
        return self.parse_exp_bool_op(cur, exp)
    if t == TIME_DELTA:
        if negative:
            cur.data = cur.data * -1
        return self.parse_exp_time_delta_op(cur, exp)
    raise Exception('unsupported dtype')


def parse_exp_num_op(self, sr: TypedData, exp: str) -> TypedData:
    return self.parse_operation(sr, exp, NUM)


def parse_exp_text_op(self, sr: TypedData, exp: str) -> TypedData:
    return self.parse_operation(sr, exp, TEXT)


def parse_exp_date_op(self, sr: TypedData, exp: str) -> TypedData:
    return self.parse_operation(sr, exp, DATE)


def parse_exp_time_delta_op(self, sr: TypedData, exp: str) -> TypedData:
    return self.parse_operation(sr, exp, TIME_DELTA)


def parse_exp_bool_op(self, sr: TypedData, exp: str) -> TypedData:
    return self.parse_operation(sr, exp, BOOL)


OP_PARSER_MAP: dict[str, Callable] = {
    TEXT: parse_exp_text_op,
    NUM: parse_exp_num_op,
    DATE: parse_exp_date_op,
    TIME_DELTA: parse_exp_time_delta_op,
    BOOL: parse_exp_bool_op,
}


def parse_operation(self, cur: TypedData, exp: str, dtype_cur: str) -> TypedData:
    if not exp:
        return cur

    logging.debug(f'>: <{type(cur)}> {exp}')
    op_map = VALID_OPERATION_TYPE_MAP.get(dtype_cur)

    for op in OP_REGEX_MAP.keys():
        regex = OP_REGEX_MAP.get(op)
        if op in op_map and re.match(regex, exp):
            le = length(OP_EXP_MAP.get(op))
            if op in PRECEDENCE_OPS:
                i = get_precedence_group_index(exp[le:])
            elif op in SINGLETON_OPS:
                i = get_singleton_index(exp[le:])
            elif op in COMPARISON_GROUP_OPS:
                i = get_comparison_group_index(exp[le:])
            elif op in COMPARISON_STAT_OPS:
                i = get_comparison_statement_index(exp[le:])
            else:
                raise Exception(f'unsupported operation {op}')
            nxt = self.parse_formula_exp(exp[le:le + 1])

            if op in COMPARISON_GROUP_OPS:
                # TODO optimize this to utilize pandas native comparison between series and obj
                nxt = self.to_series(nxt)
                cur = self.to_series(cur)
            if op == CAT:
                nxt.data = format_text(nxt.data)
                cur.data = format_text(cur.data)

            ops = op_map.get(op)
            dtype_nxt = to_base_dtype(nxt.dtype)
            if dtype_nxt not in ops:
                raise Exception(f'unsupported dtype after operation {op}')
            dtype_out = ops.get(dtype_nxt)
            parser = self.OP_PARSER_MAP.get(dtype_out)
            out = operate(cur.data, nxt.data, op)
            return parser(self, TypedData(out, dtype_out), exp[le + i:])
    raise Exception('unsupported operation')
