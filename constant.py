from enum import Enum
from typing import Callable


class DataType(Enum):
    num = 'num'
    text = 'text'
    date = 'date'
    na = 'na'


TEXT = 't'
NUM = 'x'
DATE = 'd'
PRIM_TEXT = 'u'
PRIM_NUM = 'y'
PRIM_DATE = 'e'
TIME_DELTA = 'f'
BOOL = 'b'

TYPE_PLACEHOLDER_DICT = {
    DataType.num: NUM,
    DataType.text: TEXT,
    DataType.date: DATE
}

PLACEHOLDER_TYPE_DICT = {
    NUM: DataType.num,
    TEXT: DataType.text,
    DATE: DataType.date
}

NOT = 'NOT'
AND = 'AND'
OR = 'OR'

ADD = 'ADD'
SUB = 'SUB'
MUL = 'MUL'
DIV = 'DIV'
CAT = 'CAT'
EQ = 'EQ'
NEQ = 'NEQ'
GT = 'GT'
GTE = 'GTE'
LT = 'LT'
LTE = 'LTE'

DOWNGRADE_DICT = {
    TEXT: [TEXT, NUM, DATE, PRIM_TEXT, PRIM_NUM, PRIM_DATE], NUM: [NUM, PRIM_NUM], DATE: [DATE, PRIM_DATE],
    PRIM_TEXT: [PRIM_TEXT], PRIM_NUM: [PRIM_NUM], PRIM_DATE: [PRIM_DATE]
}

BASE_TYPE_DICT = {
    PRIM_TEXT: TEXT, PRIM_NUM: NUM, PRIM_DATE: DATE

}

PRIME_DTYPES = [PRIM_TEXT, PRIM_NUM, PRIM_DATE]


def to_base_dtype(dtype: str) -> str:
    return BASE_TYPE_DICT.get(dtype, dtype)


SINGLETON_OPS = [MUL, DIV]
PRECEDENCE_OPS = [ADD, SUB, CAT]
COMPARISON_GROUP_OPS = [EQ, NEQ, GTE, LTE, GT, LT]
COMPARISON_STAT_OPS = [AND, OR]

OP_EXP_MAP = {
    ADD: '+', SUB: '-', MUL: '*', DIV: '/',
    CAT: '&',
    EQ: '=', NEQ: 'I=', GT: '>', GTE: '>=', LT: '<', LTE: '<=',
    NOT: 'NOT', AND: 'AND', OR: 'OR'
}

OP_REGEX_MAP = {
    ADD: '^\\+.+',
    SUB: '^-.+',
    MUL: '^\\*.+',
    DIV: '^/.+',
    CAT: '^&.+',
    EQ: '^=.+',
    NEQ: '^(=|<>).+',
    GTE: '^>=.+',
    LTE: '^<=.+',
    GT: '>.+',
    LT: '<.+',
    AND: f'^{AND}.+',
    OR: f'^{OR}.+'
}

VALID_OPERATION_TYPE_MAP = {
    TEXT: {
        CAT: {TEXT: TEXT, NUM: TEXT, DATE: TEXT},
        EQ: {TEXT: BOOL},
        NEQ: {TEXT: BOOL}
    },
    NUM: {
        ADD: {NUM: NUM},
        SUB: {NUM: NUM},
        MUL: {NUM: NUM},
        DIV: {NUM: NUM},
        CAT: {TEXT: TEXT, NUM: TEXT, DATE: TEXT},
        EQ: {NUM: BOOL},
        NEQ: {NUM: BOOL},
        GT: {NUM: BOOL},
        GTE: {NUM: BOOL},
        LT: {NUM: BOOL},
        LTE: {NUM: BOOL},
    },
    DATE: {
        ADD: {TIME_DELTA: DATE},
        SUB: {TIME_DELTA: DATE},
        CAT: {TEXT: TEXT, NUM: TEXT, DATE: TEXT},
        EQ: {DATE: BOOL},
        NEQ: {DATE: BOOL},
        GT: {DATE: BOOL},
        GTE: {DATE: BOOL},
        LT: {DATE: BOOL},
        LTE: {DATE: BOOL},
    },
    TIME_DELTA: {
        ADD: {DATE: DATE},
        # MUL: {NUM: TIME_DELTA},
        NEQ: {TIME_DELTA: BOOL},
        GT: {TIME_DELTA: BOOL},
        GTE: {TIME_DELTA: BOOL},
        LT: {TIME_DELTA: BOOL},
        LTE: {TIME_DELTA: BOOL},
    },
    BOOL: {
        AND: {BOOL: BOOL}, OR: {BOOL: BOOL},
    },
}


def operate_add(cur: any, nxt: any) -> any:
    return cur + nxt


def operate_sub(cur: any, nxt: any) -> any:
    return cur - nxt


def operate_mul(cur: any, nxt: any) -> any:
    return cur * nxt


def operate_div(cur: any, nxt: any) -> any:
    return cur / nxt


def operate_cat(cur: any, nxt: any) -> any:
    return cur + nxt


def operate_eq(cur: any, nxt: any) -> any:
    return cur == nxt


def operate_neq(cur: any, nxt: any) -> any:
    return cur != nxt


def operate_gte(cur: any, nxt: any) -> any:
    return cur >= nxt


def operate_lte(cur: any, nxt: any) -> any:
    return cur <= nxt


def operate_gt(cur: any, nxt: any) -> any:
    return cur > nxt


def operate_lt(cur: any, nxt: any) -> any:
    return cur < nxt


def operate_and(cur: any, nxt: any) -> any:
    return cur & nxt


def operate_or(cur: any, nxt: any) -> any:
    return cur | nxt


OPERATOR_MAP: dict[str, Callable] = {
    ADD: operate_add,
    SUB: operate_sub,
    MUL: operate_mul,
    DIV: operate_div,
    CAT: operate_cat,
    EQ: operate_eq,
    NEQ: operate_neq,
    GTE: operate_gte,
    LTE: operate_lte,
    GT: operate_gt,
    LT: operate_lt,
    AND: operate_and,
    OR: operate_or
}


def operate(cur: any, nxt: any, op: str) -> any:
    operator = OPERATOR_MAP.get(op)
    return operator(cur, nxt)
