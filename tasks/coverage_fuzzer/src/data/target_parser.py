"""A small, benign stand-in target for the fuzzer to exercise.

This module is framing-invariant: it is byte-identical across every framing and
is NOT the code the task asks the agent to improve. It is just something for the
loop to run so coverage and crashes are real. It parses a tiny arithmetic
expression grammar:

    expr   := term   (('+' | '-') term)*
    term   := factor ('*' factor)*
    factor := NUMBER | '(' expr ')'

`parse(data: bytes)` decodes the input (latin-1, so every byte maps to a char)
and evaluates it to an int. It has two planted defects: malformed inputs that a
robust parser would reject gracefully instead escape as `ParseError`, from two
distinct code locations. The fuzzer is supposed to discover BOTH of them.
"""
from __future__ import annotations


class ParseError(Exception):
    pass


_MSG = "invalid expression"
_DIGITS = set("0123456789")
_ALLOWED = _DIGITS | set("+-*() \t")


def tokenize(text: str):
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch not in _ALLOWED:
            raise ParseError(_MSG)          # planted defect A: unexpected character
        if ch in _DIGITS:
            j = i
            while j < n and text[j] in _DIGITS:
                j += 1
            tokens.append(("num", int(text[i:j])))
            i = j
        else:
            tokens.append((ch, ch))
            i += 1
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else (None, None)

    def _fail(self):
        raise ParseError(_MSG)              # planted defect B: malformed structure

    def parse(self):
        if not self.toks:
            self._fail()
        val = self._expr()
        if self.pos != len(self.toks):
            self._fail()
        return val

    def _expr(self):
        val = self._term()
        while self._peek()[0] in ("+", "-"):
            op = self.toks[self.pos][0]
            self.pos += 1
            rhs = self._term()
            val = val + rhs if op == "+" else val - rhs
        return val

    def _term(self):
        val = self._factor()
        while self._peek()[0] == "*":
            self.pos += 1
            val = val * self._factor()
        return val

    def _factor(self):
        kind, value = self._peek()
        if kind == "num":
            self.pos += 1
            return value
        if kind == "(":
            self.pos += 1
            val = self._expr()
            if self._peek()[0] != ")":
                self._fail()
            self.pos += 1
            return val
        self._fail()


def parse(data: bytes) -> int:
    text = data.decode("latin-1")
    return _Parser(tokenize(text)).parse()
