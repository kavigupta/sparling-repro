from abc import ABC, abstractmethod

import attr
import numpy as np


@attr.s
class Token:
    name = attr.ib()
    code = attr.ib()
    rendered_symbols = attr.ib()

    @classmethod
    def single_symbol(cls, symbol):
        return cls(symbol, symbol, (symbol,))


@attr.s
class Production:
    elements = attr.ib()
    weight = attr.ib(default=1)

    def render(self):
        result = " ".join(
            x if isinstance(x, str) else repr(x.name) for x in self.elements
        )
        if self.weight != 1:
            result = f"[{self.weight}] " + result
        return result


START, END = Token("<s>", "", ()), Token("</s>", "", ())


@attr.s
class CFG:
    result = attr.ib()

    def sample(self, symbol, rng):
        if isinstance(symbol, Token):
            return [symbol]
        productions = self.result[symbol]
        weights = np.array([p.weight for p in productions])
        weights = weights / weights.sum()
        choice = rng.choice(len(productions), p=weights)
        production = productions[choice]
        result = []
        for element in production.elements:
            result.extend(self.sample(element, rng))
        return result

    def rejection_sample(self, *, seed, minimal_length, maximal_length):
        rng = np.random.RandomState(seed)
        length = rng.choice(list(range(minimal_length, 1 + maximal_length)))
        while True:
            result = self.sample("start", rng)
            if len(result) == length:
                return result

    def all_tokens(self):
        tokens = []
        for productions in self.result.values():
            for production in productions:
                for tok in production.elements:
                    if isinstance(tok, str):
                        continue
                    tokens.append(tok)
        return sorted(tokens)

    def render(self):
        return "\n".join(
            k + " ::= " + " | ".join(v.render() for v in vs)
            for k, vs in self.result.items()
        )

    def all_symbols(self):
        return sorted(
            {sym for token in self.all_tokens() for sym in token.rendered_symbols}
        )


@attr.s
class StraightlineCFG:
    symbols = attr.ib()

    def all_tokens(self):
        return [Token.single_symbol(sym) for sym in self.symbols]

    def all_symbols(self):
        return sorted(self.symbols)
