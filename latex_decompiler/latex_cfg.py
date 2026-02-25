import string

from .cfg import CFG, Production, Token


def symbol_productions(base_chars="0123456789xyzabc"):
    return [Production([Token(sym, sym, (sym,))]) for sym in base_chars]


def symbol_with_sup_sub(*, single_symbol_id):
    return [
        Production([single_symbol_id]),
        Production(
            [
                single_symbol_id,
                Token("SUP(", "^{", ()),
                single_symbol_id,
                Token("SUP)", "}", ()),
            ]
        ),
        Production(
            [
                single_symbol_id,
                Token("SUB(", "_{", ()),
                single_symbol_id,
                Token("SUB)", "}", ()),
            ]
        ),
    ]


def sequential_expression_rule(*, sequence_id, item_id, infix_tokens, weight=0.25):
    return {
        sequence_id: [
            Production([item_id]),
            Production(
                [
                    item_id,
                    *[Token(t, t, t) for t in infix_tokens],
                    sequence_id,
                ],
                weight=weight,
            ),
        ]
    }


def addition_and_multiplication_rules(*, item_id, term_id, expression_id):
    return {
        **sequential_expression_rule(
            sequence_id=expression_id, item_id=term_id, infix_tokens=["+"]
        ),
        **sequential_expression_rule(
            sequence_id=term_id, item_id=item_id, infix_tokens=[]
        ),
    }


latex_cfg_hard = CFG(
    {
        "start": [
            Production(["straightline_expression"]),
            Production(
                [
                    Token("FRAC(", r"\frac{", ()),
                    "start",
                    Token("FRACMID", r"}{", ()),
                    "start",
                    Token("FRAC)", r"}", ()),
                ],
                weight=0.25,
            ),
        ],
        **addition_and_multiplication_rules(
            item_id="atom",
            term_id="term",
            expression_id="straightline_expression",
        ),
        "atom": [
            Production(["combined_symbol"]),
            Production(
                [
                    Token("PAREN(", r"\left(", "("),
                    "straightline_expression",
                    Token("PAREN)", r"\right)", ")"),
                ],
                weight=0.25,
            ),
            Production(
                [
                    Token("PAREN(", r"\left[", "["),
                    "straightline_expression",
                    Token("PAREN)", r"\right]", "]"),
                ],
                weight=0.25,
            ),
        ],
        "combined_symbol": symbol_with_sup_sub(single_symbol_id="symbol"),
        "symbol": symbol_productions(base_chars=string.ascii_letters),
    }
)

latex_cfg = CFG(
    {
        "start": [
            Production(["straightline_expression"]),
            Production(
                [
                    Token("FRAC(", r"\frac{", ()),
                    "start",
                    Token("FRACMID", r"}{", ("FRACBABR",)),
                    "start",
                    Token("FRAC)", r"}", ()),
                ],
                weight=0.25,
            ),
        ],
        **addition_and_multiplication_rules(
            item_id="atom",
            term_id="term",
            expression_id="straightline_expression",
        ),
        "atom": [
            Production(["combined_symbol"]),
            Production(
                [
                    Token("PAREN(", r"\left(", "("),
                    "straightline_expression",
                    Token("PAREN)", r"\right)", ")"),
                ],
                weight=0.25,
            ),
        ],
        "combined_symbol": symbol_with_sup_sub(single_symbol_id="symbol"),
        "symbol": symbol_productions(),
    }
)


def sequence_of_symbols(**kwargs):
    return CFG(
        {
            **sequential_expression_rule(
                sequence_id="start",
                item_id="symbol",
                infix_tokens=[],
                weight=1,
            ),
            "symbol": symbol_productions(**kwargs),
        }
    )


latex_cfg_just_string_of_symbols = sequence_of_symbols()

latex_cfg_just_string_of_symbols_sup_sub = CFG(
    {
        **sequential_expression_rule(
            sequence_id="start",
            item_id="combined_symbol",
            infix_tokens=[],
            weight=1,
        ),
        "combined_symbol": symbol_with_sup_sub(single_symbol_id="symbol"),
        "symbol": symbol_productions(),
    }
)

LATEX_CFG_SPECS = {
    "latex_cfg_just_string_of_symbols": dict(
        cfg=latex_cfg_just_string_of_symbols, maximal_length=7
    ),
    "latex_cfg_just_string_of_symbols_sup_sub": dict(
        cfg=latex_cfg_just_string_of_symbols_sup_sub, maximal_length=23
    ),
    "latex_cfg": dict(cfg=latex_cfg, maximal_length=30),
    "latex_cfg_hard": dict(cfg=latex_cfg_hard, maximal_length=30),
}
