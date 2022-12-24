\* All TLA+ specs must start with at least 4 - on each side of MODULE. Everything outside these 2
\* boundaries is ignored.
---------------------------------------- MODULE SimpleMath ----------------------------------------

(******************************************************************************
cf. https://github.com/tlaplus/Examples/blob/master/specifications/SpecifyingSystems/SimpleMath/SimpleMath.tla
******************************************************************************)

\* Declares values a, ... , g so they can be used in formulas.
CONSTANTS a, b, c, d, e, f, g

ASSUME
  {a, b, c} \ {c} = {a, b}

ASSUME
  {a, b} \in {{a, b}, c, {d, e}}

\* All TLA+ spects must have four = at the end, for backwards compatibility reasons.
===================================================================================================