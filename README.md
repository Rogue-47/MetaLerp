# MetaLerp
All the initial math, source implementation, and ideas behind this library are solely the work of Omar M. Mahmoud / Rogue-47

MetaLerp is a lightweight, heterogenuous (soon), and fast behavior construction library for monotonically increasing and decreasing mathematical functions
Can be used in: Dynamic Simulations, Graphics transformations and Shaders, Animation (customized control over lerp-based behavior), Machine Learning (as very fast (almost ReLU-fast) activation or normalization/standardization functions), Signal Systems, and more.
Generally it can be applied in any domain where non-linear transformation of something to an unbounded or bounded range in many ways, especially in situations where your values to transform are unbonuded (-inf, +inf); this extension to 2D linear interpolation makes the t step parameter accept any value and scale according to the range.

behaviors include: ascending, descending; with even or odd symmetry about the y axis - and modular hyperparameters and function construction around the y-axis in a piecewise left-right arm separated by the x=0 line,
and parametric forms that can approximate other continuous functions accurately (like the Sigmoid approximator demonstration)

Proofs have been done on paper to prove that the formulas respect and stay true to the bounding behavior; the inverses have been constructed from the base and parametric forms afterward with safe range clamping to produce unbounded behavior functions that allow the library to emulate a wider domain of functions.

Written in C with careful optimizations to unleash the speed potential of the deceptively simple mathematical structures of these formulas (simple arithmetic structure, etc.)

demonstrations:
[https://www.desmos.com/calculator/s3lmhxwrrh](https://www.desmos.com/calculator/cf0389db8e)
demonstration with approximators:
[https://www.desmos.com/calculator/cbwlyfi7yt](https://www.desmos.com/calculator/1792cfd969)


Collaboration in expanding, refining, polishing, and demonstrating/making more examples of the extensive capabilities of this open-source library and the math behind it, is welcome.
The combinational function constructor (in combos.h) of this library could very well be used to form all kinds of interestingly/weirdly behaving functions (and many useful ones, like the approximators released in the earlier phases of the lib) that will take much time to analyze and study with right-left combo & hyperparameter adjustments, etc.

