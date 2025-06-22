<!-- ---
title: "Foundations" 
description: "Foundations - the bacause everything must have a start" 
tags: ["mathematics", "programming"] 
difficulty: "beginner" 
category: "concept" 
symbol: "" 
prerequisites: [] 
related\_concepts: ["addition", "substraction", "multiplication", "division"] 
applications: ["programming"]
interactive: false 
code\_examples: true 
complexity\_analysis: false 
real\_world\_examples: true 
layout: "concept-page" 
date\_created: "2025-06-21" 
last\_updated: "2025-06-21" 
author: "" 
reviewers: [] 
version: "1.0"
--- -->

# Foundations

## The Basics
So everything has a start - a foundation. In a simple sense, nearly all math can be broken down to four operators and a set of rules. We have the four basic operators:
- Addition
- Substraction
- Multiplication
- Division

And a set of rules on how to apply these in different manners. Oh, I can feel your skepticisme, what about .. eh.. like differential equations. Well, let's look at that then.

## Common Math Symbols — Grouped and Explained

### Arithmetic & Basic Operators

| Symbol      | Name                         | Meaning                            | Programming Analogy           |
|-------------|------------------------------|------------------------------------|------------------------------|
| \( + \)     | Plus                         | Addition                            | `+`                          |
| \( - \)     | Minus                        | Subtraction                         | `-`                          |
| \( \times \), \( \cdot \), `*` | Multiplication                   | Multiply                          | `*`                          |
| \( \div \), `/` | Division                     | Divide                              | `/`                          |
| \( = \)     | Equals                       | Equality                             | `==` (compare), `=` (assign) |
| \( \neq \)  | Not equal                    | Not equal                            | `!=`                         |
| \( < \), \( > \) | Less than / Greater than      | Compare values                       | `<`, `>`                     |
| \( \leq \), \( \geq \) | Less than or equal / Greater than or equal | Compare with equality           | `<=`, `>=`                   |

---

### Algebra & Sets

| Symbol      | Name                         | Meaning                            | Programming Analogy           |
|-------------|------------------------------|------------------------------------|------------------------------|
| \( \in \)   | Element of                   | "is in" set                         | `in`                         |
| \( \notin \) | Not element of              | "not in" set                        | `not in`                     |
| \( \subset \), \( \subseteq \) | Subset / Subset or equal            | Subset                          |                              |
| \( \cup \)  | Union                        | Combine sets                        | `set.union()`                 |
| \( \cap \)  | Intersection                 | Common elements between sets        | `set.intersection()`          |
| \( \emptyset \), \( \varnothing \) | Empty set                       | No elements                      | `set()`                       |

---

### Calculus

| Symbol      | Name                         | Meaning                            | Programming Analogy           |
|-------------|------------------------------|------------------------------------|------------------------------|
| \( \Sigma \) | Summation                   | Sum over a range                    | `sum()` / for loop            |
| \( \prod \)  | Product                     | Product over a range                | `math.prod()`                 |
| \( \int \)   | Integral                    | Continuous sum / area under curve   | accumulate with small steps   |
| \( \partial \) | Partial derivative        | Derivative with respect to one variable | function sensitivity         |
| \( \nabla \) | Gradient                    | Vector of derivatives               | `gradient()` in ML            |
| \( \Delta \) | Change / difference         | \( \Delta x = x_2 - x_1 \)          | diff(x)                       |

---

### Logic

| Symbol      | Name                         | Meaning                            | Programming Analogy           |
|-------------|------------------------------|------------------------------------|------------------------------|
| \( \land \) | And (logical)                | both true                           | `and`                        |
| \( \lor \)  | Or (logical)                 | either true                         | `or`                         |
| \( \lnot \), \( \neg \) | Not (logical)     | invert truth                        | `not`                        |
| \( \Rightarrow \) | Implies                | if A then B                         | `if A: B`                     |
| \( \Leftrightarrow \) | If and only if     | logical equivalence                 | A iff B                      |

---

### Constants & Special Symbols

| Symbol      | Name                         | Meaning                            | Programming Analogy           |
|-------------|------------------------------|------------------------------------|------------------------------|
| \( \pi \)   | Pi                           | Circle constant ≈ 3.14159           | `math.pi`                     |
| \( e \)     | Euler’s number                | ≈ 2.71828, base of natural log      | `math.e`                      |
| \( i \)     | Imaginary unit                | \( \sqrt{-1} \)                     | complex numbers               |
| \( \infty \) | Infinity                     | unbounded value                     | `float('inf')`                |
| \( \approx \) | Approximately equal         | nearly equal                        | `~=` or `math.isclose()`      |

---

### Exponentials & Logarithms

| Symbol      | Name                         | Meaning                            | Programming Analogy           |
|-------------|------------------------------|------------------------------------|------------------------------|
| \( \log \), \( \ln \) | Logarithm / Natural logarithm | Solve for exponent               | `math.log()`                  |
| \( \exp \)   | Exponential function         | \( e^x \), continuous growth        | `math.exp()`                  |
| \( a^b \)    | Exponentiation               | Raise to power                      | `**`                         |

---

### Greek Letters — Commonly used

| Symbol      | Name                         | Typical Use                        |
|-------------|------------------------------|------------------------------------|
| \( \alpha \) | alpha                        | angle, learning rate (ML)           |
| \( \beta \)  | beta                         | regression coefficient              |
| \( \gamma \) | gamma                        | gamma function, discount in RL      |
| \( \delta \) | delta                        | change, difference                  |
| \( \epsilon \) | epsilon                     | small positive number (tolerance)   |
| \( \zeta \)  | zeta                         | Riemann zeta function                |
| \( \eta \)   | eta                          | efficiency, learning rate            |
| \( \theta \) | theta                        | angle, model parameter              |
| \( \lambda \) | lambda                       | eigenvalues, decay rate, regularization |
| \( \mu \)    | mu                           | mean of distribution                |
| \( \nu \)    | nu                           | frequency                           |
| \( \xi \)    | xi                           | random variable                     |
| \( \rho \)   | rho                          | correlation coefficient             |
| \( \sigma \) | sigma                        | standard deviation, sum             |
| \( \tau \)   | tau                          | time constant, Kendall’s tau        |
| \( \phi \)   | phi                          | golden ratio, angle                 |
| \( \omega \) | omega                        | angular frequency                   |
| \( \Omega \) | big Omega                    | complexity lower bound               |

---

### ML & Data Science Symbols

| Symbol      | Name                         | Common Use in ML / DS               |
|-------------|------------------------------|------------------------------------|
| \( \theta \) | theta                        | model parameters (linear, logistic regression) |
| \( w \), \( b \) | weights, bias             | neural networks, linear models      |
| \( \nabla J(\theta) \) | gradient of loss wrt parameters | gradient descent update            |
| \( L \), \( \mathcal{L} \) | loss, likelihood | model objective                     |
| \( y \), \( \hat{y} \) | actual output, predicted output | supervised learning                 |
| \( p(x) \), \( p(x|y) \) | probability, conditional probability | probabilistic models               |
| \( \mu \), \( \sigma^2 \) | mean and variance | Gaussian models, distributions      |
| \( \mathbb{E}[X] \) | expectation of X       | expected value                      |
| \( KL(P||Q) \) | Kullback-Leibler divergence | measure difference between distributions |
| \( \mathcal{N}(\mu, \sigma^2) \) | normal distribution | Gaussian probability density       |
| \( R^2 \) | coefficient of determination    | regression goodness of fit          |


With that in mind, most math problems don't seem so scary.

## Symbols
The dear mathematicians, they do love their weird way of writing things. So think of math as a new programming language, but the inventor of this one, wanted to make it special. The inventor then made most of the keywords (those we know like `if` or `int`) in this the new language to be made of symbols partialy from philosophy, partially made-up and instead of english, to describe it's concepts, a dead language was used*. Now you're stuck some weird symbols, strange words and people insists, that it makes perfect, logical sense.

Here is a list of most of the symbols you'll encounter, their name, functions and what the analog to a programming language could be. 

*On a sidenote, this is basically the story of the programming language APL, and you had to buy the special keyboard to be able to write programs in it [APL on wiki](https://en.wikipedia.org/wiki/APL_(programming_language)).

## Disclaimer
Matlab is an illigitimate programming language and it knows!