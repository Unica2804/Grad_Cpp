# cppgrad

`cppgrad` is a small C++ reverse-mode automatic differentiation engine. It
represents each number as a `Value` node in a computation graph. Operations
connect nodes together, and calling `backward()` walks that graph in reverse
topological order to calculate gradients.

## What is autograd?

Automatic differentiation (autograd) calculates derivatives by applying the
chain rule to the operations that produced a value. It is different from
symbolic differentiation, which manipulates an entire formula, and numerical
differentiation, which approximates a derivative using finite differences.

For a scalar output $L$ and an input $x$, the gradient is written as
$\frac{\partial L}{\partial x}$. Gradients answer the question: "How much
would the output change if this input changed slightly by a small amount h?"  The formula is $\frac{f(a+h) - f(a)}{h}$

`cppgrad` uses reverse-mode autodiff. This is especially useful when there is
one scalar output and several inputs, as in a loss function for machine
learning.

## How gradients are calculated

Every `Value` stores:

- `data`: its numerical value.
- `_prev`: the input nodes that produced it.
- `grad`: the accumulated derivative of the final output with respect to it.
- `_backward`: the local chain-rule calculation for its operation.

When `backward()` is called, the engine:

1. Builds a topological ordering of the graph with a depth-first search.
2. Sets the output gradient to `1.0`, because $\frac{\partial L}{\partial L}=1$.
3. Visits the nodes in reverse order and runs each node's `_backward` function.
4. Adds contributions with `+=`, which is important when a value is used more
   than once in the graph.

### Operator by operator

The local derivatives implemented by the engine are:

| Operation | Forward value | Gradient contribution |
| --- | --- | --- |
| `z = x + y` | $z=x+y$ | $\frac{\partial L}{\partial x} += \frac{\partial L}{\partial z}$, $\frac{\partial L}{\partial y} += \frac{\partial L}{\partial z}$ |
| `z = x * y` | $z=xy$ | $\frac{\partial L}{\partial x} += y\frac{\partial L}{\partial z}$, $\frac{\partial L}{\partial y} += x\frac{\partial L}{\partial z}$ |
| `z = pow(x, p)` | $z=x^p$ | $\frac{\partial L}{\partial x} += px^{p-1}\frac{\partial L}{\partial z}$ |
| `z = -x` | $z=-x$ | Implemented as `x * -1.0` |
| `z = x - y` | $z=x-y$ | Implemented as `x + (-y)` |
| `z = x / y` | $z=x/y$ | Implemented as `x * pow(y, -1.0)` |

The `+=` in each formula means that gradients are accumulated as they flow
back toward the leaves. Constants used in expressions are wrapped in
constant `Value` nodes automatically.

## Worked gradient example

Consider:

$$
L = x y + \frac{x}{y}, \qquad x=2,\ y=3
$$

The forward pass gives:

$$
L = 2\cdot3 + \frac{2}{3} = 6.6667
$$

Using the product and quotient rules:

$$
\frac{\partial L}{\partial x} = y + \frac{1}{y} = 3 + \frac{1}{3} = 3.3333
$$

$$
\frac{\partial L}{\partial y} = x - \frac{x}{y^2} = 2 - \frac{2}{9} = 1.7778
$$

Internally, division is represented as multiplication by $y^{-1}$. The
backward pass first propagates through the multiplication nodes, then through
the power node for $y^{-1}$, producing the same result through the chain rule.

## Usage

Include the header and create values with `Value::create`. Build an expression
using the supported operators, then call `backward()` on the output.

```cpp
#include "include/engine.hpp"

#include <iostream>

using namespace cppgrad;

int main() {
	auto x = Value::create(2.0);
	auto y = Value::create(3.0);

	auto output = (x * y) + (x / y);
	output->backward();

	std::cout << "Output: " << output->data << '\\n';
	std::cout << "Gradient of x: " << x->grad << '\\n';
	std::cout << "Gradient of y: " << y->grad << '\\n';
}
```

Expected values are approximately:

```text
Output: 6.66667
Gradient of x: 3.33333
Gradient of y: 1.77778
```

Powers can be created with `pow`, and scalar operands are supported:

```cpp
auto x = Value::create(2.0);
auto output = pow(x, 3.0) + 2.0 * x - 1.0;
output->backward();

// output = 11, and x->grad = 3 * 2^2 + 2 = 14
```

## Build and run

The project is header-only. A compiler with C++11 support is sufficient:

```bash
g++ -std=c++11 -I. main.cpp -o cppgrad
./cppgrad
```

## Current scope

The engine currently supports scalar `double` values and the operators `+`,
`-`, `*`, `/`, unary negation, and `pow(Value, exponent)`. It does not yet
provide tensors, common functions such as `exp` or `log`, gradient reset
helpers, or an optimizer.

## Acknowledgements

The project is inspired by the [micrograd](https://github.com/karpathy/micrograd) project. Which is a minimalistic Python implementation of a reverse-mode autodiff engine by Andrej Karpathy. The C++ implementation is a learning exercise and is not intended for production use.
