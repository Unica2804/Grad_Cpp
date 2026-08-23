#include "include/grad_engine.hpp"

#include <iostream>

int main() {
	auto x = Value::create(2.0);
	auto y = Value::create(3.0);
	auto out = (x * y) + 2*x + 3*y;

	out->backward();

	std::cout << "Output: " << out->data << '\n';
	std::cout << "Gradient of x: " << x->grad << '\n';
	std::cout << "Gradient of y: " << y->grad << '\n';
}
