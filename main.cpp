#include "include/engine.hpp"

#include <iostream>

using namespace cppgrad;

int main() {
	auto x = Value::create(2.0);
	auto y = Value::create(3.0);
	auto out = (x * y) + (x/y);

	out->backward();

	std::cout << "Output: " << out->data << '\n';
	std::cout << "Gradient of x: " << x->grad << '\n';
	std::cout << "Gradient of y: " << y->grad << '\n';
}
