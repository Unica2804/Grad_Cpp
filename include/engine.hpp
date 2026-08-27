#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>

namespace cppgrad {

class Value;
using Valueptr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value> {
    public:

        double data;
        std::vector<Valueptr> _prev;
        std::string op;
        double grad;
        std::function<void()> _backward;
        
        // Init constructor
        Value(
            double data, 
            const std::vector<Valueptr>& children = {},
            const std::string& op = ""
        ) : data(data),grad(0.0),op(op),_prev(children),_backward([](){}) {}

        // Helper method to create a shared pointer of Value
        static Valueptr create(
            double data,
            const std::vector<Valueptr>& children = {},
            const std::string& op = ""
        ) {
            return std::make_shared<Value>(data, children, op);
        }

        // Topological sort and autodiff
        void backward() {
            std::vector<Valueptr> topo;
            std::unordered_set<Value*> visited;

            //Depth first search to build the topological order
            std::function<void(const Valueptr&)> build_topo = [&](const Valueptr& v) {
                if (visited.find(v.get()) == visited.end()) {
                    visited.insert(v.get());
                    for (const auto& child : v->_prev) {
                        build_topo(child);
                    }
                    topo.push_back(v);
                }
            };

            build_topo (shared_from_this());
            // Set the gradient of o/p node to 1.0
            this->grad = 1.0;
            // apply the chain rule in reverse topological order
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                (*it)->_backward();
            }
            
        }
        
};

// + operator overload
inline Valueptr operator+(const Valueptr& self,const Valueptr& other) {
    auto out = Value::create(self->data + other->data, {self, other}, "+");
    out->_backward = [self, other, out]() {
        self->grad += out->grad;
        other->grad += out->grad;
    };
    return out;
}

// If a value which is not a shared pointer is added to a Valueptr, we need to overload the + operator for double as well
inline Valueptr operator+(const Valueptr& self,double other) {return self + Value::create(other);}
inline Valueptr operator+(double other,const Valueptr& self) {return self + Value::create(other);}

// * operator overload
inline Valueptr operator*(const Valueptr& self,const Valueptr& other) {
    auto out = Value::create(self->data * other->data, {self, other}, "*");
    out->_backward = [self, other, out] () {
        self->grad += other->data * out->grad;
        other->grad += self->data * out->grad;
    };
    return out;
}
inline Valueptr operator*(const Valueptr& self,double other) {return self * Value::create(other);}
inline Valueptr operator*(double other,const Valueptr& self) {return self * Value::create(other);}

// pow operator overload
inline Valueptr pow(const Valueptr& self,double exponent) {
    auto out = Value::create(std::pow(self->data, exponent), {self}, "^" + std::to_string(exponent));
    out->_backward = [self, exponent, out] () {
        self->grad += exponent * std::pow(self->data, exponent - 1) * out->grad;
    };
    return out;
}

// Negation operator overload
inline Valueptr operator-(const Valueptr& self) { return self * -1.0; }
inline Valueptr operator-(const Valueptr& self, const Valueptr& other) { return self + (-other); }
inline Valueptr operator-(const Valueptr& self, double other) { return self - Value::create(other); }
inline Valueptr operator-(double self, const Valueptr& other) { return Value::create(self) - other; }

// Division operator overload

inline Valueptr operator/(const Valueptr& self, const Valueptr& other) { return self * pow(other, -1.0); }
inline Valueptr operator/(const Valueptr& self, double other) { return self / Value::create(other); }
inline Valueptr operator/(double self, const Valueptr& other) { return Value::create(self) / other; }

// exponential function overload
inline Valueptr exp(const Valueptr& self) {
    auto out = Value::create(std::exp(self->data), {self}, "exp");
    out->_backward = [self, out] () {
        self->grad += out->data * out->grad;
    };
    return out;
}

// tanh function overload
inline Valueptr tanh(const Valueptr& self) {
    auto out = Value::create(std::tanh(self->data), {self}, "tanh");
    out->_backward = [self, out] () {
        self->grad += (1 - out->data * out->data) * out->grad;
    };
    return out;
}

// relu function overload
inline Valueptr relu(const Valueptr& self) {
    auto out = Value::create(self->data > 0 ? self->data : 0.0, {self}, "ReLU");
    out->_backward = [self, out] () {
        self->grad += (self->data > 0 ? 1.0 : 0.0) * out->grad;
    };
    return out;
}

// sigmoid function overload
inline Valueptr sigmoid(const Valueptr& self) {
    auto out = Value::create(1.0 / (1.0 + std::exp(-self->data)), {self}, "sigmoid");
    out->_backward = [self, out] () {
        self->grad += out->data * (1 - out->data) * out->grad;
    };
    return out;
}

}