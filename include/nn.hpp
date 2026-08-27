#pragma once
#include "engine.hpp"
#include <vector>
#include <random>

namespace cppgrad {
class Module {
    public:
        virtual ~Module() = default;
        virtual std::vector<Valueptr> parameters() = 0;

        void zero_grad() {
            for (auto& param : parameters()) {
                param->grad = 0.0;
            }
        }
        
};

inline double random_uniform(double min = -1.0, double max = 1.0) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

class Neuron : public Module {
    public:
        std::vector<Valueptr> w;
        Valueptr b;
        bool nonlin;

        Neuron(size_t nin, bool nonlin = true) : nonlin(nonlin) {
            for (size_t i=0; i<nin; ++i) {
                w.push_back(Value::create(random_uniform(-1.0, 1.0)));
            }
            b = Value::create(random_uniform(-1.0, 1.0));
        }

        Valueptr forward(const std::vector<Valueptr>& x) {
            Valueptr act = b;
            for (size_t i=0; i<w.size(); ++i) {
                act = act + (w[i] * x[i]);
            }
            return nonlin ? tanh(act) : act;
        }

        std::vector<Valueptr> parameters() override {
            std::vector<Valueptr> params = w;
            params.push_back(b);
            return params;
        }
};

class layer : public Module {
    public:
        std::vector<Neuron> neurons;

        layer(size_t nin, size_t nout, bool nonlin = true) {
            for (size_t i=0; i<nout; ++i) {
                neurons.emplace_back(nin, nonlin);
            }
        }

        std::vector<Valueptr> forward(const std::vector<Valueptr>& x) {
            std::vector<Valueptr> out;
            for (auto& neuron : neurons) {
                out.push_back(neuron.forward(x));
            }
            return out;
        }

        std::vector<Valueptr> parameters() override {
            std::vector<Valueptr> params;
            for (auto& neuron : neurons) {
                auto neuron_params = neuron.parameters();
                params.insert(params.end(), neuron_params.begin(), neuron_params.end());
            }
            return params;
        }
};

class MLP : public Module {
    public:
        std::vector<layer> layers;

        MLP(size_t nin, const std::vector<size_t>& nouts) {
            std::vector<size_t> sz = {nin};
            sz.insert(sz.end(), nouts.begin(), nouts.end());
            for (size_t i=0; i<nouts.size(); ++i) {
                bool nonlin = (i != nouts.size() - 1);
                layers.emplace_back(sz[i], sz[i+1], nonlin);
            }
        }

        std::vector<Valueptr> forward(std::vector<Valueptr> x) {
            for (auto& layer : layers) {
                x = layer.forward(x);
            }
            return x;
        }

        std::vector<Valueptr> parameters() override {
            std::vector<Valueptr> params;
            for (auto& layer : layers) {
                auto layer_params = layer.parameters();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            }
            return params;
        }
};
}