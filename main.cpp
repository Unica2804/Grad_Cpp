#include "include/engine.hpp"
#include "include/nn.hpp"
#include <iostream>
#include <vector>

using namespace cppgrad;

int main() {
    // 1. Initialize MLP: 2 inputs, two hidden layers of 4 neurons, 1 output neuron
    MLP model(2, {4, 4, 1});

    // 2. Toy dataset (4 samples with 2 features each)
    std::vector<std::vector<double>> raw_X = {
        {2.0, 3.0},
        {3.0, -1.0},
        {0.5, 1.0},
        {1.0, 1.0}
    };
    std::vector<double> raw_y = {1.0, -1.0, -1.0, 1.0}; // Targets

    std::cout << "Number of parameters: " << model.parameters().size() << "\n\n";

    // 3. Training Loop (Gradient Descent)
    double learning_rate = 0.05;

    for (int epoch = 0; epoch < 200; ++epoch) {
        // --- Forward Pass & Loss Computation ---
        Valueptr total_loss = Value::create(0.0);

        for (size_t i = 0; i < raw_X.size(); ++i) {
            std::vector<Valueptr> x = {Value::create(raw_X[i][0]), Value::create(raw_X[i][1])};
            Valueptr y_target = Value::create(raw_y[i]);

            auto y_pred = model.forward(x)[0];
            
            // Squared error: (y_pred - y_target)^2
            auto diff = y_pred - y_target;
            auto loss = diff * diff;
            total_loss = total_loss + loss;
        }

        // --- Backward Pass ---
        model.zero_grad();
        total_loss->backward();

        // --- SGD Parameter Update ---
        for (auto& p : model.parameters()) {
            p->data -= learning_rate * p->grad;
        }
		std::cout << "Epoch " << epoch << " | Loss: " << total_loss->data << "\n";
    }

    // 4. Inference on Trained Model
    std::cout << "\nPredictions after training:\n";
    for (size_t i = 0; i < raw_X.size(); ++i) {
        std::vector<Valueptr> x = {Value::create(raw_X[i][0]), Value::create(raw_X[i][1])};
        auto pred = model.forward(x)[0];
        std::cout << "Target: " << raw_y[i] << " | Predicted: " << pred->data << "\n";
    }

    return 0;
}