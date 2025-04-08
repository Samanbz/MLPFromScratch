// MLPFromScratch.cpp : Defines the entry point for the application.
//

#include "MLPFromScratch.h"

#include "./src/Functions/Activation.h"
#include "./src/MLP/MLP.h"

using namespace std;

int main() {
    MLP mlp({1, 3, 1});

    Vector in(vector({1.0}));

    Vector out = mlp.forward(in);

    cout << "Input: " << in.to_string() << endl;

    cout << "Output: " << out.to_string() << endl;

    try {
        // Backward propagation
        mlp.backward(Vector(1, 3));
    } catch (const std::exception& e) {
        cout << "An error occurred: " << e.what() << endl;
    } catch (...) {
        cout << "An unknown error occurred." << endl;
    }
}
