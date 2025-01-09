
#include <iostream>
#include <iomanip>
#include <chrono>

// Function to perform the calculation
double calculate(int iterations, int param1, int param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        int j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }
    return result;
}

int main() {
    // Start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate the result
    double result = calculate(100000000, 4, 1) * 4;

    // End time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = end_time - start_time;
    
    // Display the result
    std::cout << std::fixed << std::setprecision(12) << "Result: " << result << std::endl;
    std::cout << std::fixed << std::setprecision(6) << "Execution Time: " << exec_time.count() << " seconds" << std::endl;
    
    return 0;
}
