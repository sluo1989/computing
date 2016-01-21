#include <iostream>
#include <iterator>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <algorithm>

const int size=10;

int main() {
    thrust::host_vector<double> x(size), y(size);

    thrust::sequence(x.begin(), x.end()); 
    thrust::sequence(y.begin(), y.end(), 0.0, 0.1); 
    
    thrust::device_vector<double> d_x(x), d_y(y), d_z(size); 
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_z.begin(),
                      thrust::plus<double>());
    
    thrust::host_vector<double> z(d_z); 
    
    std::copy(z.begin(), z.end(), 
              std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl; 
    
    return 0; 
}
