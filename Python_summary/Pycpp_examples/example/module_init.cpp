#include <fstream>
#include <boost/python.hpp>

void initialize() {
    std::ofstream ofs("module_init.txt"); 
    ofs << "Module has been initialized!";
}

BOOST_PYTHON_MODULE(module_init) {
    initialize(); 
}
