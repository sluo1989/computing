#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <iostream>

void greet(const std::string& name)
{
   std::cout << "Hello " + name + "!" << "\n";
}

BOOST_PYTHON_MODULE(hello)
{
    boost::python::def("greet", greet);
}
