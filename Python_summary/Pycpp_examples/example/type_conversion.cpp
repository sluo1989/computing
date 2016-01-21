#include <Python.h>
#include <boost/python.hpp>

struct Coordinate {
    Coordinate (long x, long y) : x_(x), y_(y) {}
    long x_; 
    long y_;
};

struct Coordinate_to_tuple
{
    static PyObject* convert(const Coordinate& c)
    {
        return boost::python::incref<>(boost::python::make_tuple(c.x_, c.y_).ptr());
    }
};

boost::python::to_python_converter<Coordinate, Coordiate_to_tuple>();

static void* convertible(PyObject* obj_ptr)
{
    if (!PyTuple_Check(obj_ptr)) return 0; 
    if (PyTuple_Size(obj_ptr) != 2) return 0;
    if (!PyNumber_Check(PyTuple_GetItem(obj_ptr, 0))) return 0;
    if (!PyNumber_Check(PyTuple_GetItem(obj_ptr, 1))) return 0; 
    
    return obj_ptr;
}

static void construct(PyObject* obj_ptr,
                      boost::python::converter::rvalue_from_python_stage1_data* data)
{
    long x_coord = PyLong_AsLong(PyNumber_Long(PyTuple_GetItem(obj_ptr, 0)));
    long y_coord = PyLong_AsLong(PyNumber_Long(PyTuple_GetItem(obj_ptr, 1)));

    void* storage = ((boost::python::converter::rvalue_from_python_storage<Coordinate>*)
                     data)->storage.bytes;
    new (storage) Coordinate(x_coord, y_coord);
    
    data->convertible = storage;
}

boost::python::converter::registry::push_back(
    &convertible, 
    &construct,
    boost::python::type_id<Coordinate>());
