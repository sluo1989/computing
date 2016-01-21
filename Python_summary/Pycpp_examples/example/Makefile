#location of the Python header files and library
PYTHON_INC = /Library/Frameworks/Python.framework/Versions/2.7/include/python2.7
PYTHON_LIB = /Library/Frameworks/Python.framework/Versions/2.7/lib

# location of the Boost Python include files and library
BOOST_INC = /usr/local/include
BOOST_LIB = /usr/local/lib
 
# compile target
TARGET = hello

$(TARGET).so: $(TARGET).o
	g++ -shared $(TARGET).o -L$(BOOST_LIB) -lboost_python -L$(PYTHON_LIB) -lpython2.7 -o $(TARGET).so
 
$(TARGET).o: $(TARGET).cpp
	g++ -I$(PYTHON_INC) -I$(BOOST_INC) -fPIC -c $(TARGET).cpp
