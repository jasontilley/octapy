#include <iostream>
#include "data extern.h"

// Default constructor
Data::Data () {}

// Overloaded constructor
Data::Data (long *datetime, double u, double v, double w, double temp,
            double sal) {
    this->datetime = &datetime[0];
    this->u = u;
    this->v = v;
    this->w = w;
    this->temp = temp;
    this->sal = sal;
}

// Destructor
Data::~Data () {}
