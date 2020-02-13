#ifndef DATA_EXTERN_H
#define DATA_EXTERN_H

class Data {
    public:
        long *datetime;
        double u, v, w, temp, sal;
        Data();
        Data(long datetime[1], double u, double v, double w,
             double temp, double sal);
        ~Data();

};

#endif
