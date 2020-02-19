# distutils: language = c++

cdef class Data:

    # cdef _Data* _c_data
    cdef _Data c_data

    def __cinit__(self, vector[long] datetime, vector[double] u, vector[double] v,
                  vector[double] w, vector[double] temp, vector[double] sal):
        self.c_data = _Data(datetime, u, v, w, temp, sal)

    def __dealloc__(self):
        self.c_data.datetime.clear()
        self.c_data.u.clear()
        self.c_data.v.clear()
        self.c_data.w.clear()
        self.c_data.temp.clear()
        self.c_data.sal.clear()

    @property
    def datetime(self):
        return self.c_data.datetime
    @datetime.setter
    def datetime(self, vector[long] datetime):
        self.c_data.datetime = datetime

    @property
    def u(self):
        return self.c_data.u
    @u.setter
    def u(self, vector[double] u):
        self.c_data.u = u

    @property
    def v(self):
        return self.c_data.v
    @v.setter
    def v(self, vector[double] v):
        self.c_data.v = v

    @property
    def w(self):
        return self.c_data.w
    @w.setter
    def w(self, vector[double] w):
        self.c_data.w = w

    @property
    def temp(self):
        return self.c_data.temp
    @temp.setter
    def temp(self, vector[double] temp):
        self.c_data.temp = temp

    @property
    def sal(self):
        return self.c_data.sal
    @sal.setter
    def sal(self, vector[double] sal):
        self.c_data.sal = sal



