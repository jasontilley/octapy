# distutils: language = c++

cdef class Data:

    cdef _Data *c_data
    cdef long[::1] datetime

    def __cinit__(self, long[::1] datetime, u, v, w, temp, sal):
        self.c_data = new _Data(&datetime[0], u, v, w, temp, sal)

    def __dealloc__(self):
        del self.c_data

    @property
    def datetime(self):
        return self.c_data.datetime
    @datetime.setter
    def datetime(self, datetime):
        self.c_data.datetime = datetime

    @property
    def u(self):
        return self.c_data.u
    @u.setter
    def u(self, u):
        self.c_data.u = u

    @property
    def v(self):
        return self.c_data.v
    @v.setter
    def v(self, v):
        self.c_data.v = v
        
    @property
    def w(self):
        return self.c_data.w
    @w.setter
    def w(self, w):
        self.c_data.w = w
        
    @property
    def temp(self):
        return self.c_data.temp
    @temp.setter
    def temp(self, temp):
        self.c_data.temp = temp
        
    @property
    def sal(self):
        return self.c_data.sal
    @sal.setter
    def sal(self, sal):
        self.c_data.sal = sal




