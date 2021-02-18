# distutils: language = c++

# from libc.stdlib cimport free

cdef class Data:

    # cdef _Data* _c_data
    cdef _Data c_data

    def __cinit__(self, vector[float] u, vector[float] v,
                  vector[float] w, vector[float] temp, vector[float] sal,
                  vector[float] ssh):
        self.c_data = _Data(u, v, w, temp, sal, ssh)

    def __dealloc__(self):
    #     # free(self.c_data)
        self.c_data.u.clear()
        self.c_data.v.clear()
        self.c_data.w.clear()
        self.c_data.temp.clear()
        self.c_data.sal.clear()
        self.c_data.ssh.clear()

    @property
    def u(self):
        return self.c_data.u
    @u.setter
    def u(self, vector[float] u):
        self.c_data.u = u

    @property
    def v(self):
        return self.c_data.v
    @v.setter
    def v(self, vector[float] v):
        self.c_data.v = v

    @property
    def w(self):
        return self.c_data.w
    @w.setter
    def w(self, vector[float] w):
        self.c_data.w = w

    @property
    def temp(self):
        return self.c_data.temp
    @temp.setter
    def temp(self, vector[float] temp):
        self.c_data.temp = temp

    @property
    def sal(self):
        return self.c_data.sal
    @sal.setter
    def sal(self, vector[float] sal):
        self.c_data.sal = sal

    @property
    def ssh(self):
        return self.c_data.ssh
    @ssh.setter
    def ssh(self, vector[float] ssh):
        self.c_data.ssh = ssh