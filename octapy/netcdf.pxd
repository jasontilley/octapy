cdef extern from "netcdf.h" nogil:
    ctypedef int nc_type
    cdef enum:
        NC_NAT
        NC_BYTE
        NC_CHAR
        NC_SHORT
        NC_INT
        NC_LONG
        NC_FLOAT
        NC_DOUBLE
        NC_UBYTE
        NC_USHORT
        NC_UINT
        NC_INT64
        NC_UINT64
        NC_STRING
        NC_MAX_ATOMIC_TYPE
    cdef enum:
        NC_VLEN
        NC_OPAQUE
        NC_ENUM
        NC_COMPOUND
        NC_FIRSTUSERTYPEID
    cdef enum:
        NC_FILL_BYTE
        NC_FILL_CHAR
        NC_FILL_SHORT
        NC_FILL_INT
        NC_FILL_FLOAT
        NC_FILL_DOUBLE
        NC_FILL_UBYTE
        NC_FILL_USHORT
        NC_FILL_UINT 
        NC_FILL_INT64
        NC_FILL_UINT64 
        NC_FILL_STRING
    cdef enum:
        NC_MAX_BYTE
        NC_MIN_BYTE
        NC_MAX_CHAR
        NC_MAX_SHORT
        NC_MIN_SHORT
        NC_MAX_INT
        NC_MIN_INT
        NC_MAX_FLOAT
        NC_MIN_FLOAT
        NC_MAX_DOUBLE
        NC_MIN_DOUBLE
        NC_MAX_UBYTE
        NC_MAX_USHORT
        NC_MAX_UINT
        NC_MAX_INT64
        NC_MIN_INT64
        NC_MAX_UINT64
    cdef enum:
        NC_FILL
        NC_NOFILL
    cdef enum:
        NC_NOWRITE
        NC_WRITE
        NC_CLOBBER
        NC_NOCLOBBER
        NC_DISKLESS
        NC_MMAP
        NC_64BIT_DATA
        NC_CDF5
        NC_CLASSIC_MODE
        NC_64BIT_OFFSET
    cdef enum:
        NC_LOCK
        NC_SHARE
        NC_NETCDF4
        NC_MPIIO
        NC_MPIPOSIX
        NC_INMEMORY 
        NC_PNETCDF
    cdef enum:
        NC_FORMAT_CLASSIC
        NC_FORMAT_64BIT_OFFSET
        NC_FORMAT_64BIT
        NC_FORMAT_NETCDF4
        NC_FORMAT_NETCDF4_CLASSIC
        NC_FORMAT_64BIT_DATA 
        NC_FORMAT_CDF5
    cdef enum:
        NC_FORMATX_NC3
        NC_FORMATX_NC_HDF5
        NC_FORMATX_NC4  
        NC_FORMATX_NC_HDF4 
        NC_FORMATX_PNETCDF 
        NC_FORMATX_DAP2
        NC_FORMATX_DAP4 
        NC_FORMATX_UNDEFINED
    cdef enum:
        NC_FORMAT_NC3 
        NC_FORMAT_NC_HDF5 
        NC_FORMAT_NC4 
        NC_FORMAT_NC_HDF4 
        NC_FORMAT_PNETCDF
        NC_FORMAT_DAP2 
        NC_FORMAT_DAP4 
        NC_FORMAT_UNDEFINED
    cdef enum:
        NC_SIZEHINT_DEFAULT
        NC_ALIGN_CHUNK
        NC_UNLIMITED
        NC_GLOBAL
    cdef enum:
        NC_MAX_DIMS
        NC_MAX_ATTRS
        NC_MAX_VARS
        NC_MAX_NAME
        NC_MAX_VAR_DIMS
        NC_MAX_HDF4_NAME
    cdef enum:
        NC_ENDIAN_NATIVE
        NC_ENDIAN_LITTLE
        NC_ENDIAN_BIG
    cdef enum:
        NC_CHUNKED
        NC_CONTIGUOUS
    cdef enum:
        NC_NOCHECKSUM
        NC_FLETCHER32
    cdef enum:
        NC_NOSHUFFLE
        NC_SHUFFLE
    cdef enum:
        NC_MIN_DEFLATE_LEVEL
        NC_MAX_DEFLATE_LEVEL
    cdef bint NC_ISYSERR(int err)
    cdef enum:
        NC_NOERR 
        NC2_ERR 
        NC_EBADID
        NC_ENFILE
        NC_EEXIST
        NC_EINVAL
        NC_EPERM
        NC_ENOTINDEFINE
        NC_EINDEFINE
        NC_EINVALCOORDS
        NC_EMAXDIMS
        NC_ENAMEINUSE
        NC_ENOTATT
        NC_EMAXATTS
        NC_EBADTYPE
        NC_EBADDIM
        NC_EUNLIMPOS
        NC_EMAXVARS
        NC_ENOTVAR
        NC_EGLOBAL
        NC_ENOTNC
        NC_ESTS
        NC_EMAXNAME
        NC_EUNLIMIT
        NC_ENORECVARS
        NC_ECHAR
        NC_EEDGE
        NC_ESTRIDE
        NC_EBADNAME
        NC_ERANGE
        NC_ENOMEM
        NC_EVARSIZE
        NC_EDIMSIZE
        NC_ETRUNC
        NC_EAXISTYPE
        NC_EDAP
        NC_ECURL
        NC_EIO
        NC_ENODATA
        NC_EDAPSVC
        NC_EDAS
        NC_EDDS
        NC_EDMR
        NC_EDATADDS
        NC_EDATADAP
        NC_EDAPURL
        NC_EDAPCONSTRAINT
        NC_ETRANSLATION
        NC_EACCESS
        NC_EAUTH
        NC_ENOTFOUND
        NC_ECANTREMOVE
        NC4_FIRST_ERROR
        NC_EHDFERR
        NC_ECANTREAD
        NC_ECANTWRITE
        NC_ECANTCREATE
        NC_EFILEMETA
        NC_EDIMMETA
        NC_EATTMETA
        NC_EVARMETA
        NC_ENOCOMPOUND
        NC_EATTEXISTS
        NC_ENOTNC4
        NC_ESTRICTNC3
        NC_ENOTNC3
        NC_ENOPAR
        NC_EPARINIT
        NC_EBADGRPID
        NC_EBADTYPID
        NC_ETYPDEFINED
        NC_EBADFIELD
        NC_EBADCLASS
        NC_EMAPTYPE
        NC_ELATEFILL
        NC_ELATEDEF
        NC_EDIMSCALE
        NC_ENOGRP
        NC_ESTORAGE
        NC_EBADCHUNK
        NC_ENOTBUILT
        NC_EDISKLESS
        NC_ECANTEXTEND
        NC_EMPI
        NC_EFILTER
        NC_ERCFILE
        NC_ENULLPAD
        NC4_LAST_ERROR
    cdef char *DIM_WITHOUT_VARIABLE = b"This is a netCDF dimension but not a netCDF variable."
    cdef enum:
        NC_HAVE_NEW_CHUNKING_API
        NC_EURL
        NC_ECONSTRAINT
    ctypedef struct nc_vlen_t:
        size_t len
        void *p
    const char *nc_inq_libvers()
    const char *nc_strerror(int ncerr)
    int nc__create(const char *path, int cmode, size_t initialsz, size_t *chunksizehintp, int *ncidp)
    int nc_create(const char *path, int cmode, int *ncidp)
    int nc__open(const char *path, int mode, size_t *chunksizehintp, int *ncidp)
    int nc_open(const char *path, int mode, int *ncidp)
    int nc_inq_path(int ncid, size_t *pathlen, char *path)
    int nc_inq_ncid(int ncid, const char *name, int *grp_ncid)
    int nc_inq_grps(int ncid, int *numgrps, int *ncids)
    int nc_inq_grpname(int ncid, char *name)
    int nc_inq_grpname_full(int ncid, size_t *lenp, char *full_name)
    int nc_inq_grpname_len(int ncid, size_t *lenp)
    int nc_inq_grp_parent(int ncid, int *parent_ncid)
    int nc_inq_grp_ncid(int ncid, const char *grp_name, int *grp_ncid)
    int nc_inq_grp_full_ncid(int ncid, const char *full_name, int *grp_ncid)
    int nc_inq_varids(int ncid, int *nvars, int *varids)
    int nc_inq_dimids(int ncid, int *ndims, int *dimids, int include_parents)
    int nc_inq_typeids(int ncid, int *ntypes, int *typeids)
    int nc_inq_type_equal(int ncid1, nc_type typeid1, int ncid2, nc_type typeid2, int *equal)
    int nc_def_grp(int parent_ncid, const char *name, int *new_ncid)
    int nc_rename_grp(int grpid, const char *name)
    int nc_def_compound(int ncid, size_t size, const char *name, nc_type *typeidp)
    int nc_insert_compound(int ncid, nc_type xtype, const char *name, size_t offset, nc_type field_typeid)
    int nc_insert_array_compound(int ncid, nc_type xtype, const char *name, size_t offset, nc_type field_typeid, int ndims, const int *dim_sizes)
    int nc_inq_type(int ncid, nc_type xtype, char *name, size_t *size)
    int nc_inq_typeid(int ncid, const char *name, nc_type *typeidp)
    int nc_inq_compound(int ncid, nc_type xtype, char *name, size_t *sizep, size_t *nfieldsp)
    int nc_inq_compound_name(int ncid, nc_type xtype, char *name)
    int nc_inq_compound_size(int ncid, nc_type xtype, size_t *sizep)
    int nc_inq_compound_nfields(int ncid, nc_type xtype, size_t *nfieldsp)
    int nc_inq_compound_field(int ncid, nc_type xtype, int fieldid, char *name, size_t *offsetp, nc_type *field_typeidp, int *ndimsp, int *dim_sizesp)
    int nc_inq_compound_fieldname(int ncid, nc_type xtype, int fieldid, char *name)
    int nc_inq_compound_fieldindex(int ncid, nc_type xtype, const char *name, int *fieldidp)
    int nc_inq_compound_fieldoffset(int ncid, nc_type xtype, int fieldid, size_t *offsetp)
    int nc_inq_compound_fieldtype(int ncid, nc_type xtype, int fieldid, nc_type *field_typeidp)
    int nc_inq_compound_fieldndims(int ncid, nc_type xtype, int fieldid, int *ndimsp)
    int nc_inq_compound_fielddim_sizes(int ncid, nc_type xtype, int fieldid, int *dim_sizes)
    int nc_def_vlen(int ncid, const char *name, nc_type base_typeid, nc_type *xtypep)
    int nc_inq_vlen(int ncid, nc_type xtype, char *name, size_t *datum_sizep, nc_type *base_nc_typep)
    int nc_free_vlen(nc_vlen_t *vl)
    int nc_free_vlens(size_t len, nc_vlen_t vlens[])
    int nc_put_vlen_element(int ncid, int typeid1, void *vlen_element, size_t len, const void *data)
    int nc_get_vlen_element(int ncid, int typeid1, const void *vlen_element, size_t *len, void *data)
    int nc_free_string(size_t len, char **data)
    int nc_inq_user_type(int ncid, nc_type xtype, char *name, size_t *size, nc_type *base_nc_typep, size_t *nfieldsp, int *classp)
    int nc_put_att(int ncid, int varid, const char *name, nc_type xtype, size_t len, const void *op)
    int nc_get_att(int ncid, int varid, const char *name, void *ip)
    int nc_def_enum(int ncid, nc_type base_typeid, const char *name, nc_type *typeidp)
    int nc_insert_enum(int ncid, nc_type xtype, const char *name, const void *value)
    int nc_inq_enum(int ncid, nc_type xtype, char *name, nc_type *base_nc_typep, size_t *base_sizep, size_t *num_membersp)
    int nc_inq_enum_member(int ncid, nc_type xtype, int idx, char *name, void *value)
    int nc_inq_enum_ident(int ncid, nc_type xtype, long long value, char *identifier)
    int nc_def_opaque(int ncid, size_t size, const char *name, nc_type *xtypep)
    int nc_inq_opaque(int ncid, nc_type xtype, char *name, size_t *sizep)
    int nc_put_var(int ncid, int varid,  const void *op)
    int nc_get_var(int ncid, int varid,  void *ip)
    int nc_put_var1(int ncid, int varid,  const size_t *indexp, const void *op)
    int nc_get_var1(int ncid, int varid,  const size_t *indexp, void *ip)
    int nc_put_vara(int ncid, int varid,  const size_t *startp, const size_t *countp, const void *op)
    int nc_get_vara(int ncid, int varid,  const size_t *startp, const size_t *countp, void *ip)
    int nc_put_vars(int ncid, int varid,  const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const void *op)
    int nc_get_vars(int ncid, int varid,  const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, void *ip)
    int nc_put_varm(int ncid, int varid,  const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const void *op)
    int nc_get_varm(int ncid, int varid,  const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, void *ip)
    int nc_def_var_deflate(int ncid, int varid, int shuffle, int deflate, int deflate_level)
    int nc_inq_var_deflate(int ncid, int varid, int *shufflep, int *deflatep, int *deflate_levelp)
    int nc_inq_var_szip(int ncid, int varid, int *options_maskp, int *pixels_per_blockp)
    int nc_def_var_fletcher32(int ncid, int varid, int fletcher32)
    int nc_inq_var_fletcher32(int ncid, int varid, int *fletcher32p)
    int nc_def_var_chunking(int ncid, int varid, int storage, const size_t *chunksizesp)
    int nc_inq_var_chunking(int ncid, int varid, int *storagep, size_t *chunksizesp)
    int nc_def_var_fill(int ncid, int varid, int no_fill, const void *fill_value)
    int nc_inq_var_fill(int ncid, int varid, int *no_fill, void *fill_valuep)
    int nc_def_var_endian(int ncid, int varid, int endian)
    int nc_inq_var_endian(int ncid, int varid, int *endianp)
    int nc_def_var_filter(int ncid, int varid, unsigned int id, size_t nparams, const unsigned int* parms)
    int nc_inq_var_filter(int ncid, int varid, unsigned int* idp, size_t* nparams, unsigned int* params)
    int nc_set_fill(int ncid, int fillmode, int *old_modep)
    int nc_set_default_format(int format, int *old_formatp)
    int nc_set_chunk_cache(size_t size, size_t nelems, float preemption)
    int nc_get_chunk_cache(size_t *sizep, size_t *nelemsp, float *preemptionp)
    int nc_set_var_chunk_cache(int ncid, int varid, size_t size, size_t nelems, float preemption)
    int nc_get_var_chunk_cache(int ncid, int varid, size_t *sizep, size_t *nelemsp, float *preemptionp)
    int nc_redef(int ncid)
    int nc__enddef(int ncid, size_t h_minfree, size_t v_align, size_t v_minfree, size_t r_align)
    int nc_enddef(int ncid)
    int nc_sync(int ncid)
    int nc_abort(int ncid)
    int nc_close(int ncid)
    int nc_inq(int ncid, int *ndimsp, int *nvarsp, int *nattsp, int *unlimdimidp)
    int nc_inq_ndims(int ncid, int *ndimsp)
    int nc_inq_nvars(int ncid, int *nvarsp)
    int nc_inq_natts(int ncid, int *nattsp)
    int nc_inq_unlimdim(int ncid, int *unlimdimidp)
    int nc_inq_unlimdims(int ncid, int *nunlimdimsp, int *unlimdimidsp)
    int nc_inq_format(int ncid, int *formatp)
    int nc_inq_format_extended(int ncid, int *formatp, int* modep)
    int nc_def_dim(int ncid, const char *name, size_t len, int *idp)
    int nc_inq_dimid(int ncid, const char *name, int *idp)
    int nc_inq_dim(int ncid, int dimid, char *name, size_t *lenp)
    int nc_inq_dimname(int ncid, int dimid, char *name)
    int nc_inq_dimlen(int ncid, int dimid, size_t *lenp)
    int nc_rename_dim(int ncid, int dimid, const char *name)
    int nc_inq_att(int ncid, int varid, const char *name, nc_type *xtypep, size_t *lenp)
    int nc_inq_attid(int ncid, int varid, const char *name, int *idp)
    int nc_inq_atttype(int ncid, int varid, const char *name, nc_type *xtypep)
    int nc_inq_attlen(int ncid, int varid, const char *name, size_t *lenp)
    int nc_inq_attname(int ncid, int varid, int attnum, char *name)
    int nc_copy_att(int ncid_in, int varid_in, const char *name, int ncid_out, int varid_out)
    int nc_rename_att(int ncid, int varid, const char *name, const char *newname)
    int nc_del_att(int ncid, int varid, const char *name)
    int nc_put_att_text(int ncid, int varid, const char *name, size_t len, const char *op)
    int nc_get_att_text(int ncid, int varid, const char *name, char *ip)
    int nc_put_att_string(int ncid, int varid, const char *name, size_t len, const char **op)
    int nc_get_att_string(int ncid, int varid, const char *name, char **ip)
    int nc_put_att_uchar(int ncid, int varid, const char *name, nc_type xtype, size_t len, const unsigned char *op)
    int nc_get_att_uchar(int ncid, int varid, const char *name, unsigned char *ip)
    int nc_put_att_schar(int ncid, int varid, const char *name, nc_type xtype, size_t len, const signed char *op)
    int nc_get_att_schar(int ncid, int varid, const char *name, signed char *ip)
    int nc_put_att_short(int ncid, int varid, const char *name, nc_type xtype, size_t len, const short *op)
    int nc_get_att_short(int ncid, int varid, const char *name, short *ip)
    int nc_put_att_int(int ncid, int varid, const char *name, nc_type xtype, size_t len, const int *op)
    int nc_get_att_int(int ncid, int varid, const char *name, int *ip)
    int nc_put_att_long(int ncid, int varid, const char *name, nc_type xtype, size_t len, const long *op)
    int nc_get_att_long(int ncid, int varid, const char *name, long *ip)
    int nc_put_att_float(int ncid, int varid, const char *name, nc_type xtype, size_t len, const float *op)
    int nc_get_att_float(int ncid, int varid, const char *name, float *ip)
    int nc_put_att_double(int ncid, int varid, const char *name, nc_type xtype, size_t len, const double *op)
    int nc_get_att_double(int ncid, int varid, const char *name, double *ip)
    int nc_put_att_ushort(int ncid, int varid, const char *name, nc_type xtype, size_t len, const unsigned short *op)
    int nc_get_att_ushort(int ncid, int varid, const char *name, unsigned short *ip)
    int nc_put_att_uint(int ncid, int varid, const char *name, nc_type xtype, size_t len, const unsigned int *op)
    int nc_get_att_uint(int ncid, int varid, const char *name, unsigned int *ip)
    int nc_put_att_longlong(int ncid, int varid, const char *name, nc_type xtype, size_t len, const long long *op)
    int nc_get_att_longlong(int ncid, int varid, const char *name, long long *ip)
    int nc_put_att_ulonglong(int ncid, int varid, const char *name, nc_type xtype, size_t len, const unsigned long long *op)
    int nc_get_att_ulonglong(int ncid, int varid, const char *name, unsigned long long *ip)
    int nc_def_var(int ncid, const char *name, nc_type xtype, int ndims, const int *dimidsp, int *varidp)
    int nc_inq_var(int ncid, int varid, char *name, nc_type *xtypep, int *ndimsp, int *dimidsp, int *nattsp)
    int nc_inq_varid(int ncid, const char *name, int *varidp)
    int nc_inq_varname(int ncid, int varid, char *name)
    int nc_inq_vartype(int ncid, int varid, nc_type *xtypep)
    int nc_inq_varndims(int ncid, int varid, int *ndimsp)
    int nc_inq_vardimid(int ncid, int varid, int *dimidsp)
    int nc_inq_varnatts(int ncid, int varid, int *nattsp)
    int nc_rename_var(int ncid, int varid, const char *name)
    int nc_copy_var(int ncid_in, int varid, int ncid_out)
    int nc_put_var1_text(int ncid, int varid, const size_t *indexp, const char *op)
    int nc_get_var1_text(int ncid, int varid, const size_t *indexp, char *ip)
    int nc_put_var1_uchar(int ncid, int varid, const size_t *indexp, const unsigned char *op)
    int nc_get_var1_uchar(int ncid, int varid, const size_t *indexp, unsigned char *ip)
    int nc_put_var1_schar(int ncid, int varid, const size_t *indexp, const signed char *op)
    int nc_get_var1_schar(int ncid, int varid, const size_t *indexp, signed char *ip)
    int nc_put_var1_short(int ncid, int varid, const size_t *indexp, const short *op)
    int nc_get_var1_short(int ncid, int varid, const size_t *indexp, short *ip)
    int nc_put_var1_int(int ncid, int varid, const size_t *indexp, const int *op)
    int nc_get_var1_int(int ncid, int varid, const size_t *indexp, int *ip)
    int nc_put_var1_long(int ncid, int varid, const size_t *indexp, const long *op)
    int nc_get_var1_long(int ncid, int varid, const size_t *indexp, long *ip)
    int nc_put_var1_float(int ncid, int varid, const size_t *indexp, const float *op)
    int nc_get_var1_float(int ncid, int varid, const size_t *indexp, float *ip)
    int nc_put_var1_double(int ncid, int varid, const size_t *indexp, const double *op)
    int nc_get_var1_double(int ncid, int varid, const size_t *indexp, double *ip)
    int nc_put_var1_ushort(int ncid, int varid, const size_t *indexp, const unsigned short *op)
    int nc_get_var1_ushort(int ncid, int varid, const size_t *indexp, unsigned short *ip)
    int nc_put_var1_uint(int ncid, int varid, const size_t *indexp, const unsigned int *op)
    int nc_get_var1_uint(int ncid, int varid, const size_t *indexp, unsigned int *ip)
    int nc_put_var1_longlong(int ncid, int varid, const size_t *indexp, const long long *op)
    int nc_get_var1_longlong(int ncid, int varid, const size_t *indexp, long long *ip)
    int nc_put_var1_ulonglong(int ncid, int varid, const size_t *indexp, const unsigned long long *op)
    int nc_get_var1_ulonglong(int ncid, int varid, const size_t *indexp, unsigned long long *ip)
    int nc_put_var1_string(int ncid, int varid, const size_t *indexp, const char **op)
    int nc_get_var1_string(int ncid, int varid, const size_t *indexp, char **ip)
    int nc_put_vara_text(int ncid, int varid, const size_t *startp, const size_t *countp, const char *op)
    int nc_get_vara_text(int ncid, int varid, const size_t *startp, const size_t *countp, char *ip)
    int nc_put_vara_uchar(int ncid, int varid, const size_t *startp, const size_t *countp, const unsigned char *op)
    int nc_get_vara_uchar(int ncid, int varid, const size_t *startp, const size_t *countp, unsigned char *ip)
    int nc_put_vara_schar(int ncid, int varid, const size_t *startp, const size_t *countp, const signed char *op)
    int nc_get_vara_schar(int ncid, int varid, const size_t *startp, const size_t *countp, signed char *ip)
    int nc_put_vara_short(int ncid, int varid, const size_t *startp, const size_t *countp, const short *op)
    int nc_get_vara_short(int ncid, int varid, const size_t *startp, const size_t *countp, short *ip)
    int nc_put_vara_int(int ncid, int varid, const size_t *startp, const size_t *countp, const int *op)
    int nc_get_vara_int(int ncid, int varid, const size_t *startp, const size_t *countp, int *ip)
    int nc_put_vara_long(int ncid, int varid, const size_t *startp, const size_t *countp, const long *op)
    int nc_get_vara_long(int ncid, int varid, const size_t *startp, const size_t *countp, long *ip)
    int nc_put_vara_float(int ncid, int varid, const size_t *startp, const size_t *countp, const float *op)
    int nc_get_vara_float(int ncid, int varid, const size_t *startp, const size_t *countp, float *ip)
    int nc_put_vara_double(int ncid, int varid, const size_t *startp, const size_t *countp, const double *op)
    int nc_get_vara_double(int ncid, int varid, const size_t *startp, const size_t *countp, double *ip)
    int nc_put_vara_ushort(int ncid, int varid, const size_t *startp, const size_t *countp, const unsigned short *op)
    int nc_get_vara_ushort(int ncid, int varid, const size_t *startp, const size_t *countp, unsigned short *ip)
    int nc_put_vara_uint(int ncid, int varid, const size_t *startp, const size_t *countp, const unsigned int *op)
    int nc_get_vara_uint(int ncid, int varid, const size_t *startp, const size_t *countp, unsigned int *ip)
    int nc_put_vara_longlong(int ncid, int varid, const size_t *startp, const size_t *countp, const long long *op)
    int nc_get_vara_longlong(int ncid, int varid, const size_t *startp, const size_t *countp, long long *ip)
    int nc_put_vara_ulonglong(int ncid, int varid, const size_t *startp, const size_t *countp, const unsigned long long *op)
    int nc_get_vara_ulonglong(int ncid, int varid, const size_t *startp, const size_t *countp, unsigned long long *ip)
    int nc_put_vara_string(int ncid, int varid, const size_t *startp, const size_t *countp, const char **op)
    int nc_get_vara_string(int ncid, int varid, const size_t *startp, const size_t *countp, char **ip)
    int nc_put_vars_text(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const char *op)
    int nc_get_vars_text(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, char *ip)
    int nc_put_vars_uchar(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const unsigned char *op)
    int nc_get_vars_uchar(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, unsigned char *ip)
    int nc_put_vars_schar(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const signed char *op)
    int nc_get_vars_schar(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, signed char *ip)
    int nc_put_vars_short(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const short *op)
    int nc_get_vars_short(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, short *ip)
    int nc_put_vars_int(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const int *op)
    int nc_get_vars_int(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, int *ip)
    int nc_put_vars_long(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const long *op)
    int nc_get_vars_long(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, long *ip)
    int nc_put_vars_float(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const float *op)
    int nc_get_vars_float(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, float *ip)
    int nc_put_vars_double(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const double *op)
    int nc_get_vars_double(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, double *ip)
    int nc_put_vars_ushort(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const unsigned short *op)
    int nc_get_vars_ushort(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, unsigned short *ip)
    int nc_put_vars_uint(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const unsigned int *op)
    int nc_get_vars_uint(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, unsigned int *ip)
    int nc_put_vars_longlong(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const long long *op)
    int nc_get_vars_longlong(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, long long *ip)
    int nc_put_vars_ulonglong(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const unsigned long long *op)
    int nc_get_vars_ulonglong(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, unsigned long long *ip)
    int nc_put_vars_string(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const char **op)
    int nc_get_vars_string(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, char **ip)
    int nc_put_varm_text(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const char *op)
    int nc_get_varm_text(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, char *ip)
    int nc_put_varm_uchar(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const unsigned char *op)
    int nc_get_varm_uchar(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, unsigned char *ip)
    int nc_put_varm_schar(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const signed char *op)
    int nc_get_varm_schar(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, signed char *ip)
    int nc_put_varm_short(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const short *op)
    int nc_get_varm_short(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, short *ip)
    int nc_put_varm_int(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const int *op)
    int nc_get_varm_int(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, int *ip)
    int nc_put_varm_long(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const long *op)
    int nc_get_varm_long(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, long *ip)
    int nc_put_varm_float(int ncid, int varid,const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const float *op)
    int nc_get_varm_float(int ncid, int varid,const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, float *ip)
    int nc_put_varm_double(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t *imapp, const double *op)
    int nc_get_varm_double(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, double *ip)
    int nc_put_varm_ushort(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, const unsigned short *op)
    int nc_get_varm_ushort(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, unsigned short *ip)
    int nc_put_varm_uint(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, const unsigned int *op)
    int nc_get_varm_uint(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, unsigned int *ip)
    int nc_put_varm_longlong(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, const long long *op)
    int nc_get_varm_longlong(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, long long *ip)
    int nc_put_varm_ulonglong(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, const unsigned long long *op)
    int nc_get_varm_ulonglong(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, unsigned long long *ip)
    int nc_put_varm_string(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, const char **op)
    int nc_get_varm_string(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, char **ip)
    int nc_put_var_text(int ncid, int varid, const char *op)
    int nc_get_var_text(int ncid, int varid, char *ip)
    int nc_put_var_uchar(int ncid, int varid, const unsigned char *op)
    int nc_get_var_uchar(int ncid, int varid, unsigned char *ip)
    int nc_put_var_schar(int ncid, int varid, const signed char *op)
    int nc_get_var_schar(int ncid, int varid, signed char *ip)
    int nc_put_var_short(int ncid, int varid, const short *op)
    int nc_get_var_short(int ncid, int varid, short *ip)
    int nc_put_var_int(int ncid, int varid, const int *op)
    int nc_get_var_int(int ncid, int varid, int *ip)
    int nc_put_var_long(int ncid, int varid, const long *op)
    int nc_get_var_long(int ncid, int varid, long *ip)
    int nc_put_var_float(int ncid, int varid, const float *op)
    int nc_get_var_float(int ncid, int varid, float *ip)
    int nc_put_var_double(int ncid, int varid, const double *op)
    int nc_get_var_double(int ncid, int varid, double *ip)
    int nc_put_var_ushort(int ncid, int varid, const unsigned short *op)
    int nc_get_var_ushort(int ncid, int varid, unsigned short *ip)
    int nc_put_var_uint(int ncid, int varid, const unsigned int *op)
    int nc_get_var_uint(int ncid, int varid, unsigned int *ip)
    int nc_put_var_longlong(int ncid, int varid, const long long *op)
    int nc_get_var_longlong(int ncid, int varid, long long *ip)
    int nc_put_var_ulonglong(int ncid, int varid, const unsigned long long *op)
    int nc_get_var_ulonglong(int ncid, int varid, unsigned long long *ip)
    int nc_put_var_string(int ncid, int varid, const char **op)
    int nc_get_var_string(int ncid, int varid, char **ip)
    int nc_put_att_ubyte(int ncid, int varid, const char *name, nc_type xtype, size_t len, const unsigned char *op)
    int nc_get_att_ubyte(int ncid, int varid, const char *name, unsigned char *ip)
    int nc_put_var1_ubyte(int ncid, int varid, const size_t *indexp, const unsigned char *op)
    int nc_get_var1_ubyte(int ncid, int varid, const size_t *indexp, unsigned char *ip)
    int nc_put_vara_ubyte(int ncid, int varid, const size_t *startp, const size_t *countp, const unsigned char *op)
    int nc_get_vara_ubyte(int ncid, int varid, const size_t *startp, const size_t *countp, unsigned char *ip)
    int nc_put_vars_ubyte(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const unsigned char *op)
    int nc_get_vars_ubyte(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, unsigned char *ip)
    int nc_put_varm_ubyte(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, const unsigned char *op)
    int nc_get_varm_ubyte(int ncid, int varid, const size_t *startp, const size_t *countp, const ptrdiff_t *stridep, const ptrdiff_t * imapp, unsigned char *ip)
    int nc_put_var_ubyte(int ncid, int varid, const unsigned char *op)
    int nc_get_var_ubyte(int ncid, int varid, unsigned char *ip)
    int nc_set_log_level(int new_level)
    int nc_show_metadata(int ncid)
    int nc__create_mp(const char *path, int cmode, size_t initialsz, int basepe, size_t *chunksizehintp, int *ncidp)
    int nc__open_mp(const char *path, int mode, int basepe, size_t *chunksizehintp, int *ncidp)
    int nc_delete(const char *path)
    int nc_delete_mp(const char *path, int basepe)
    int nc_set_base_pe(int ncid, int pe)
    int nc_inq_base_pe(int ncid, int *pe)
    int nctypelen(nc_type datatype)
    cdef enum:
        FILL_BYTE
        FILL_CHAR
        FILL_SHORT
        FILL_LONG
        FILL_FLOAT
        FILL_DOUBLE
        MAX_NC_DIMS
        MAX_NC_ATTRS
        MAX_NC_VARS
        MAX_NC_NAME
        MAX_VAR_DIMS
    cdef enum:
        NC_ENTOOL
        NC_EXDR
        NC_SYSERR
        NC_FATAL 
        NC_VERBOSE
    ctypedef int nclong
    int nccreate(const char* path, int cmode)
    int ncopen(const char* path, int mode)
    int ncsetfill(int ncid, int fillmode)
    int ncredef(int ncid)
    int ncendef(int ncid)
    int ncsync(int ncid)
    int ncabort(int ncid)
    int ncclose(int ncid)
    int ncinquire(int ncid, int *ndimsp, int *nvarsp, int *nattsp, int *unlimdimp)
    int ncdimdef(int ncid, const char *name, long len)
    int ncdimid(int ncid, const char *name)
    int ncdiminq(int ncid, int dimid, char *name, long *lenp)
    int ncdimrename(int ncid, int dimid, const char *name)
    int ncattput(int ncid, int varid, const char *name, nc_type xtype, int len, const void *op)
    int ncattinq(int ncid, int varid, const char *name, nc_type *xtypep, int *lenp)
    int ncattget(int ncid, int varid, const char *name, void *ip)
    int ncattcopy(int ncid_in, int varid_in, const char *name, int ncid_out, int varid_out)
    int ncattname(int ncid, int varid, int attnum, char *name)
    int ncattrename(int ncid, int varid, const char *name, const char *newname)
    int ncattdel(int ncid, int varid, const char *name)
    int ncvardef(int ncid, const char *name, nc_type xtype, int ndims, const int *dimidsp)
    int ncvarid(int ncid, const char *name)
    int ncvarinq(int ncid, int varid, char *name, nc_type *xtypep, int *ndimsp, int *dimidsp, int *nattsp)
    int ncvarput1(int ncid, int varid, const long *indexp, const void *op)
    int ncvarget1(int ncid, int varid, const long *indexp, void *ip)
    int ncvarput(int ncid, int varid, const long *startp, const long *countp, const void *op)
    int ncvarget(int ncid, int varid, const long *startp, const long *countp, void *ip)
    int ncvarputs(int ncid, int varid, const long *startp, const long *countp, const long *stridep, const void *op)
    int ncvargets(int ncid, int varid, const long *startp, const long *countp, const long *stridep, void *ip)
    int ncvarputg(int ncid, int varid, const long *startp, const long *countp, const long *stridep, const long *imapp, const void *op)
    int ncvargetg(int ncid, int varid, const long *startp, const long *countp, const long *stridep, const long *imapp, void *ip)
    int ncvarrename(int ncid, int varid, const char *name)
    int ncrecinq(int ncid, int *nrecvarsp, int *recvaridsp, long *recsizesp)
    int ncrecget(int ncid, long recnum, void **datap)
    int ncrecput(int ncid, long recnum, void *const *datap)
    int nc_finalize()