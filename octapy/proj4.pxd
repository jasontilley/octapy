cdef extern from "proj_api.h" nogil:
    cdef enum:
        PJ_VERSION
        PJ_LOCALE_SAFE
    cdef double RAD_TO_DEG = 57.295779513082321
    cdef double DEG_TO_RAD = .017453292519943296
    cdef const char *pj_release
    cdef int  pj_errno
    ctypedef struct projUV:
        double u
        double v
    ctypedef struct projUVW:
        double u
        double v
        double w
    ctypedef void *projPJ
    ctypedef projUV projXY
    ctypedef projUV projLP
    ctypedef projUVW projXYZ
    ctypedef projUVW projLPZ
    ctypedef void *projCtx

    ctypedef int *PAFile
    ctypedef struct projFileAPI:
        PAFile  (*FOpen)(projCtx ctx, const char *filename, const char *access)
        size_t  (*FRead)(void *buffer, size_t size, size_t nmemb, PAFile file)
        int     (*FSeek)(PAFile file, long offset, int whence)
        long    (*FTell)(PAFile file)
        void    (*FClose)(PAFile)

    projXY pj_fwd(projLP, projPJ)
    projLP pj_inv(projXY, projPJ)
    projXYZ pj_fwd3d(projLPZ, projPJ)
    projLPZ pj_inv3d(projXYZ, projPJ)
    int pj_transform( projPJ src, projPJ dst, long point_count, int point_offset,
                    double *x, double *y, double *z )
    int pj_datum_transform( projPJ src, projPJ dst, long point_count, int point_offset,
                            double *x, double *y, double *z )
    int pj_geocentric_to_geodetic( double a, double es,
                                long point_count, int point_offset,
                                double *x, double *y, double *z )
    int pj_geodetic_to_geocentric( double a, double es,
                                long point_count, int point_offset,
                                double *x, double *y, double *z )
    int pj_compare_datums( projPJ srcdefn, projPJ dstdefn )
    int pj_apply_gridshift( projCtx, const char *, int,
                            long point_count, int point_offset,
                            double *x, double *y, double *z )
    void pj_deallocate_grids()
    void pj_clear_initcache()
    int pj_is_latlong(projPJ)
    int pj_is_geocent(projPJ)
    void pj_get_spheroid_defn(projPJ defn, double *major_axis, double *eccentricity_squared)
    void pj_pr_list(projPJ)
    void pj_free(projPJ)
    void pj_set_finder( const char *(*)(const char *) )
    void pj_set_searchpath ( int count, const char **path )
    projPJ pj_init(int, char **)
    projPJ pj_init_plus(const char *)
    projPJ pj_init_ctx( projCtx, int, char ** )
    projPJ pj_init_plus_ctx( projCtx, const char * )
    char *pj_get_def(projPJ, int)
    projPJ pj_latlong_from_proj( projPJ )
    void *pj_malloc(size_t)
    void pj_dalloc(void *)
    void *pj_calloc (size_t n, size_t size)
    void *pj_dealloc (void *ptr)
    char *pj_strerrno(int)
    int *pj_get_errno_ref()
    const char *pj_get_release()
    void pj_acquire_lock()
    void pj_release_lock()
    void pj_cleanup_lock()

    projCtx pj_get_default_ctx()
    projCtx pj_get_ctx( projPJ )
    void pj_set_ctx( projPJ, projCtx )
    projCtx pj_ctx_alloc()
    void    pj_ctx_free( projCtx )
    int pj_ctx_get_errno( projCtx )
    void pj_ctx_set_errno( projCtx, int )
    void pj_ctx_set_debug( projCtx, int )
    void pj_ctx_set_logger( projCtx, void (*)(void *, int, const char *) )
    void pj_ctx_set_app_data( projCtx, void * )
    void *pj_ctx_get_app_data( projCtx )
    void pj_ctx_set_fileapi( projCtx, projFileAPI *)
    projFileAPI *pj_ctx_get_fileapi( projCtx )

    void pj_log( projCtx ctx, int level, const char *fmt, ...)
    void pj_stderr_logger( void *, int, const char * )

    projFileAPI *pj_get_default_fileapi()
    PAFile pj_ctx_fopen(projCtx ctx, const char *filename, const char *access)
    size_t pj_ctx_fread(projCtx ctx, void *buffer, size_t size, size_t nmemb, PAFile file)
    int    pj_ctx_fseek(projCtx ctx, PAFile file, long offset, int whence)
    long   pj_ctx_ftell(projCtx ctx, PAFile file)
    void   pj_ctx_fclose(projCtx ctx, PAFile file)
    char  *pj_ctx_fgets(projCtx ctx, char *line, int size, PAFile file)
    PAFile pj_open_lib(projCtx, const char *, const char *)
    int pj_run_selftests (int verbosity)

    cdef enum:
        PJ_LOG_NONE
        PJ_LOG_ERROR
        PJ_LOG_DEBUG_MAJOR
        PJ_LOG_DEBUG_MINOR