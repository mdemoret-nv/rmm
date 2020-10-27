# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libc.stdint cimport int8_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.functional cimport function
from rmm._lib.lib cimport cudaStream_t

cdef extern from "memory_resource_wrappers.hpp" nogil:
    cdef cppclass device_memory_resource_wrapper:
        shared_ptr[device_memory_resource_wrapper] get_mr() except +

    cdef cppclass default_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        default_memory_resource_wrapper(int device) except +

    cdef cppclass cuda_memory_resource_wrapper(device_memory_resource_wrapper):
        cuda_memory_resource_wrapper() except +

    cdef cppclass managed_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        managed_memory_resource_wrapper() except +

    cdef cppclass pool_memory_resource_wrapper(device_memory_resource_wrapper):
        pool_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            size_t initial_pool_size,
            size_t maximum_pool_size
        ) except +

    cdef cppclass fixed_size_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        fixed_size_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            size_t block_size,
            size_t blocks_to_preallocate
        ) except +

    cdef cppclass binning_memory_resource_wrapper(
        device_memory_resource_wrapper
    ):
        binning_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr
        ) except +
        binning_memory_resource_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            int8_t min_size_exponent,
            int8_t max_size_exponent
        ) except +
        void add_bin(
            size_t allocation_size,
            shared_ptr[device_memory_resource_wrapper] bin_mr
        ) except +
        void add_bin(
            size_t allocation_size
        ) except +

    cdef cppclass logging_resource_adaptor_wrapper(
        device_memory_resource_wrapper
    ):
        logging_resource_adaptor_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            string filename
        ) except +
        void flush() except +

    cdef cppclass callback_resource_adaptor_wrapper(
        device_memory_resource_wrapper
    ):
        ctypedef function[void(bool, void*, size_t, cudaStream_t)] c_callback
        ctypedef void (*cy_callback) (void*, bool, void*, size_t, cudaStream_t)

        @staticmethod
        c_callback make_std_function(cy_callback, void*)

        callback_resource_adaptor_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
            c_callback callback
        ) except +

    cdef cppclass thread_safe_resource_adaptor_wrapper(
        device_memory_resource_wrapper
    ):
        thread_safe_resource_adaptor_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr,
        ) except +

    void set_per_device_resource(
        int device,
        shared_ptr[device_memory_resource_wrapper] new_resource
    ) except +


cdef class MemoryResource:
    cdef shared_ptr[device_memory_resource_wrapper] c_obj

cdef class CudaMemoryResource(MemoryResource):
    pass

cdef class ManagedMemoryResource(MemoryResource):
    pass

cdef class PoolMemoryResource(MemoryResource):
    pass

cdef class FixedSizeMemoryResource(MemoryResource):
    pass

cdef class BinningMemoryResource(MemoryResource):
    cpdef add_bin(self, size_t allocation_size, object bin_resource=*)

cdef class LoggingResourceAdaptor(MemoryResource):
    cdef object _log_file_name
    cpdef get_file_name(self)
    cpdef flush(self)

cdef class CallbackResourceAdaptor(MemoryResource):
    cdef object _py_callback
    cdef void _cy_callback(self, bool isAlloc, void* p, size_t bytes, cudaStream_t stream) with gil

    cpdef set_callback(self, object py_callback)
