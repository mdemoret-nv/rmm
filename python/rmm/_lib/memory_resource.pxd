# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libc.stdint cimport int8_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.memory cimport unique_ptr
from libcpp.map cimport map

cdef extern from "<iostream>" namespace "std":
    cdef cppclass basic_istream[T]:
        pass

    cdef cppclass basic_ostream[T]:
        pass

    ctypedef basic_istream[char] istream

    ctypedef basic_ostream[char] ostream

cdef extern from "<sstream>" namespace "std":
    cdef cppclass basic_ostringstream[T](basic_ostream[T]):
        basic_ostringstream()

        string str()

    ctypedef basic_ostringstream[char] ostringstream

cdef extern from "rmm/detail/stack_trace.hpp" namespace "rmm::detail" nogil:
    cdef cppclass stack_trace:
        stack_trace()

    ostream& operator<<(ostream& os, const stack_trace& st) except +

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

    cdef cppclass tracking_resource_adaptor_wrapper(
        device_memory_resource_wrapper
    ):
        struct allocation_counts:
            allocation_counts()

            ssize_t current_bytes
            ssize_t current_count
            ssize_t peak_bytes
            ssize_t peak_count
            size_t total_bytes
            size_t total_count

        struct allocation_info:
            allocation_info(size_t size, bool capture_stack)

            unique_ptr[stack_trace] strace
            size_t allocation_size

            string get_strace_str() const

        tracking_resource_adaptor_wrapper(
            shared_ptr[device_memory_resource_wrapper] upstream_mr
        ) except +

        void reset_allocation_counts() except +
        allocation_counts get_allocation_counts() except +
        map[void*, allocation_info] get_outstanding_allocations()


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

cdef class AllocInfo:
    cdef public int allocation_size
    cdef public string stack_str

cdef class TrackingMemoryResource(MemoryResource):
    pass
