#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "geminifs.cuh"
#include "ds_py_geminifs_handle.cuh"

// Wrapper function for geminifs_gpu_open_file to handle reference parameter
using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<deepspeed_geminifs_handle_t>(m, "geminifs_handle")
        .def(py::init<const std::string&, const uint64_t, const int, const int>(),
            "Geminifs handle constructor",
            "config_path"_a,
            "file_size"_a,
            "block_size"_a = 64 * 1024, // for geminifs
            "nr_files"_a = 1024)       // for geminifs

        .def("get_block_size", &deepspeed_geminifs_handle_t::get_block_size)
        .def("get_alignment", &deepspeed_geminifs_handle_t::get_alignment)
        .def("get_file_size", &deepspeed_geminifs_handle_t::get_file_size)

        .def("read",
            &deepspeed_geminifs_handle_t::read,
            "Synchronous and non-parallel file read. Returns count of completed read ops",
            "buffer"_a,
            "gpu_file_id"_a,
            "validate"_a,
            "file_offset"_a = 0,
            "stream_id"_a = 0)

        .def("write",
            &deepspeed_geminifs_handle_t::write,
            "Synchronous and non-parallel file write. Returns count of completed write ops",
            "buffer"_a,
            "gpu_file_id"_a,
            "validate"_a,
            "file_offset"_a = 0,
            "stream_id"_a = 0)

        .def("new_pinned_device_tensor",
            &deepspeed_geminifs_handle_t::new_pinned_device_tensor,
            "Allocate pinned device tensor.",
            "num_elem"_a,
            "example_tensor"_a)

        .def("free_pinned_device_tensor",
            &deepspeed_geminifs_handle_t::free_pinned_device_tensor,
            "Free pinned device tensor.",
            "tensor"_a)

        .def("pin_device_tensor",
            &deepspeed_geminifs_handle_t::pin_device_tensor,
            "Pin device tensor.",
            "tensor"_a)

        .def("unpin_device_tensor",
            &deepspeed_geminifs_handle_t::unpin_device_tensor,
            "Unpin device tensor.",
            "tensor"_a)
        
        // file management
        .def("get_geminifs_gpu_file",
            &deepspeed_geminifs_handle_t::get_geminifs_gpu_file,
            "Get GeminiFS GPU file handle.",
            "device_id"_a)
        
        .def("put_geminifs_gpu_file",
            &deepspeed_geminifs_handle_t::put_geminifs_gpu_file,
            "Release GeminiFS GPU file handle.",
            "device_id"_a,
            "gpu_file_id"_a)
        .def("close_geminifs", &deepspeed_geminifs_handle_t::_close_geminifs_controller,
            "Close GeminiFS controller");
}