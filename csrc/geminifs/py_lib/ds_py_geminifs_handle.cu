#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>

#include "ds_py_geminifs_handle.cuh"
#include "gpu_file_manager.cuh"

#define GEMINIFS_MIN_PAGE 4096

deepspeed_geminifs_handle_t::deepspeed_geminifs_handle_t(
    const std::string &config_path,
    const uint64_t file_size,
    const int block_size, 
    const int nr_files)
    : block_size_(block_size > GEMINIFS_MIN_PAGE ? block_size : GEMINIFS_MIN_PAGE), nr_files_(nr_files), file_size_(file_size){
    _init_geminifs_controller(config_path);
}

deepspeed_geminifs_handle_t::~deepspeed_geminifs_handle_t() {
    _close_geminifs_controller();
}
int deepspeed_geminifs_handle_t::get_block_size() {
    return block_size_;
}
 
int deepspeed_geminifs_handle_t::get_alignment() {
    return GEMINIFS_MIN_PAGE;
}

void deepspeed_geminifs_handle_t::_init_geminifs_controller(const std::string &config_path) {
    if (!geminifs_initialized_) {
        std::lock_guard<std::mutex> lock(geminifs_mutex_);
        std::filesystem::path config_file_path(config_path);
        auto file_size = {1ul, 1ul, static_cast<size_t>(file_size_)};
        // the config determines devices to use
        geminifs_instance_ = std::make_shared<GeminiFS>(config_file_path, nr_files_, file_size, true);
        geminifs_initialized_ = true;
        std::cout << "Initialized GeminiFS controller with config: " << config_file_path << std::endl;
    } else {
        std::cout << "GeminiFS controller is already initialized." << std::endl;
    }
}

void deepspeed_geminifs_handle_t::_close_geminifs_controller() {
    std::lock_guard<std::mutex> lock(geminifs_mutex_);

    if (geminifs_initialized_) {
        geminifs_initialized_ = false;
        geminifs_instance_.reset();
    }
}

torch::Tensor deepspeed_geminifs_handle_t::new_pinned_device_tensor(const size_t num_elem, 
                                                                    const torch::Tensor& example_tensor) {
    if (!geminifs_initialized_) {
        throw std::runtime_error("GeminiFS controller is not initialized.");
    }

    auto dev_tensor = torch::empty_like(example_tensor);
    if (!geminifs_instance_->geminifs_register_tensor_with_gpu(dev_tensor, get_block_size())) {
        throw std::runtime_error("Failed to register tensor with GeminiFS.");
    } 
    return dev_tensor;
}

bool deepspeed_geminifs_handle_t::free_pinned_device_tensor(torch::Tensor& tensor) {
    if (!geminifs_initialized_) {
        throw std::runtime_error("GeminiFS controller is not initialized.");
    }
    return geminifs_instance_->geminifs_unregister_tensor_from_gpu(tensor);
}

bool deepspeed_geminifs_handle_t::pin_device_tensor(const torch::Tensor& buffer) {
    if (!geminifs_initialized_) {
        throw std::runtime_error("GeminiFS controller is not initialized.");
    }
    return geminifs_instance_->geminifs_register_tensor_with_gpu(buffer, get_block_size());
}

bool deepspeed_geminifs_handle_t::unpin_device_tensor(const torch::Tensor& buffer) {
    if (!geminifs_initialized_) {
        throw std::runtime_error("GeminiFS controller is not initialized.");
    }

    return geminifs_instance_->geminifs_unregister_tensor_from_gpu(buffer);
}

// read / write implementations
bool deepspeed_geminifs_handle_t::read(torch::Tensor& buffer,
                                      const GPUFileId gpu_file_id,
                                      const bool validate, 
                                      const int64_t file_offset,
                                      const int64_t stream_id) {
    if (!geminifs_initialized_) {
        std::cerr << "GeminiFS controller is not initialized." << std::endl;
        return false;
    }
    const auto device_id = buffer.get_device();
    if (device_id < 0) {
        std::cerr << "Buffer must be a GPU tensor for GeminiFS read." << std::endl;
        return false;
    }

    const auto stream = reinterpret_cast<cudaStream_t>(stream_id);
    const auto gpu_controller = geminifs_instance_->geminifs_get_gpu_controller(device_id);
    if (gpu_controller == nullptr) {
        std::cerr << "Failed to get GPU controller for device " << device_id << std::endl;
        return false;
    }
    
    return geminifs_instance_->geminifs_xfer_kernel_one_tensor(buffer, gpu_file_id, file_offset, block_size_, gpu_controller, true, stream);
}

bool deepspeed_geminifs_handle_t::write(const torch::Tensor& buffer,
                                       const GPUFileId gpu_file_id,
                                       const bool validate, 
                                       const int64_t file_offset, 
                                       const int64_t stream_id) {
    if (!geminifs_initialized_) {
        std::cerr << "GeminiFS controller is not initialized." << std::endl;
        exit(EXIT_FAILURE);
    }
    const auto device_id = buffer.get_device();
    if (device_id < 0) {
        std::cerr << "Buffer must be a GPU tensor for GeminiFS write." << std::endl;
        return false;
    }

    const auto stream = reinterpret_cast<cudaStream_t>(stream_id);
    const auto gpu_controller = geminifs_instance_->geminifs_get_gpu_controller(device_id);
    if (gpu_controller == nullptr) {
        std::cerr << "Failed to get GPU controller for device " << device_id << std::endl;
        return false;
    }
    
    return geminifs_instance_->geminifs_xfer_kernel_one_tensor(buffer, gpu_file_id, file_offset, block_size_, gpu_controller, false, stream);
}

std::tuple<bool, GPUFileId> deepspeed_geminifs_handle_t::get_geminifs_gpu_file(const int device_id) {
    GPUFileId gpu_file_id;
    bool success = geminifs_instance_->geminifs_gpu_open_file(device_id, gpu_file_id);
    return std::make_tuple(success, gpu_file_id);
}

void deepspeed_geminifs_handle_t::put_geminifs_gpu_file(const int device_id, const GPUFileId gpu_file_id) {
    geminifs_instance_->geminifs_gpu_close_file(device_id, gpu_file_id);
}