#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include "geminifs.cuh"
#include "gpu_file_manager.cuh"

struct deepspeed_geminifs_handle_t {
    deepspeed_geminifs_handle_t(const std::string &config_path,
                                const int block_size,
                                const int nr_files);
    
    ~deepspeed_geminifs_handle_t();

    int get_block_size();
    int get_alignment();

    int read(torch::Tensor& buffer,
             const GPUFileId gpu_file_id,
             const bool validate,
             const int64_t file_offset, 
             const int64_t stream_id);

    int write(const torch::Tensor& buffer,
              const GPUFileId gpu_file_id,
              const bool validate,
              const int64_t file_offset,
              const int64_t stream_id);

    std::tuple<bool, GPUFileId> get_geminifs_gpu_file(const int device_id);

    void put_geminifs_gpu_file(const int device_id, const GPUFileId gpu_file_id);

    torch::Tensor new_pinned_device_tensor(const size_t num_elem,
                                           const torch::Tensor& example_tensor);
    

    bool free_pinned_device_tensor(torch::Tensor&);

    bool pin_device_tensor(const torch::Tensor& buffer);

    bool unpin_device_tensor(const torch::Tensor& buffer);

    void _init_geminifs_controller(const std::string &config_path);

    void _close_geminifs_controller();
    


private:
    const int block_size_;
    const int nr_files_;

    std::atomic<bool> geminifs_initialized_{false};
    std::shared_ptr<GeminiFS> geminifs_instance_;
    std::mutex geminifs_mutex_;
};