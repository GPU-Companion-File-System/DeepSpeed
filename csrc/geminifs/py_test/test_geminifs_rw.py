from deepspeed.ops.op_builder import GeminiFSBuilder
import torch
import time
import faulthandler
faulthandler.enable()


def check_equal(t1, t2):
    """GPU 上检查是否一致"""
    if not torch.allclose(t1, t2, atol=1e-3, rtol=1e-3):
        diff = (t1 - t2).abs()
        print("❌ Mismatch detected! max diff =", diff.max().item())
        return False
    return True


if __name__ == "__main__":
    # -----------------------
    # 1. 初始化 GeminiFS
    # -----------------------
    block_size = 64 * 1024
    nr_files = 64
    config_path = "/home/yjq/LoRA-DeepSpeed/Geminifs/sys_config.ini"

    handle = GeminiFSBuilder().load().geminifs_handle(config_path, 64*1024*1024, block_size, nr_files)
    print("Geminifs initialization test passed.")

    # -----------------------
    # 2. CUDA 设置
    # -----------------------
    device_id = 0
    torch.cuda.set_device(device_id)

    nr_xfer_files = 32

    # -----------------------
    # 3. 获取 GeminiFS GPU 文件句柄
    # -----------------------
    gpu_files = []
    for _ in range(nr_xfer_files):
        ok, f = handle.get_geminifs_gpu_file(device_id)
        assert ok, "Failed to get GeminiFS GPU file"
        gpu_files.append(f)

    print("Allocated GPU files:", len(gpu_files))

    # -----------------------
    # 4. 构建要写入的数据
    # -----------------------
    write_tensors = [
        torch.ones(128*1024*1024, dtype=torch.int8, device="cuda")
        for _ in range(nr_xfer_files)
    ]

    for i, t in enumerate(write_tensors):
        assert handle.pin_device_tensor(t), f"Failed to pin tensor {i}"

    # -----------------------
    # 5. 执行写入
    # -----------------------
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    print("Starting GeminiFS write test...")
    t0 = time.time()

    for i in range(nr_xfer_files):
        gpu_file = gpu_files[i]
        tensor = write_tensors[i]
        offset = 0
        validate = False
        print(f"write {gpu_file}")

        ok = handle.write(tensor, gpu_file, validate, offset, stream.cuda_stream)
        assert ok, f"Failed to write file {i}"

    stream.synchronize()
    t1 = time.time()

    bw_write = block_size * nr_xfer_files / (t1 - t0) / (1024**2)
    print(f"Write time: {t1 - t0:.4f}s, BW: {bw_write:.2f} MB/s")
    input("stop")

    # -----------------------
    # 6. 读回验证
    # -----------------------
    print("Starting GeminiFS read test...")

    read_tensors = [
        torch.empty(block_size, dtype=torch.int8, device="cuda")
        for _ in range(nr_xfer_files)
    ]

    for i, t in enumerate(read_tensors):
        assert handle.pin_device_tensor(t), f"Failed to pin read tensor {i}"

    t2 = time.time()
    for i in range(nr_xfer_files):
        gpu_file = gpu_files[i]
        out_tensor = read_tensors[i]
        offset = 0

        ok = handle.read(out_tensor, gpu_file, offset, False, stream.cuda_stream)
        assert ok, f"Failed to read file {i}"

    stream.synchronize()
    t3 = time.time()

    bw_read = block_size * nr_xfer_files / (t3 - t2) / (1024**2)
    print(f"Read time: {t3 - t2:.4f}s, BW: {bw_read:.2f} MB/s")


    # -----------------------
    # 7. 校验数据是否一致
    # -----------------------
    print("Validating data correctness...")

    all_ok = True
    for i in range(nr_xfer_files):
        if not check_equal(write_tensors[i], read_tensors[i]):
            print(f"❌ Data mismatch for file index {i}")
            all_ok = False
            break

    if all_ok:
        print("✅ All files validated correctly! GeminiFS read/write matches.")


    # -----------------------
    # 8. 清理资源
    # -----------------------
    print("Releasing GPU files...")
    for f in gpu_files:
        handle.put_geminifs_gpu_file(device_id, f)

    print("Closing GeminiFS...")
    handle.close_geminifs()
    torch.cuda.synchronize()

    print("Done.")
