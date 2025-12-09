# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import torch
from deepspeed.utils.logging import logger
from deepspeed.accelerator import get_accelerator

from deepspeed import comm as dist

MIN_AIO_BYTES = 1024**2
AIO_ALIGNED_BYTES = 1024
MIN_SWAPPABLE_BYTES = MIN_AIO_BYTES


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pread(buffer, path, 0) == 0)


def swap_out_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pwrite(buffer, path, 0) == 0)


def print_object(obj, name, exclude_list=[]):
    logger.info('{}:'.format(name))
    for arg in sorted(vars(obj)):
        if arg not in exclude_list:
            dots = '.' * (29 - len(arg))
            logger.info('  {} {} {}'.format(arg, dots, getattr(obj, arg)))


class SwapBuffer(object):

    def __init__(self, buffer):
        self.buffer = buffer
        self.reset()

    def reset(self):
        self.offset = 0
        self.swap_tensors = {}
        self.compute_tensors = {}
        self.swap_paths = {}
        self.num_elem = 0

    def insert_tensor(self, tensor, swap_path, aligned_numel):
        swap_tensor, compute_tensor = self.allocate_tensor(swap_path, tensor.numel(), aligned_numel)
        compute_tensor.data.copy_(tensor.data)
        return swap_tensor, compute_tensor

    def allocate_tensor(self, swap_path, numel, aligned_numel):
        assert self.has_space(aligned_numel)
        assert self.offset not in self.swap_tensors

        allocate_offset = self.offset
        swap_tensor = self.buffer.narrow(0, allocate_offset, aligned_numel)
        dest_tensor = swap_tensor.narrow(0, 0, numel)

        self.swap_tensors[allocate_offset] = swap_tensor
        self.compute_tensors[allocate_offset] = dest_tensor
        self.swap_paths[allocate_offset] = swap_path
        self.offset += aligned_numel
        self.num_elem += numel

        return self.swap_tensors[allocate_offset], self.compute_tensors[allocate_offset]

    def has_space(self, numel):
        return (self.offset + numel) <= self.buffer.numel()

    def get_swap_tensors(self):
        return [tensor for tensor in self.swap_tensors.values()]

    def get_swap_paths(self):
        return [path for path in self.swap_paths.values()]

    def get_compute_tensors(self):
        return [tensor for tensor in self.compute_tensors.values()]

    def get_num_elem(self):
        return self.num_elem

    def get_swap_tensor(self, offset):
        return self.swap_tensors.get(offset, None)

    def get_compute_tensor(self, offset):
        return self.compute_tensors.get(offset, None)

    def get_swap_path(self, offset):
        return self.swap_paths(offset, None)


class SwapBufferPool(object):

    def __init__(self, buffers, use_geminifs=False):
        if not use_geminifs:
            assert all([get_accelerator().is_pinned(buf) for buf in buffers])
        self.buffers = [SwapBuffer(buf) for buf in buffers]
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        for buffer in self.buffers:
            buffer.reset()

    def allocate_tensor(self, numel, swap_path, aligned_numel):
        if self.has_space(aligned_numel):
            swap_tensor, compute_tensor = self._get_current_buffer().allocate_tensor(swap_path, numel, aligned_numel)
            return swap_tensor, compute_tensor

        return None, None

    def insert_tensor(self, tensor, swap_path, aligned_numel):
        if self.has_space(aligned_numel):
            swap_tensor, compute_tensor = self._get_current_buffer().insert_tensor(tensor, swap_path, aligned_numel)
            return swap_tensor, compute_tensor

        return None, None

    def get_swap_tensors(self):
        swap_tensors = []
        for buffer in self._get_used_buffers():
            swap_tensors += buffer.get_swap_tensors()

        return swap_tensors

    def get_swap_paths(self):
        swap_paths = []
        for buffer in self._get_used_buffers():
            swap_paths += buffer.get_swap_paths()

        return swap_paths

    def get_compute_tensors(self):
        compute_tensors = []
        for buffer in self._get_used_buffers():
            compute_tensors += buffer.get_compute_tensors()

        return compute_tensors

    def has_space(self, numel):
        if self._get_current_buffer().has_space(numel):
            return True

        if self.current_index == len(self.buffers) - 1:
            return False

        self.current_index += 1
        return self._get_current_buffer().has_space(numel)

    def swap_out(self, aio_handle, async_op=False):
        swap_tensors = self.get_swap_tensors()
        swap_paths = self.get_swap_paths()
        assert all([p is not None for p in swap_paths])

        swap_out_tensors(aio_handle, swap_tensors, swap_paths)

        if not async_op:
            assert len(swap_tensors) == aio_handle.wait()

    def swap_in(self, aio_handle, async_op=False):
        swap_tensors = self.get_swap_tensors()
        swap_paths = self.get_swap_paths()
        assert all([p is not None for p in swap_paths])

        swap_in_tensors(aio_handle, swap_tensors, swap_paths)

        if not async_op:
            assert len(swap_tensors) == aio_handle.wait()

    def _get_current_buffer(self):
        return self.buffers[self.current_index]

    def _get_used_buffers(self):
        return self.buffers[:self.current_index + 1]


class SwapBufferManager(object):

    def __init__(self, num_elems, count, dtype):
        self.num_elems = num_elems
        self.count = count
        self.dtype = dtype
        self.all_buffers = [
            get_accelerator().pin_memory(torch.zeros(num_elems, device='cpu', dtype=dtype), align_bytes=0)
            for _ in range(count)
        ]
        self.free_buffer_index = [i for i in range(count)]
        self.used_buffer_index = {}
        self.gigabytes = (self.all_buffers[0].element_size() * num_elems * count) / (1024**3)

        if dist.get_rank() == 0:
            exclude_list = ['all_buffers']
            print_object(obj=self, name='SwapBufferManager', exclude_list=exclude_list)

    def allocate(self, num_elems, count, dtype):
        assert dtype == self.dtype
        assert num_elems <= self.num_elems
        if count > len(self.free_buffer_index):
            return None

        used_indices = self.free_buffer_index[-count:]
        self.free_buffer_index = self.free_buffer_index[:-count]

        buffers = []
        for i in used_indices:
            tmp_buffer = self.all_buffers[i].narrow(0, 0, num_elems)
            buffers.append(tmp_buffer)
            self.used_buffer_index[id(tmp_buffer)] = i
        return buffers

    def allocate_all(self, num_elems, dtype):
        return self.allocate(num_elems=num_elems, count=len(self.free_buffer_index), dtype=dtype)

    def free(self, buffers):
        buffer_ids = []
        for buf in buffers:
            buffer_ids.append(id(buf))

        assert all([b_id in self.used_buffer_index for b_id in buffer_ids])

        for b_id in buffer_ids:
            self.free_buffer_index.append(self.used_buffer_index[b_id])
            del (self.used_buffer_index[b_id])


def get_sized_buffer(buffer, num_elems):
    assert num_elems <= buffer.numel(), \
        f'num_elems {num_elems} > buffer {buffer.numel()}'
    return buffer.narrow(0, 0, num_elems) if num_elems < buffer.numel() else buffer


def get_sized_buffers(buffer_list, num_elems_list):
    swap_buffers = [
        get_sized_buffer(buffer, num_elems) \
        for buffer, num_elems in zip(buffer_list, num_elems_list)
    ]
    return swap_buffers


class SwapSliceBufferManager(object):

    class SliceBuffer:
        def __init__(self, buffer, offset, length, slice_id):
            self.buffer = buffer
            self.offset = offset
            self.length = length
            self.slice_id = slice_id

        @property
        def tensor(self):
            return self.buffer[self.offset:self.offset + self.length]

        def __repr__(self):
            return f"SliceBuffer(id={self.slice_id}, offset={self.offset}, length={self.length})"


    def __init__(self, buffer, aligned_size=64 * 1024, verbose=False):
        self.buffer = buffer.flatten()
        assert self.buffer.is_contiguous()
        self.element_size = self.buffer.element_size()

        self.free_list = [(0, self.buffer.numel())]
        self.aligned_size = aligned_size
        self.verbose = verbose         

        self.slice_map = {}
        self.next_id = 1
        self.free_ids = []
        self.min_free_bytes = self.buffer.numel()


    def _aligned_len(self, length_elements):
        size_bytes = length_elements * self.element_size
        aligned_bytes = ((size_bytes + self.aligned_size - 1) //
                         self.aligned_size) * self.aligned_size
        return aligned_bytes // self.element_size

    def _allocate(self, length, try_to_merge=True):

        aligned_len = self._aligned_len(length)
        for i, (offset, free_len) in enumerate(self.free_list):
            if free_len >= aligned_len:
                if self.free_ids:
                    slice_id = self.free_ids.pop()
                else:
                    slice_id = self.next_id
                    self.next_id += 1

                slice_buffer = self.SliceBuffer(
                    self.buffer, offset, aligned_len, slice_id
                )

                self.slice_map[slice_id] = slice_buffer

                if free_len == aligned_len:
                    self.free_list.pop(i)
                else:
                    self.free_list[i] = (offset + aligned_len,
                                         free_len - aligned_len)
                if self.verbose:
                    total_free = self.get_total_free_bytes()
                    if self.min_free_bytes > total_free:
                        self.min_free_bytes = total_free
                        logger.warning(f"Allocated {slice_buffer}, "
                                       f"min free bytes so far: {self.min_free_bytes * self.element_size / 1024 ** 3} GB")

                return slice_buffer

        total_free = sum(l for _, l in self.free_list)
        if total_free >= aligned_len and try_to_merge:
            self.try_to_merge_free_list()
            return self._allocate(length, try_to_merge=False)

        return None
    
    def get_buffer_id(self, length):
        slice_buffer = self._allocate(length)
        if slice_buffer is None:
            return None
        return slice_buffer.slice_id

    def _free(self, slice_buffer):
        self.free_list.append((slice_buffer.offset, slice_buffer.length))
        sid = slice_buffer.slice_id
        self.slice_map.pop(sid, None)

        self.free_ids.append(sid)
        if len(self.free_list) > 16:
            self.try_to_merge_free_list()

    def get(self, slice_id):
        slice_buffer = self.slice_map.get(slice_id, None)
        if slice_buffer is not None:
            return slice_buffer.tensor

    def put(self, slice_id):
        slice_buffer = self.slice_map.get(slice_id, None)
        if slice_buffer is not None:
            self._free(slice_buffer)

    def try_to_merge_free_list(self):
        if not self.free_list or len(self.free_list) == 1:
            return

        self.free_list.sort()
        merged = []
        cur_off, cur_len = self.free_list[0]

        for off, length in self.free_list[1:]:
            if cur_off + cur_len == off:
                cur_len += length
            else:
                merged.append((cur_off, cur_len))
                cur_off, cur_len = off, length

        merged.append((cur_off, cur_len))
        self.free_list = merged

    def get_total_free_bytes(self):
        total_free = sum(l for _, l in self.free_list)
        return total_free 