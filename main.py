import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np
import pyopencl as cl
import hashlib

mf = cl.mem_flags


def pad_bytes(b: bytearray):
    len_b = len(b)
    padding = (len_b + 9) % 64
    b += b'\x80'
    if padding > 0:
        b += (64 - padding) * b'\0'
    b += (len_b * 8).to_bytes(8, 'big')
    return b


def bswap(b: bytearray) -> bytearray:
    assert len(b) % 4 == 0
    for i in range(0, len(b), 4):
        b[i], b[i + 1], b[i + 2], b[i + 3] = b[i + 3], b[i + 2], b[i + 1], b[i]
    return b


class BenchKernel(ABC):
    @abstractmethod
    def prepare_header(self, header_g) -> None:
        pass

    @abstractmethod
    def run(self, batch_size, offset, header_g, output_g) -> None:
        pass


class SimpleKernel(BenchKernel):
    def __init__(self, ctx, queue, prg) -> None:
        self.ctx = ctx
        self.queue = queue
        self.search_hash = prg.search_hash
        self.search_hash.set_scalar_arg_dtypes([np.uint32, None, None])

    def prepare_header(self, header_g) -> None:
        pass

    def run(self, batch_size, offset, header_g, output_g) -> None:
        self.search_hash(self.queue, (batch_size,), None, offset, header_g, output_g)


class KernelMidstate2(BenchKernel):
    def __init__(self, ctx, queue, prg) -> None:
        self.ctx = ctx
        self.queue = queue
        self.search_hash = prg.search_hash
        self.search_hash.set_scalar_arg_dtypes([np.uint32, None, np.uint32, np.uint32, None])
        self.compute_pre_pow_layer = prg.compute_pre_pow_layer
        self.pre_pow_layer_0 = 0
        self.pre_pow_layer_1 = 0

    def prepare_header(self, header_g) -> None:
        pre_pow_layer_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, 8)
        pre_pow_layer_h = np.zeros(2, np.uint32)
        self.compute_pre_pow_layer(self.queue, (1,), None, header_g, pre_pow_layer_g)
        cl.enqueue_copy(self.queue, pre_pow_layer_h, pre_pow_layer_g)
        self.pre_pow_layer_0 = pre_pow_layer_h[0]
        self.pre_pow_layer_1 = pre_pow_layer_h[1]

    def run(self, batch_size, offset, header_g, output_g) -> None:
        self.search_hash(self.queue, (batch_size,), None,
                         offset, header_g, self.pre_pow_layer_0, self.pre_pow_layer_1, output_g)


class KernelMidstate3(BenchKernel):
    def __init__(self, ctx, queue, prg) -> None:
        self.ctx = ctx
        self.queue = queue
        self.search_hash = prg.search_hash
        self.search_hash.set_scalar_arg_dtypes([
            np.uint32,
            None,
            np.uint32, np.uint32,
            np.uint32, np.uint32, np.uint32, np.uint32, np.uint32, np.uint32, np.uint32,
            None,
        ])
        self.compute_pre_pow_layer = prg.compute_pre_pow_layer
        self.compute_pre_chain_layer = prg.compute_pre_chain_layer
        self.pre_pow_layer_0 = 0
        self.pre_pow_layer_1 = 0
        self.pre_chain_layer = []

    def prepare_header(self, header_g) -> None:
        pre_layer_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, 28)
        pre_layer_h = np.zeros(7, np.uint32)
        self.compute_pre_pow_layer(self.queue, (1,), None, header_g, pre_layer_g)
        cl.enqueue_copy(self.queue, pre_layer_h, pre_layer_g)
        self.pre_pow_layer_0 = pre_layer_h[0]
        self.pre_pow_layer_1 = pre_layer_h[1]
        self.compute_pre_chain_layer(self.queue, (1,), None, header_g, pre_layer_g)
        cl.enqueue_copy(self.queue, pre_layer_h, pre_layer_g)
        self.pre_chain_layer = list(pre_layer_h)

    def run(self, batch_size, offset, header_g, output_g) -> None:
        args = list(self.pre_chain_layer)
        args.append(output_g)
        self.search_hash(self.queue, (batch_size,), None,
                         offset, header_g,
                         self.pre_pow_layer_0, self.pre_pow_layer_1,
                         *args)


@dataclass
class BenchCase:
    kernel: Callable[[Any, Any, Any], BenchKernel]
    nonces_per_batch: int
    inner_iterations: int
    kernel_name: str
    result_avg: float = 0


bench_cases = [
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=1,
        kernel_name='miner-naive',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=4,
        kernel_name='miner-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=8,
        kernel_name='miner-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=16,
        kernel_name='miner-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=32,
        kernel_name='miner-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=64,
        kernel_name='miner-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=1,
        kernel_name='miner-midstate-1',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=4,
        kernel_name='miner-midstate-1-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=8,
        kernel_name='miner-midstate-1-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=16,
        kernel_name='miner-midstate-1-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=32,
        kernel_name='miner-midstate-1-loop16',
    ),
    BenchCase(
        kernel=SimpleKernel,
        nonces_per_batch=0x100_0000,
        inner_iterations=64,
        kernel_name='miner-midstate-1-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate2,
        nonces_per_batch=0x100_0000,
        inner_iterations=1,
        kernel_name='miner-midstate-2',
    ),
    BenchCase(
        kernel=KernelMidstate2,
        nonces_per_batch=0x100_0000,
        inner_iterations=4,
        kernel_name='miner-midstate-2-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate2,
        nonces_per_batch=0x100_0000,
        inner_iterations=8,
        kernel_name='miner-midstate-2-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate2,
        nonces_per_batch=0x100_0000,
        inner_iterations=16,
        kernel_name='miner-midstate-2-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate2,
        nonces_per_batch=0x100_0000,
        inner_iterations=32,
        kernel_name='miner-midstate-2-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate2,
        nonces_per_batch=0x100_0000,
        inner_iterations=64,
        kernel_name='miner-midstate-2-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate3,
        nonces_per_batch=0x100_0000,
        inner_iterations=1,
        kernel_name='miner-midstate-3',
    ),
    BenchCase(
        kernel=KernelMidstate3,
        nonces_per_batch=0x100_0000,
        inner_iterations=4,
        kernel_name='miner-midstate-3-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate3,
        nonces_per_batch=0x100_0000,
        inner_iterations=8,
        kernel_name='miner-midstate-3-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate3,
        nonces_per_batch=0x100_0000,
        inner_iterations=16,
        kernel_name='miner-midstate-3-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate3,
        nonces_per_batch=0x100_0000,
        inner_iterations=32,
        kernel_name='miner-midstate-3-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate3,
        nonces_per_batch=0x100_0000,
        inner_iterations=64,
        kernel_name='miner-midstate-3-loop16',
    ),
    BenchCase(
        kernel=KernelMidstate3,
        nonces_per_batch=0x100_0000,
        inner_iterations=1,
        kernel_name='miner-midstate-4',
    ),
]


def run_bench():
    header = """\
    0000000000000000000000000000000000000000000000000000000000000000\
    ffff001d\
    2fb142600000\
    000000000000\
    ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\
    """
    header_h = bytearray.fromhex(header)
    bswap(header_h)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    header_g = cl.Buffer(ctx, mf.READ_ONLY, 80)
    output_g = cl.Buffer(ctx, mf.WRITE_ONLY, 256)
    cl.enqueue_copy(queue, header_g, header_h)

    pow_layer = bytearray.fromhex("""\
                ffff001d\
                2fb142600000\
                000000000000\
                ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\
                """)

    batches_share = 16
    for bench_case in bench_cases:
        print('running', bench_case)
        prg = cl.Program(ctx, open(f"kernels/{bench_case.kernel_name}.cl").read())\
            .build([f'-DITERATIONS={bench_case.inner_iterations}'])
        kernel = bench_case.kernel(ctx, queue, prg)
        kernel.prepare_header(header_g)

        total_mhashes = 0.0
        nonce_size = 0x1_0000_0000
        batch_size = bench_case.nonces_per_batch // bench_case.inner_iterations
        num_batches = nonce_size // bench_case.nonces_per_batch // batches_share
        for batch in range(num_batches):
            output_h = bytearray(256)

            t0 = time.time()
            cl.enqueue_copy(queue, output_g, output_h)
            offset = batch * bench_case.nonces_per_batch
            kernel.run(batch_size, offset, header_g, output_g)
            cl.enqueue_copy(queue, output_h, output_g)
            queue.finish()
            t1 = time.time()

            mhashes = bench_case.nonces_per_batch / (t1 - t0) / 1e6
            total_mhashes += mhashes
            print('batch', batch, '/', num_batches, mhashes, 'avg', total_mhashes / (batch + 1))

            if output_h[0]:
                nonce = offset + int.from_bytes(output_h[4:8], 'little')
                print('found nonce:', nonce)

                pow_layer[12:16] = nonce.to_bytes(4, 'big')
                chain_layer = bytes(32) + hashlib.sha256(pow_layer).digest()
                print('hash:', hashlib.sha256(chain_layer).hexdigest())
                print()
        bench_case.result_avg = total_mhashes / num_batches
        print('exhaustively searched')

    for bench_case in bench_cases:
        print(bench_case)


if __name__ == '__main__':
    run_bench()
