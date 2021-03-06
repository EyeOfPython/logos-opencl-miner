import time
from dataclasses import dataclass

import numpy as np
import pyopencl as cl
import os
import hashlib


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


@dataclass
class BenchCase:
    nonces_per_batch: int
    inner_iterations: int
    kernel_name: str
    result_avg: float = 0


bench_cases = [
    BenchCase(
        nonces_per_batch=0x100_0000,
        inner_iterations=16,
        kernel_name='miner-loop16',
    ),
    BenchCase(
        nonces_per_batch=0x100_0000,
        inner_iterations=1,
        kernel_name='miner-naive',
    ),
    BenchCase(
        nonces_per_batch=0x100_0000,
        inner_iterations=1,
        kernel_name='miner-midstate-1',
    ),
]


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

mf = cl.mem_flags

header_g = cl.Buffer(ctx, mf.READ_ONLY, 80)
output_g = cl.Buffer(ctx, mf.WRITE_ONLY, 200)
cl.enqueue_copy(queue, header_g, header_h)


batches_share = 1
for bench_case in bench_cases:
    prg = cl.Program(ctx, open(f"kernels/{bench_case.kernel_name}.cl").read()).build()
    search_hash = prg.search_hash
    search_hash.set_scalar_arg_dtypes([np.uint32, None, None])

    total_mhashes = 0.0
    nonce_size = 0x1_0000_0000
    batch_size = bench_case.nonces_per_batch // bench_case.inner_iterations
    inner_iterations = 16
    num_batches = nonce_size // bench_case.nonces_per_batch // batches_share
    for batch in range(num_batches):
        output_h = bytearray(200)

        t0 = time.time()
        cl.enqueue_copy(queue, output_g, output_h)
        offset = batch * bench_case.nonces_per_batch
        search_hash(queue, (batch_size,), None, offset, header_g, output_g)
        cl.enqueue_copy(queue, output_h, output_g)
        queue.finish()
        t1 = time.time()

        mhashes = bench_case.nonces_per_batch / (t1 - t0) / 1e6
        total_mhashes += mhashes
        print('batch', batch, '/', num_batches, mhashes, 'avg', total_mhashes / (batch + 1))

        if output_h[0]:
            nonce = offset + int.from_bytes(output_h[4:8], 'little')
            print('found nonce:', nonce)

            pow_layer = bytearray.fromhex("""\
            ffff001d\
            2fb142600000\
            000000000000\
            ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\
            """)

            pow_layer[12:16] = nonce.to_bytes(4, 'big')
            chain_layer = bytes(32) + hashlib.sha256(pow_layer).digest()
            print('hash:', hashlib.sha256(chain_layer).hexdigest())
            print()
    bench_case.result_avg = total_mhashes / num_batches
    print('exhaustively searched')

for bench_case in bench_cases:
    print(bench_case)
