import time

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

prg = cl.Program(ctx, open("miner.cl").read()).build()

header_g = cl.Buffer(ctx, mf.READ_ONLY, 80)
output_g = cl.Buffer(ctx, mf.WRITE_ONLY, 200)
cl.enqueue_copy(queue, header_g, header_h)
search_hash = prg.search_hash
search_hash.set_scalar_arg_dtypes([np.uint32, None, None])

total_mhashes = 0.0
nonce_size = 0x1_0000_0000
batch_size = 0x100_0000
for batch in range(nonce_size // batch_size):
    output_h = bytearray(200)

    t0 = time.time()
    cl.enqueue_copy(queue, output_g, output_h)
    offset = batch * batch_size
    search_hash(queue, (batch_size,), None, offset, header_g, output_g)
    cl.enqueue_copy(queue, output_h, output_g)
    queue.finish()
    t1 = time.time()

    mhashes = batch_size / (t1 - t0) / 1e6
    total_mhashes += mhashes
    print('batch', batch, mhashes, 'avg', total_mhashes / (batch + 1))

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

print('exhaustively searched')
