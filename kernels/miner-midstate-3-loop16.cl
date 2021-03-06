typedef uint num_t;

__constant uint H[8] = { 
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__constant uint K[64] = { 
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant uint POW_LAYER_PAD[4] = {
    0x80000000, 0x00000000, 0x00000000, 0x00000180
};

__constant uint CHAIN_LAYER_SCHEDULE_ARRAY[64] = {
    0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
    0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
    0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549,
    0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
    0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7,
    0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
    0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484,
};

#define rot(x, y) rotate((num_t)x, (num_t)y)
#define rotr(x, y) rotate((num_t)x, (num_t)(32-y))

num_t sigma0(num_t a) {
    return rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
}

num_t sigma1(num_t e) {
    return rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
}

num_t choose(num_t e, num_t f, num_t g) {
    return (e & f) ^ (~e & g);
}

num_t majority(num_t a, num_t b, num_t c) {
    return (a & b) ^ (a & c) ^ (b & c);
}

#define extend_s0_part(array, i) \
    (rotr(array[i- 2], 17) ^ rotr(array[i- 2], 19) ^ (array[i- 2] >> 10)) + \
    array[i-7]

void sha256_extend(
    __private num_t *schedule_array
) {
    for (uint i = 16; i < 64; ++i) {
        num_t s0 = rotr(schedule_array[i-15],  7) ^ rotr(schedule_array[i-15], 18) ^ (schedule_array[i-15] >> 3);
        num_t s1 = rotr(schedule_array[i- 2], 17) ^ rotr(schedule_array[i- 2], 19) ^ (schedule_array[i- 2] >> 10);
        schedule_array[i] = schedule_array[i-16] + s0 + schedule_array[i-7] + s1;
    }
}

void sha256_extend_range(
    __private num_t *schedule_array,
    uint offset,
    uint end
) {
    for (uint i = 16 + offset; i < end; ++i) {
        num_t s0 = rotr(schedule_array[i-15],  7) ^ rotr(schedule_array[i-15], 18) ^ (schedule_array[i-15] >> 3);
        num_t s1 = rotr(schedule_array[i- 2], 17) ^ rotr(schedule_array[i- 2], 19) ^ (schedule_array[i- 2] >> 10);
        schedule_array[i] = schedule_array[i-16] + s0 + schedule_array[i-7] + s1;
    }
}

void sha256_compress(
    __private num_t *schedule_array,
    __private num_t *hash,
    uint offset,
    uint end
) {
    // working vars for the compression function
    num_t a = hash[0], b = hash[1], c = hash[2], d = hash[3],
          e = hash[4], f = hash[5], g = hash[6], h = hash[7];
    for (uint i = offset; i < end; ++i) {
        num_t tmp1 = h + sigma1(e) + choose(e, f, g) + K[i] + schedule_array[i];
        num_t tmp2 = sigma0(a) + majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + tmp1;
        d = c;
        c = b;
        b = a;
        a = tmp1 + tmp2;
    }
    hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
}

void sha256_compress_const(
    __constant num_t *schedule_array,
    __private num_t *hash
) {
    // working vars for the compression function
    num_t a = hash[0], b = hash[1], c = hash[2], d = hash[3],
          e = hash[4], f = hash[5], g = hash[6], h = hash[7];
    for (uint i = 0; i < 64; ++i) {
        num_t tmp1 = h + sigma1(e) + choose(e, f, g) + K[i] + schedule_array[i];
        num_t tmp2 = sigma0(a) + majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + tmp1;
        d = c;
        c = b;
        b = a;
        a = tmp1 + tmp2;
    }
    hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
}

void sha256_pow_layer(
    __private num_t *schedule_array,
    __private num_t *hash
) {
    for (uint i = 0; i < 8; ++i) {
        hash[i] = H[i];
    }
    sha256_extend(schedule_array);
    sha256_compress(schedule_array, hash, 0, 64);
}

void sha256_chain_layer(
    __private num_t *schedule_array,
    __private num_t *hash
) {
    for (uint i = 0; i < 8; ++i) {
        hash[i] = H[i];
    }
    sha256_extend(schedule_array);
    sha256_compress(schedule_array, hash, 0, 64);
    sha256_compress_const(CHAIN_LAYER_SCHEDULE_ARRAY, hash);
}

__kernel void search_hash(
    const uint offset,
    __global uint *header,
    const uint pre_pow_layer_0,
    const uint pre_pow_layer_1,
    const uint pre_chain_layer_0,
    const uint pre_chain_layer_1,
    const uint pre_chain_layer_2,
    const uint pre_chain_layer_3,
    const uint pre_chain_layer_4,
    const uint pre_chain_layer_5,
    const uint pre_chain_layer_6,
    __global uint *output
) {
    num_t pow_layer[64];
    num_t chain_layer[64];
    num_t hash[8];
    // Copy prevBlockHash into chain layer
    for (uint i = 0; i < 8; ++i) {
        chain_layer[i] = header[i];
    }
    // Copy nBits, nTime and the first 2 bytes of nNonce into pow_layer
    for (uint i = 0; i < 3; ++i) {
        pow_layer[i] = header[i + 8];
    }
    // Copy hashTxLayer into pow_layer
    for (uint i = 0; i < 8; ++i) {
        pow_layer[i + 4] = header[i + 12];
    }
    // Copy pad into pow_layer
    for (uint i = 0; i < 4; ++i) {
        pow_layer[i + 12] = POW_LAYER_PAD[i];
    }
    for (uint iteration = 0; iteration < ITERATIONS; ++iteration) {
        // Set nNonce of pow_layer
        pow_layer[3] = offset + get_global_id(0) * ITERATIONS + iteration;

        // EXTEND step, pow_layer
        // i = 16, 17 can be pre computed partially
        pow_layer[16]   = pre_pow_layer_0 + extend_s0_part(pow_layer, 16);
        pow_layer[16+1] = pre_pow_layer_1 + extend_s0_part(pow_layer, 17);
        sha256_extend_range(pow_layer, 2, 64);

        // COMPRESS step, pow_layer
        num_t *pow_layer_hash = &chain_layer[8];
        // Initialize pow_layer_hash with initial values
        for (uint i = 0; i < 8; ++i) {
            pow_layer_hash[i] = H[i];
        }
        sha256_compress(pow_layer, pow_layer_hash, 0, 64);

        // EXTEND step, chain_layer
        for (uint i = 0; i < 8; ++i) {
            hash[i] = H[i];
        }
        // i = 16, 17, ..., 22 can be pre computed partially
        chain_layer[16]   = pre_chain_layer_0 + extend_s0_part(chain_layer, 16);
        chain_layer[16+1] = pre_chain_layer_1 + extend_s0_part(chain_layer, 17);
        chain_layer[16+2] = pre_chain_layer_2 + extend_s0_part(chain_layer, 18);
        chain_layer[16+3] = pre_chain_layer_3 + extend_s0_part(chain_layer, 19);
        chain_layer[16+4] = pre_chain_layer_4 + extend_s0_part(chain_layer, 20);
        chain_layer[16+5] = pre_chain_layer_5 + extend_s0_part(chain_layer, 21);
        chain_layer[16+6] = pre_chain_layer_6 + extend_s0_part(chain_layer, 22);
        sha256_extend_range(chain_layer, 7, 64);

        // COMPRESS step, chain_layer
        sha256_compress(chain_layer, hash, 0, 64);
        sha256_compress_const(CHAIN_LAYER_SCHEDULE_ARRAY, hash);
        
        if (hash[0] == 0) {
            output[0] = 1;
            output[1] = get_global_id(0) * ITERATIONS + iteration;
        }
    }
}

__kernel void compute_pre_pow_layer(
    __global uint *header,
    __global uint *result
) {
    num_t pow_layer[64];
    // Copy nBits, nTime and the first 2 bytes of nNonce into pow_layer
    for (uint i = 0; i < 3; ++i) {
        pow_layer[i] = header[i + 8];
    }

    for (uint i = 16; i < 18; ++i) {
        num_t s0 = rotr(pow_layer[i-15],  7) ^ rotr(pow_layer[i-15], 18) ^ (pow_layer[i-15] >> 3);
        result[i - 16] = pow_layer[i-16] + s0;
    }
}

__kernel void compute_pre_chain_layer(
    __global uint *header,
    __global uint *result
) {
    num_t chain_layer[64];
    // Copy prevBlockHash into chain_layer
    for (uint i = 0; i < 8; ++i) {
        chain_layer[i] = header[i];
    }

    for (uint i = 16; i < 23; ++i) {
        num_t s0 = rotr(chain_layer[i-15],  7) ^ rotr(chain_layer[i-15], 18) ^ (chain_layer[i-15] >> 3);
        result[i - 16] = chain_layer[i-16] + s0;
    }
}

__kernel void extend_schedule_array(
    __global uint *schedule_array_global
) {
    num_t schedule_array[64];
    for (uint i = 0; i < 64; ++i) {
        schedule_array[i] = schedule_array_global[i];
    }
    sha256_extend(schedule_array);
    for (uint i = 0; i < 64; ++i) {
        schedule_array_global[i] = schedule_array[i];
    }
}

__kernel void partial_hash(
    const uint length,
    __global uint *message,
    __global uint *result_hash
) {
    num_t schedule_array[64];
    num_t hash[8];
    for (uint i = 0; i < length; ++i) {
        schedule_array[i] = message[i];
    }
    sha256_extend(schedule_array);
    sha256_compress(schedule_array, hash, 0, length);
    for (uint i = 0; i < 8; ++i) {
        result_hash[i] = hash[i];
    }
}
