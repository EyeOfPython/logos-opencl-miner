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

__constant uint CHAIN_LAYER_ROUND2[16] = {
    0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
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

void sha256_round(
    __private num_t *schedule_array,
    __private num_t *hash
) {
    for (uint i = 16; i < 64; ++i) {
        num_t s0 = rotr(schedule_array[i-15],  7) ^ rotr(schedule_array[i-15], 18) ^ (schedule_array[i-15] >> 3);
        num_t s1 = rotr(schedule_array[i- 2], 17) ^ rotr(schedule_array[i- 2], 19) ^ (schedule_array[i- 2] >> 10);
        schedule_array[i] = schedule_array[i-16] + s0 + schedule_array[i-7] + s1;
    }

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
    sha256_round(schedule_array, hash);
}

void sha256_chain_layer(
    __private num_t *schedule_array,
    __private num_t *hash
) {
    for (uint i = 0; i < 8; ++i) {
        hash[i] = H[i];
    }
    sha256_round(schedule_array, hash);
    for (uint i = 0; i < 16; ++i) {
        schedule_array[i] = CHAIN_LAYER_ROUND2[i];
    }
    sha256_round(schedule_array, hash);
}

__kernel void search_hash(
    const uint offset,
    __global uint *header,
    __global uint *output
) {
    num_t pow_layer[64];
    num_t chain_layer[64];
    num_t hash[8];
    for (uint i = 0; i < 8; ++i) {
        chain_layer[i] = header[i];
    }
    for (uint i = 0; i < 12; ++i) {
        pow_layer[i] = header[i + 8];
    }
    for (uint i = 0; i < 4; ++i) {
        pow_layer[i + 12] = POW_LAYER_PAD[i];
    }

    pow_layer[3] = offset + get_global_id(0);

    sha256_pow_layer(pow_layer, &chain_layer[8]);
    sha256_chain_layer(chain_layer, hash);
    
    if (hash[0] == 0) {
        output[0] = 1;
        output[1] = get_global_id(0);
    }
}
