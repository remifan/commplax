# Copyright 2021 The Commplax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from jax import lax, jit
import numpy as np
from functools import partial


def idquant(p, n):
    '''
    solve P', P' = argmin(P') (D_KL(P', P)), subject to P' is n-type

    [1] G. Böcherer and B. C. Geiger. (Mar. 2015). “Optimal quantization for distribution synthesis.” 
        [Online]. Available: http://arxiv.org/abs/1307.6843
    '''
    p = np.asarray(p)
    assert np.allclose(p.sum(), 1.0), 'sum(p) does not equal to 1.0'

    n_i = np.zeros(len(p), dtype=int)
    t = np.log(1. / p)
    p_quant = t.copy()
    for _ in range(n):
        index = np.argmin(p_quant)
        cj = n_i[index] + 1
        n_i[index] = cj
        p_quant[index] = (cj + 1) * np.log(cj + 1) - cj * np.log(cj) + t[index]
    
    return n_i


def n_choose_k_log2(n, k):
    ''' eval log_2 of binominal coefficient (n choose k)
        function sums up logarithm of k quotients starting from n/k downto n-k+1/1
    '''
    I = np.arange(k) + 1
    nck = np.log2((n - (k - I)) / I).sum()
    return nck


def n_choose_ks_log2(n, ks):
    ''' eval log_2(n!/prod_i k(i)!) '''

    assert np.sum(ks) == n

    ks = np.sort(ks)[:-1]
    ns = n - np.pad(ks, [1, 0]).cumsum()[:len(ks)]
    out = np.sum([n_choose_k_log2(n, k) for n, k in zip(ns, ks)])
    return out


def quant_to_ntype(p, n):
    ''' quantize to n-type distribution '''
    n_i = idquant(p, n)
    num_info_bits = int(np.floor(n_choose_ks_log2(n, n_i)))
    return n_i, num_info_bits


def find_n_by_maxbitnum(p, b):
    t = 1
    num = n_choose_ks_log2(t, idquant(p, t))
    while num < b:
        t *= 2
        num = n_choose_ks_log2(t, idquant(p, t))
    n = t
    while (abs(num - b) > 1) & (t != 1):
        t /= 2
        if num > b:
            n -= t
        else:
            n += t
        num = n_choose_ks_log2(n, idquant(p, int(n)))
    n = int(n) - 1
    return n


def CCDM(rate, freqs, num_state_bits=16):
    '''
    [1] Schulte, Patrick, and Georg Böcherer. "Constant composition distribution matching."
        IEEE Transactions on Information Theory 62.1 (2015): 430-434.
    [2] Said, Amir. "Introduction to arithmetic coding-theory and practice."
          Hewlett Packard Laboratories Report (2004): 1057-7149.
    '''
    assert num_state_bits <= 30

    n_bits, n_syms = rate
    c_syms = freqs.shape[0]

    full_range = 1 << num_state_bits
    half_range = full_range >> 1
    quarter_range = half_range >> 1
    minimum_range = quarter_range + 2 
    maximum_total = minimum_range
    state_mask = full_range - 1

    empty_syms = jnp.full(n_syms, -1)
    empty_bits = jnp.full(n_bits, -1)


    def freqs2cdf(F):
        N = jnp.sum(F)
        F = jnp.pad(F, [1, 0])
        cs = jnp.cumsum(F)
        lb = cs[:-1]
        ub = cs[1:]
        cdf = jnp.stack([lb, ub], axis=1)
        return cdf, N

    def read_bits(bits, bi):
        bit = jnp.where(bi < bits.shape[0], bits[bi], 0)
        bi = bi + 1
        return bit, bi

    def write_bits(bits, bi, bit):
        bits = bits.at[bi].set(bit)
        bi = bi + 1
        return bits, bi

    def post_shift(low, high):
        low, high = (low << 1) & state_mask, ((high << 1) & state_mask) | 1
        return low, high

    def post_underflow(low, high):
        low, high = (low << 1) ^ half_range, ((high ^ half_range) << 1) | half_range | 1
        return low, high

    def enc_shift(S):
        code, low, high, bi, bits = S
        bit, bi = read_bits(bits, bi)
        code = ((code << 1) & state_mask) | bit
        low, high = post_shift(low, high)
        return code, low, high, bi, bits

    def enc_underflow(S):
        code, low, high, bi, bits = S
        bit, bi = read_bits(bits, bi)
        code = (code & half_range) | ((code << 1) & (state_mask >> 1)) | bit
        low, high = post_underflow(low, high)
        S = code, low, high, bi, bits
        return S

    def dec_shift(S):
        num_underflow, low, high, bi, bits = S
        bit = low >> (num_state_bits - 1) 
        bit = jnp.where(low <= high, bit, 1) # ready to terminate & flush bits
        bits, bi = write_bits(bits, bi, bit)
        
        # Write out the saved underflow bits
        num_underflow, bits, bi = lax.while_loop(
            lambda v: v[0] > 0,
            lambda v: (v[0] - 1, *write_bits(*v[1:], bit ^ 1)),
            (num_underflow, bits, bi)
            )
        
        low, high = post_shift(low, high)

        S = num_underflow, low, high, bi, bits
        return S

    def dec_underflow(S):
        num_underflow, low, high, bi, bits = S
        num_underflow = num_underflow + 1
        low, high = post_underflow(low, high)
        S = num_underflow, low, high, bi, bits
        return S

    def update(S, sym, shift_fn, underflow_fn):
        x, freqs, low, high, bi, bits = S

        range_ = high - low + 1
        
        cdf, total = freqs2cdf(freqs)
        symlow = cdf[sym, 0]
        symhigh = cdf[sym, 1]

        # if symlow == symhigh:
        #     raise ValueError("Symbol has zero frequency")
        # Update range
        low, high  = low + symlow  * range_ // total, low + symhigh * range_ // total - 1
        # While low and high have the same top bit value, shift them out
        x, low, high, bi, bits = lax.while_loop(
            lambda v: ((v[1] ^ v[2]) & half_range) == 0,
            shift_fn,
            (x, low, high, bi, bits),
            )
        # Now low's top bit must be 0 and high's top bit must be 1
        
        # While low's top two bits are 01 and high's are 10, delete the second highest bit of both
        x, low, high, bi, bits = lax.while_loop(
            lambda v: (v[1] & ~v[2] & quarter_range) != 0,
            underflow_fn,
            (x, low, high, bi, bits),
            )

        freqs = freqs.at[sym].set(freqs[sym] - 1)
        S = x, freqs, low, high, bi, bits

        return S

    def encode_step(S):
        code, freqs, low, high, bi, bits = S

        # Translate from coding range scale to frequency table scale
        cdf, total = freqs2cdf(freqs)
        range_ = high - low + 1
        offset = code - low
        value = ((offset + 1) * total - 1) // range_
        
        # A kind of binary search. Find highest symbol such that freqs.get_low(symbol) <= value.
        start = 0
        end = c_syms

        sym = lax.while_loop(
            lambda v: v[2] - v[0] > 1,
            lambda v: lax.cond(
                cdf[v[1], 0] > value,
                lambda *_: (v[0], (v[0] + v[1]) >> 1, v[1]),
                lambda *_: (v[1], (v[1] + v[2]) >> 1, v[2]),
                ),
            (start, (start + end) >> 1, end),
            )[0]

        S = code, freqs, low, high, bi, bits
        S = update(S, sym, shift_fn=enc_shift, underflow_fn=enc_underflow)
        
        return S, sym

    def decode_step(S, sym):
        S = update(S, sym, shift_fn=dec_shift, underflow_fn=dec_underflow)
        return S

    def encode(bits):
        assert freqs.shape[0] == c_syms

        code = lax.scan(lambda c, b: (c << 1 | b, 0), 0, bits[:num_state_bits])[0]
        bi = num_state_bits
        low = 0
        high = state_mask
        S = code, freqs, low, high, bi, bits
        syms = lax.scan(lambda c, _: encode_step(c), S, empty_syms)[1]
        return syms

    def decode(syms):
        assert freqs.shape[0] == c_syms

        num_underflows = 0
        bi = 0
        low = 0
        high = state_mask
        S = num_underflows, freqs, low, high, bi, empty_bits
        S = lax.scan(lambda c, s: (decode_step(c, s), 0), S, syms)[0]
        bits = decode_step(S, 0)[-1] # terminate & flush
        return bits

    return encode, decode


def CCDM_helper(target_p, n_syms=128, max_no_bits=None, *args, **kwargs):
    assert np.allclose(np.sum(target_p), 1.0)

    p = np.asarray(target_p)
    if max_no_bits is not None:
        n_syms = find_n_by_maxbitnum(p, max_no_bits)
    freqs, n_bits = quant_to_ntype(p, n_syms)
    rate = n_bits, n_syms

    enc, dec = CCDM(rate, freqs, *args, **kwargs)

    return freqs, rate, enc, dec

