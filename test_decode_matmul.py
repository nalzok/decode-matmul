from itertools import product

import torch

from decode_matmul_cuda import decode_matmul as decode_matmul_cuda


class Benchmark:
    def __init__(self, name):
        self.name = name
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self.start_event

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        print(f"[{self.name.ljust(16)}] Elapsed: {elapsed_time_ms:.4f}ms")


Mask1 = sum(torch.tensor(1, dtype=torch.int64) << (i*8) for i in range(8))
Mask2 = sum(torch.tensor(2, dtype=torch.int64) << (i*8) for i in range(8))


def decode(weight_compressed, codebook_abs, codebook_sign):
    x = codebook_abs[weight_compressed & ((1 << 8) - 1)]
    x = x ^ codebook_sign[(weight_compressed >> 8) & ((1 << 7) - 1)]
    if (weight_compressed & (1 << 15)):
        x = x - Mask2
    x = x | Mask1
    decoded = torch.tensor([(x >> (8*i)) & ((1 << 8) - 1) for i in range(8)], dtype=torch.uint8).to(torch.int8).to(torch.int32)
    return decoded


def decode_naive(weight_compressed, codebook_abs, codebook_sign):
    bits_abs = weight_compressed & ((1 << 8) - 1)
    bits_sign = (weight_compressed >> 8) & ((1 << 7) - 1)
    bit_shift = (weight_compressed >> 15) & ((1 << 1) - 1)

    abs = codebook_abs[bits_abs].item()
    sign = codebook_sign[bits_sign].item()
    decoded = torch.empty(8, dtype=torch.int8, device=weight_compressed.device)
    for i in range(8):
        value = torch.tensor(abs & 0xFF, dtype=torch.uint8).to(torch.int8)
        assert value % 1 == 0

        if sign & 0xFF:
            value *= -1

        if bit_shift:
            value -= 1
        else:
            value += 1

        decoded[i] = value
        abs >>= 8
        sign >>= 8

    decoded = decoded.to(torch.int8).to(torch.int32)

    return decoded


def decode_matmul_torch(x, weights_compressed, codebook_abs, codebook_sign, decoder_impl):
    M, K = x.shape
    N, _ = weights_compressed.shape

    result = torch.zeros((M, N), dtype=torch.int32, device=x.device)
    for i in range(M):
        for j in range(N):
            for k in range(K // 8):
                weight_compressed = weights_compressed[j, k]
                weight = decoder_impl(weight_compressed, codebook_abs, codebook_sign)
                for kk in range(8):
                    result[i, j] += x[i, 8*k+kk] * weight[kk]

    return result


def build_codebook_abs(device):
    E8_CB = []
    for xs in product(*([[0.5,1.5,2.5]]*8)):
        if sum(xs) % 2 != 0:
            xs = (*xs[:-1], -xs[-1])
        z = sum(y**2 for y in xs)
        if z <= 10 or ((z == 12) and all(abs(y) <= 1.5 for y in xs) and (sum(abs(y) == 0.5 for y in xs[0:4]) % 2 == 0)):
            E8_CB.append(xs)
    E8_CB.append((0.5,0.5,0.5,1.5,1.5,1.5,1.5,-1.5))
    E8_CB = torch.tensor(E8_CB)
    E8_CB = E8_CB[E8_CB.square().sum(1).argsort(),:]
    E8_CBL = (((E8_CB * 4).to(torch.int64) & 255) << (8 * torch.arange(8))).sum(1)
    return E8_CBL.to(device)


def build_codebook_sign(device):
    E8_SignMask = torch.zeros(2, 2, 2, 2, 2, 2, 2, dtype=torch.int64, device=device)
    sm = ((torch.tensor(1, dtype=torch.int64, device=device) << 6) - 1) << 2
    rsh = (2,)
    for i in range(7):
        E8_SignMask = E8_SignMask ^ torch.tensor([0,((sm << (8*i)) ^ (sm << (8*7)))], device=device).view(rsh)
        rsh = rsh + (1,)
    E8_SignMask = E8_SignMask.reshape(128)
    return E8_SignMask


def build_x_weights(M, N, K, device):
    int8_iinfo = torch.iinfo(torch.int8)
    int16_iinfo = torch.iinfo(torch.int16)
    int8_min, int8_max = int8_iinfo.min, int8_iinfo.max + 1
    int16_min, int16_max = int16_iinfo.min, int16_iinfo.max + 1

    x = torch.randint(int8_min, int8_max, size=(M, K), dtype=torch.int8, device=device)
    weights_compressed = torch.randint(int16_min, int16_max, size=(N, K // 8), dtype=torch.int16, device=device)

    return x, weights_compressed


def test_decode_matmul():
    device = torch.device("cuda")

    codebook_abs = build_codebook_abs(device)
    codebook_sign = build_codebook_sign(device)


    M, N, K = 8 * 2, 32 * 3, 16 * 5
    x, weights_compressed = build_x_weights(M, N, K, device)

    result_torch_naive = decode_matmul_torch(x, weights_compressed, codebook_abs, codebook_sign, decode_naive)
    result_torch_bitwise = decode_matmul_torch(x, weights_compressed, codebook_abs, codebook_sign, decode)
    result_cuda = decode_matmul_cuda(x, weights_compressed, codebook_abs, codebook_sign)
    print("Correct (bitwise):", torch.all(result_torch_bitwise == result_cuda).item())
    print("Correct (CUDA):", torch.all(result_torch_naive == result_cuda).item())


    M, N, K = 8, 8192, 8192
    x, weights_compressed = build_x_weights(M, N, K, device)

    for _ in range(3):
        with Benchmark("CUDA"):
            result_cuda = decode_matmul_cuda(x, weights_compressed, codebook_abs, codebook_sign)


if __name__ == '__main__':
    torch.random.manual_seed(42)
    test_decode_matmul()
