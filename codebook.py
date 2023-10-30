import torch
import itertools


E8_CB = []
for xs in itertools.product(*([[0.5,1.5,2.5]]*8)):
    if sum(xs) % 2 != 0:
        xs = (*xs[:-1], -xs[-1])
    z = sum(y**2 for y in xs)
    if z <= 10 or ((z == 12) and all(abs(y) <= 1.5 for y in xs) and (sum(abs(y) == 0.5 for y in xs[0:4]) % 2 == 0)):
        E8_CB.append(xs)
E8_CB.append((0.5,0.5,0.5,1.5,1.5,1.5,1.5,-1.5))
E8_CB = torch.tensor(E8_CB)
E8_CB = E8_CB[E8_CB.square().sum(1).argsort(),:]
E8_CBL = (((E8_CB * 4).to(torch.int64) & 255) << (8 * torch.arange(8))).sum(1)

E8_SignMask = torch.zeros(2,2,2,2,2,2,2,dtype=torch.int64)
sm = ((torch.tensor(1,dtype=torch.int64) << 6) - 1) << 2
rsh = (2,)
for i in range(7):
    E8_SignMask = E8_SignMask ^ torch.tensor([0,((sm << (8*i)) ^ (sm << (8*7)))]).view(rsh)
    rsh = rsh + (1,)
E8_SignMask = E8_SignMask.reshape(128)

Mask1 = sum(torch.tensor(1,dtype=torch.int64) << (i*8) for i in range(8))
Mask2 = sum(torch.tensor(2,dtype=torch.int64) << (i*8) for i in range(8))

def decode(i):
    i = torch.tensor(i)
    x = E8_CBL[i & ((1 << 8) - 1)]
    x = x ^ E8_SignMask[(i >> 8) & ((1 << 7) - 1)]
    if (i & (1 << 15)):
        x = x - Mask2
    x = x | Mask1
    return torch.tensor([(x >> (8*i)) & ((1 << 8) - 1) for i in range(8)],dtype=torch.uint8).to(torch.int8).to(torch.float32) / 4


def decode_naive(i):
    i = torch.tensor(i)
    abs = torch.clone(E8_CBL[i & ((1 << 8) - 1)])
    sign = torch.clone(E8_SignMask[(i >> 8) & ((1 << 7) - 1)])

    decoded = torch.empty(8, dtype=torch.int8)
    for j in range(8):
        value = abs & 0xFF
        value = torch.tensor(value, dtype=torch.uint8).to(torch.int8)

        if sign & 0xFF:
            value *= -1

        if (i & (1 << 15)):
            value -= 1
        else:
            value += 1

        decoded[j] = value
        abs >>= 8
        sign >>= 8

    decoded = decoded.to(torch.float32) / 4
    return decoded


E8_Vectors = torch.zeros(1 << 16, 8)
for i in range(1 << 16):
    assert torch.all(decode(i) == decode_naive(i)), f"{i = },\n{decode(i) = },\n{decode_naive(i) = }"
    E8_Vectors[i,:] = decode(i)
