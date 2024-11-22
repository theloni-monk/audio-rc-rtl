import torch

def bool_tensor_to_int8(bool_tensor):
    # Reshape the boolean tensor to have 8 bits per int8 value
    bool_tensor = bool_tensor.view(-1, 8)
    int8_tensor = torch.sum(bool_tensor * (1 << torch.arange(8)), dim=-1).to(torch.int8)
    return int8_tensor

def int32_to_bool_tensor(int32_tensor):
    # Convert int8 tensor to binary representation
    bool_tensor = (int32_tensor.unsqueeze(-1) & (1 << torch.arange(32))) > 0
    return bool_tensor.flatten()


class Int8_LFSR:
    def __init__(self, polydegrees: torch.IntTensor = torch.tensor([10, 7, 0]), seed: torch.IntTensor = torch.tensor(-69)):
        self.degree = max(polydegrees)
        self.poly_degrees = torch.tensor(polydegrees)

        self.seed(seed)

    def seed(self, seed):
        self.statebits = int32_to_bool_tensor(seed.to(torch.int32))[:self.degree+1]

    def gen_bit(self):
        feedback = torch.tensor(int(torch.sum(self.statebits[self.poly_degrees - 1])) % 2, dtype=torch.bool)
        self.statebits = torch.cat((torch.tensor([feedback]), self.statebits[:-1]))
        return torch.tensor([self.statebits[0]])

    def gen_int8(self):
        return bool_tensor_to_int8(torch.cat(tuple(self.gen_bit() for _ in range(8))))

    def gen_n_int8s(self, n):
        return torch.tensor([self.gen_int8() for _ in range(n)])

    def peek_n_int8s(self, n):
        start = self.statebits
        next_n = self.gen_n(n)
        self.state = start
        return next_n
