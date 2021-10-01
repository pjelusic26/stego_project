from os import stat
from numpy import float32
import numpy as np
from stego import stego_block

class seed_key:

    def __init__(self, seed):
        self.seed = seed
        pass

    def generator(self, length):

        np.random.seed(self.seed)
        key1 = np.random.randint(2, size = (length, 1)).astype(np.float32)
        key2 = np.random.randint(2, size = (length, 1)).astype(np.float32)

        return key1, key2

    @staticmethod
    def randomize(total):

        list_ordered = np.arange(16)
        return np.random.permutation(list_ordered)
        

seedTest = seed_key(8)

# key1, key2 = seedTest.generator(5)

# print(key1)
# print(key2)

# random_list = seed_key.randomize(16)
# print(random_list)
# print(random_list[0])
# print(random_list[0].dtype)



stego_object = stego_block(seed_pattern = 5, seed_message = 6, seed_permutation = 7)

pattern_a, pattern_b = stego_object.generate_pattern(length = 5)
print(pattern_a)
print(pattern_b)

message = stego_object.generate_message(bit_amount = 5)
print(message)

permutation = stego_object.generate_permutation(block_number = 16)
print(permutation)