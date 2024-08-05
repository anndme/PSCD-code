import hashlib
import numpy as np

class FM_Sketch:
    def __init__(self, num_hashes, bit_length):
        self.num_hashes = num_hashes
        self.bit_length = bit_length
        self.sketch = np.zeros((num_hashes, bit_length), dtype=int)


    def _hash(self, value, seed):
        value_str = str(value)
        seed_str = str(seed)
        hasher = hashlib.sha256()
        hasher.update(value_str.encode('utf-8') + seed_str.encode('utf-8'))
        hash_value = int(hasher.hexdigest(), 16)
        return hash_value


    def _get_bucket(self, hash_value):
        #
        for j in range(self.bit_length):
            if (hash_value >> (self.bit_length - j - 1)) & 1 == 0:
                return j
        return self.bit_length - 1


    def _count_trailing_zeros(self, hash_value):
        # 计算二进制表示中最右侧连续0的个数
        return (hash_value ^ (hash_value - 1)).bit_length() - 1


    def record(self, item):
        for i in range(self.num_hashes):
            seed = f'seed{i}'.encode('utf-8')
            hash_value = self._hash(item, seed)
            bucket = self._get_bucket(hash_value)
            trailing_zeros = self._count_trailing_zeros(hash_value)
            self.sketch[i, bucket] = max(self.sketch[i, bucket], trailing_zeros)


    def query(self):
        R = [np.max(row) for row in self.sketch]
        if not R:
            return 0

        avg_R = np.mean(R)
        #
        estimate = 1.2928 * (2 ** avg_R)
        return estimate


# if __name__ == "__main__":
#     num_hashes = 10
#     bit_length = 128
#
#     fm_sketch = FM_Sketch(num_hashes, bit_length)
#
#     items = ['apple', 'banana', 'orange', 'apple', 'banana', 'apple', 'banana', 'kiwi','apple1', 'banana1', 'apple1', 'banana1', 'kiwi1']
#     for item in items:
#         fm_sketch.record(item)
#
#     estimate = fm_sketch.query()
#     print(f"q: {estimate:.2f}")

#
# import numpy as np
#
# #
# num_hashes = 10
# bit_length = 128
# fm_sketch = FM_Sketch(num_hashes, bit_length)
#
#
# matrix = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [1, 2, 3],
#     [7, 8, 9],
#     [7, 6, 9],
#     [7, 5, 9],
#     [7, 4, 9],
#     [2, 2, 3],
#     [2, 5, 6],
#     [2, 2, 3],
#     [2, 8, 19],
#     [2, 6, 19],
#     [2, 5, 19],
#     [2, 4, 19]
# ])
#
#
# for row in matrix:
#     fm_sketch.record(tuple(row))
#
#
# estimated_unique_count = fm_sketch.query()
# print(f"q: {estimated_unique_count}")