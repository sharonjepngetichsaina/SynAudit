"""Check the Bloom filter: no missed items, and about 1% false positives."""

from bloom import BloomFilter

bloom = BloomFilter(expected_items=10000, false_positive_rate=0.01)
for i in range(10000):
    bloom.add(f"item-{i}")

print("bits:", bloom.m, "hash functions:", bloom.k)
print("all added items found:", all(f"item-{i}" in bloom for i in range(10000)))

false_hits = sum(f"other-{i}" in bloom for i in range(10000))
print("false positive rate:", false_hits / 10000)