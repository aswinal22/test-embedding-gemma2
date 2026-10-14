# Python: Binary Search

Binary search finds a target in a sorted collection in logarithmic time by
halving the search space at every step.

## The algorithm

Track a low and high index. Look at the middle element. If it matches, you are
done. If it is smaller than the target, the answer must be to the right;
otherwise it must be to the left. Repeat.

```python
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

## Why logarithmic time matters

A linear scan of one million items takes up to a million comparisons. Binary
search takes about twenty. The cost is that **the input must already be
sorted** — the halving is only valid if order carries information.

## Common mistakes

- Writing `low < high` instead of `low <= high`, which skips the final element.
- Forgetting that the middle index needs integer division. Python's `//`
  handles this; in C or Java you must cast explicitly.
- Returning `None` when `-1` is what the callers expect.

## Related approaches

For a sorted list that is queried constantly, binary search is often the wrong
choice. A dictionary lookup is constant time and usually simpler.