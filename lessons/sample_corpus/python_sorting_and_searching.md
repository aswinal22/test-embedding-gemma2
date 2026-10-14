# Sorting and Searching

Quicksort, merge sort, and knowing when to use neither.

## Quicksort

Pick a pivot, partition the list so smaller values are left and larger values
are right, then recurse on each side.

```python
def quicksort(items):
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    lower = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    higher = [x for x in items if x > pivot]
    return quicksort(lower) + equal + quicksort(higher)
```

Average case is O(n log n). Worst case is O(n squared), which happens when the
pivot is repeatedly the smallest or largest element — for example on
already-sorted input with a naive middle-element pivot.

## Merge sort

Split in half, sort each half, merge. Consistently O(n log n) regardless of
input order, at the cost of extra memory for the merge step. Preferred when the
data is on disk, where predictable access patterns matter more than constants.

## The practical point

For a list you are searching repeatedly, sorting once at O(n log n) and then
binary searching is worse than a dictionary at O(1) per lookup. Sorting pays
off only when you also need the order.

Do not sort to search unless you also want the sequence sorted. That is a
frequent and expensive misunderstanding.