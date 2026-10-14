# Python: Reversing a List

A common interview question that has several answers of differing quality.

## Slice notation

The fastest and most readable option. `items[::-1]` walks the list backwards
and returns a new list:

```python
def reverse_list(items):
    return items[::-1]
```

The trailing `-1` is what makes it walk backwards. This creates a **new**
list; the original is untouched.

## In-place reversal

If you need to reverse without allocating a second list, walk from both ends
and swap:

```python
def reverse_in_place(items):
    left, right = 0, len(items) - 1
    while left < right:
        items[left], items[right] = items[right], items[left]
        left += 1
        right -= 1
    return items
```

Half the iterations, and it mutates the caller's list. Choose it when memory
matters more than not surprising the caller.

## Choosing between them

Slice notation for almost everything. Reach for in-place only when the list is
large enough that the copy is a real cost, or when the caller explicitly asked
for mutation.