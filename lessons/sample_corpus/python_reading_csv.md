# Reading a CSV

Getting tabular data out of a file is one of the first things you do, and one of
the easiest to get subtly wrong.

## With the standard library

```python
import csv

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))
```

Two details matter here. `newline=''` tells the CSV module to handle line
endings itself, and passing `encoding` explicitly avoids the platform default
tripping you up on a file written elsewhere.

`DictReader` uses the first row as headers, so each row comes back as a
dictionary. That is far more usable than a list of lists, where every access
needs a column index you have to remember.

## When the delimiter is not a comma

Some files are separated by tabs or semicolons. Pass `delimiter='\t'` rather
than splitting strings yourself, because quoted fields can contain the
delimiter.

## Files with preamble

Scientific and export data often starts with comment lines before the real
header. Find the header row, feed everything before it back as text to ignore:

```python
reader = csv.DictReader(handle, fieldnames=None)
```

and skip leading rows until the header looks right. This is fiddly, which is
why saving clean CSV in the first place saves so much time.

## Bigger files

For files that do not fit comfortably in memory, stream the reader instead of
calling `list()` on it. The standard library is adequate well beyond what most
people assume.