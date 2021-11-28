# Get the Data

In the ``apebase`` directory, all 10,000 ape images are stored as PNGs in the ``ipfs`` directory.

If you want to convert them to JPEGs, first, use a simple bash command similar to:

```bash
for i in *.png ; do convert "$i" "${i%.*}.jpg" ; done
```

Good luck!
