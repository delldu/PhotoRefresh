# convert images/a.png output/a.png +append /tmp/a.png
# convert images/b.png output/b.png +append /tmp/b.png
# convert images/c.png output/c.png +append /tmp/c.png
# convert images/d.png output/d.png +append /tmp/d.png

for f in images/*.png; do
    f=`basename $f`

    convert images/$f output/$f +append /tmp/$f

    display /tmp/$f
done
