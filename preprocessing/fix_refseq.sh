#!/bin/bash

# refseq file
REFSEQ="../data/assembly_summary_refseq.txt"
REFSEQ_FIX="../data/assembly_summary_refseq.fixed.txt"

# fix refseq file column count
awk -F'\t' -v OFS='\t' '
NF<39 {print; next}
NF==39 {
  $16 = $16 " " $17
  for (i=17; i<39; i++) $i = $(i+1)
  NF=38
  print
  next
}
{ print > "/dev/stderr" }
' "$REFSEQ" > "$REFSEQ_FIX"

# move fixed file to original
mv "$REFSEQ_NEW" "$REFSEQ"