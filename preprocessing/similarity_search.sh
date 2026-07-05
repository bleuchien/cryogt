#!/bin/bash
set -u

# script removes proteins with high similarity in the first stage and then groups
# the remaining protein sequences to allow splits on cluster boundaries
# to prevent further leakage
#
# mmseqs documentation https://github.com/soedinglab/mmseqs2/wiki
# 
# download mmseqs if necessary -> only tested on Fedora/Linux and MacOS!

# absolute path to this script (works even if called via symlink when readlink -f exists)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# If the script is in cryogt/preprocessing/, project root is one level up:
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

# proteomes directory
PROTEOMES="data/proteomes_sampled"

# mmseqs working directory
MMSEQS_DIR="data/mmseqs"

# mmseqs binary
MMSEQS="preprocessing/mmseqs/bin/mmseqs"

# move to the project root
pushd "$PROJECT_ROOT"

# checking if binary is available
if [[ ! -x $MMSEQS ]]
then
    echo "mmseqs binary doesn not exist. Trying to download it."
    pushd "$SCRIPT_DIR"
    OS=$(uname -o)
    if [[ "$OS" == "Darwin" ]]
    then
        curl -L -O "https://mmseqs.com/latest/mmseqs-osx-universal.tar.gz"
        if [[ -f "mmseqs-osx-universal.tar.gz" ]]
        then
            tar xf mmseqs-osx-universal.tar.gz
        fi
    elif [[ "$OS" == "GNU/Linux" ]]
    then
        curl -L -O "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
                if [[ -f "mmseqs-linux-avx2.tar.gz" ]]
        then
            tar xf mmseqs-linux-avx2.tar.gz
        fi
    fi
    if [[ $? != 0 ]]
    then
        echo "mmseqs install failed. Aborting!"
        exit 1
    fi
    popd
    if [[ ! -x "$MMSEQS" ]]
    then
        echo "mmseqs binary missing. Aborting!"
        exit 1
    fi
fi

# create directory(s) if necessary
if [[ -d $MMSEQS_DIR ]]
then
    echo "mmseqs2 directories already exist. Reusing the existing database might lead to unexpected outcomes!"
else
    mkdir -p "$MMSEQS_DIR/"{stage1,stage2,temp_stage1,temp_stage2}
fi

# stage 1 - remove highly similar sequences

# combine all proteomes into one faa file
echo "Combining sampled proteomes."
cat "$PROTEOMES/"*.faa > "$MMSEQS_DIR/proteomes_combined.faa"

SEQUENCE_COUNT_PRE=$(grep '^>' "$MMSEQS_DIR/proteomes_combined.faa" | wc -l)
echo "Analyzing $SEQUENCE_COUNT_PRE sequences."

echo "Stage 1 : Removing highly similar sequences."

# create mmseqs database
echo "Creating mmseqs database."    
"$MMSEQS" createdb "$MMSEQS_DIR/proteomes_combined.faa" "$MMSEQS_DIR/stage1/proteomes_combined.db"

# cluster the sequences
# --min-seq-id minimum pairwise sequence identity -> high for removing highly similar sequences 0.90-0.99
# -c minimum coverage of the alignment -> high for highly similar sequences 0.8-0.95
# --cov-mode require coverage threshold on both sequences (strict) -> 0
echo "Clustering with high similarity."
"$MMSEQS" cluster "$MMSEQS_DIR/stage1/proteomes_combined.db" "$MMSEQS_DIR/stage1/cluster.db" "$MMSEQS_DIR/temp_stage1" --min-seq-id 0.95 -c 0.9 --cov-mode 0

# create TSV
echo "Creating a TSV formatted output."
"$MMSEQS" createtsv "$MMSEQS_DIR/stage1/proteomes_combined.db" "$MMSEQS_DIR/stage1/proteomes_combined.db" "$MMSEQS_DIR/stage1/cluster.db" "$MMSEQS_DIR/stage1/clusters.tsv"

# keeping only one per cluster
echo "Keeping one sequence per cluster. Output saved to $MMSEQS_DIR/stage1/representatives.txt"
cut -f1 "$MMSEQS_DIR/stage1/clusters.tsv" | sort -u > "$MMSEQS_DIR/stage1/representatives.txt"

# echo "Exporting representative sequences into new FASTA file."
"$MMSEQS" createsubdb "$MMSEQS_DIR/stage1/representatives.txt" "$MMSEQS_DIR/stage1/proteomes_combined.db" "$MMSEQS_DIR/stage1/representatives.db" --id-mode 1
"$MMSEQS" convert2fasta "$MMSEQS_DIR/stage1/representatives.db" "$MMSEQS_DIR/stage1/representatives.faa"

SEQUENCE_COUNT_POST=$(cat "$MMSEQS_DIR/stage1/representatives.txt" | wc -l)
echo "Stage 1 complete: $SEQUENCE_COUNT_POST representative sequences remaining (from $SEQUENCE_COUNT_PRE)."

# stage 2 - clustering the remaining sequences

echo "Stage 2 : Clustering the remaining sequences for train/test/validation split."

# create new database
echo "Creating new database from the representative sequences."
"$MMSEQS" createdb "$MMSEQS_DIR/stage1/representatives.faa" "$MMSEQS_DIR/stage2/representatives.db"

# cluster
# --min-seq-id minimum pairwise sequence identity -> low to group distant relatives 0.3-0.6
# -c minimum coverage of the alignment -> lower coverage 0.5-0.8
# --cov-mode require coverage threshold on both sequences (strict) -> 0
echo "Clustering with low similarity."
"$MMSEQS" cluster "$MMSEQS_DIR/stage2/representatives.db" "$MMSEQS_DIR/stage2/cluster.db" "$MMSEQS_DIR/temp_stage2" --min-seq-id 0.3 -c 0.9 --cov-mode 0

echo "Creating a TSV formatted output."
"$MMSEQS" createtsv "$MMSEQS_DIR/stage2/representatives.db" "$MMSEQS_DIR/stage2/representatives.db" "$MMSEQS_DIR/stage2/cluster.db" "$MMSEQS_DIR/stage2/split_clusters.tsv"

CLUSTERS=$(cut -f1 "$MMSEQS_DIR/stage2/split_clusters.tsv" | sort -u | wc -l)
echo "Stage 2 complete. $CLUSTERS clusters identified. Output written to $MMSEQS_DIR/stage2/split_clusters.tsv."

SEQUENCE_COUNT_DIFF=$((SEQUENCE_COUNT_PRE - SEQUENCE_COUNT_POST))
echo "SUMMARY
  $SEQUENCE_COUNT_DIFF sequences removed due to high similarity. $SEQUENCE_COUNT_POST remaining sequences.
  The remaining sequences can be (roughly) grouped into $CLUSTERS clusters."

# return to where we where
popd