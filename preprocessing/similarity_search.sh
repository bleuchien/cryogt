#!/bin/bash

# run mmseqs to find similar protein sequences for removal from the dataset to prevent leakage
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
if [[ ! -x "$MMSEQS" ]]
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
if [[ ! -d "$MMSEQS_DIR" ]]
then
    mkdir -p "$MMSEQS_DIR/temp"
fi

# step 1 combine all proteomes into one faa file
echo "Combining sampled proteomes."
cat "$PROTEOMES/"*.faa > "$MMSEQS_DIR/proteomes_combined.faa"

# create mmseqs database
echo "Creating mmseqs database."    
"$MMSEQS" createdb "$MMSEQS_DIR/proteomes_combined.faa" "$MMSEQS_DIR/proteomes_combined.db"

# cluster the sequences
echo "Clustering the sequences."
"$MMSEQS" cluster "$MMSEQS_DIR/proteomes_combined.db" "$MMSEQS_DIR/cluster.db" "$MMSEQS_DIR/temp" --min-seq-id 0.95 -c 0.9 --cov-mode 0

# create TSV
echo "Creating a TSV formatted output."
"$MMSEQS" createtsv "$MMSEQS_DIR/proteomes_combined.db" "$MMSEQS_DIR/proteomes_combined.db" "$MMSEQS_DIR/cluster.db" "$MMSEQS_DIR/clusters.tsv"

# keeping only one per cluster
echo "Keeping one sequence per cluster. Output saved to $MMSEQS_DIR/representatives.tsv"
cut -f1 "$MMSEQS_DIR/clusters.tsv" | sort -u > "$MMSEQS_DIR/representatives.tsv"

# return to where we where
popd