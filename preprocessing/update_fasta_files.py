from pathlib import Path
import gzip
import re
import os

# The default fasta files contain as description a protein identifier
# but no taxonomy identifier which is necessary to relate the protein
# entry to the OGT from the dataset.
# This script updates each proteome file and adds the taxonomy ID
# after the > symbol followed by a | and the original comment

# path to the proteomes
proteomes_path = Path('../data/proteomes')

count = 0
failed = 0
skipped = 0

print('Updating proteome fasta files.')

# iterate over every file
for file in sorted(proteomes_path.glob('*.faa.gz')):
    m = re.match(r'^(\d+)_', file.name)
    if not m:
        print(f'FAILED to extract taxonomy ID from {file.name}.')
        failed += 1
        continue
    taxid = m.group(1)

    # read file
    try:
        with gzip.open(file, 'rt') as f:
            content = f.read()
    except (OSError, EOFError) as e:
        print(f'FAILED to read {file.name}: {e}')
        failed += 1
        continue

    # check if the file has already been updated
    if re.search(rf'^>{re.escape(taxid)}\|', content, re.MULTILINE):
        skipped += 1
        continue

    # update file content
    new_content, n_subs = re.subn(r'^>(.*)', lambda m: f'>{taxid}|{m.group(1)}', content, flags=re.MULTILINE)

    if n_subs == 0:
        print(f'WARNING: no FASTA headers found in {file.name}.')
        continue

    # write file
    tmp_path = file.with_suffix(file.suffix + '.tmp')
    with gzip.open(tmp_path, 'wt') as f:
        f.write(new_content)
    os.replace(tmp_path, file)

    count += 1

print(f'{count} files updated, {skipped} already tagged, {failed} failed.')