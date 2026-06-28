from pathlib import Path
import tarfile
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# download the given file
def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024, timeout: int = 120) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path

# download the taxonomy dump file
def ensure_taxdump_files(data_dir: Path, max_age_days: int = 30) -> tuple[Path, Path, Path]:
    taxdump_url = 'https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz'
    taxdump_path = data_dir / 'taxdump.tar.gz'
    merged_path = data_dir / 'merged.dmp'
    delnodes_path = data_dir / 'delnodes.dmp'

    max_age_seconds = max_age_days * 24 * 3600
    is_stale = (not taxdump_path.exists()
                or taxdump_path.stat().st_size == 0
                or time.time() - taxdump_path.stat().st_mtime > max_age_seconds)

    if is_stale:
        download_file(taxdump_url, taxdump_path)
        # Force re-extraction when archive is refreshed
        merged_path.unlink(missing_ok=True)
        delnodes_path.unlink(missing_ok=True)

    if not (merged_path.exists() and delnodes_path.exists()):
        with tarfile.open(taxdump_path, 'r:gz') as tar:
            tar.extract('merged.dmp', path=data_dir)
            tar.extract('delnodes.dmp', path=data_dir)

    return taxdump_path, merged_path, delnodes_path

# read the merged IDs file
def load_merged(merged_file: Path) -> pd.DataFrame:
    records = []
    with open(merged_file) as f:
        for line in f:
            parts = line.split('\t|\t')
            if len(parts) >= 2:
                records.append((int(parts[0]), int(parts[1].strip().rstrip('\t|'))))
    return pd.DataFrame(records, columns=['old_taxid', 'new_taxid'])

# read the deleted IDs file
def load_delnodes(delnodes_file: Path) -> set[int]:
    taxids = set()
    with open(delnodes_file) as f:
        for line in f:
            parts = line.split('\t|')
            if parts[0].strip().isdigit():
                taxids.add(int(parts[0].strip()))
    return taxids

# download the RefSeq assembly file
def ensure_refseq_assembly_summary(data_dir: Path, max_age_days: int = 30) -> Path:
    url = 'https://ftp.ncbi.nlm.nih.gov/genomes/refseq/assembly_summary_refseq.txt'
    summary_path = data_dir / 'assembly_summary_refseq.txt'

    max_age_seconds = max_age_days * 24 * 3600
    if (not summary_path.exists()) or (time.time() - summary_path.stat().st_mtime > max_age_seconds):
        download_file(url, summary_path)

    return summary_path

# load the RefSeq assembly data
def load_assembly_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', skiprows=1, low_memory=False)
    df.columns = [c.lstrip('# ') for c in df.columns]
    return df

# check that the given list of taxonomy IDs exist in the assembly and are downloadable
def refseq_available_taxids(asm: pd.DataFrame, input_taxids: set[int]) -> set[int]:
    taxid = pd.to_numeric(asm['taxid'], errors='coerce')
    species_taxid = pd.to_numeric(asm.get('species_taxid', pd.NA), errors='coerce')

    mask = taxid.isin(input_taxids) | species_taxid.isin(input_taxids)
    sub = asm[mask].copy()
    sub['taxid'] = taxid[mask]
    sub['species_taxid'] = species_taxid[mask]

    sub = sub[sub['ftp_path'].notna() & (sub['ftp_path'] != 'na')]
    sub['input_taxid'] = sub['taxid'].where(sub['taxid'].isin(input_taxids), sub['species_taxid'])
    sub = sub[sub['input_taxid'].notna()]
    return set(sub['input_taxid'].astype(int).unique())

# select the "best" assembly entry
# check the assembly's taxid and also species_taxid
# check that the entries are downloadable
# rank the available data -> most curated + most complete + newest is preferable
def pick_best_assemblies(asm: pd.DataFrame, input_taxids: set[int]) -> pd.DataFrame:
    asm = asm.copy()

    asm['taxid'] = pd.to_numeric(asm['taxid'], errors='coerce')
    if 'species_taxid' in asm.columns:
        asm['species_taxid'] = pd.to_numeric(asm['species_taxid'], errors='coerce')
    else:
        asm['species_taxid'] = pd.NA

    # keep rows that match either taxid or species_taxid
    sub = asm[
        (asm['taxid'].isin(input_taxids)) |
        (asm['species_taxid'].isin(input_taxids))
    ].copy()

    sub = sub[sub['ftp_path'].notna() & (sub['ftp_path'] != 'na')].copy()
    sub['seq_rel_date'] = pd.to_datetime(sub['seq_rel_date'], errors='coerce')

    # assign: which input taxid does this assembly satisfy?
    # prefer direct taxid match; otherwise species_taxid match
    sub['input_taxid'] = sub['taxid'].where(sub['taxid'].isin(input_taxids), sub['species_taxid'])
    sub = sub[sub['input_taxid'].notna()].copy()
    sub['input_taxid'] = sub['input_taxid'].astype(int)

    # lower rank is better
    refseq_rank = {'reference genome': 0, 'representative genome': 1}
    assembly_level_rank = {'Complete Genome': 0, 'Chromosome': 1, 'Scaffold': 2, 'Contig': 3}

    sub['refseq_rank'] = sub['refseq_category'].map(refseq_rank).fillna(9).astype(int)
    sub['level_rank'] = sub['assembly_level'].map(assembly_level_rank).fillna(9).astype(int)

    sub = sub.sort_values(
        by=['input_taxid', 'refseq_rank', 'level_rank', 'seq_rel_date'],
        ascending=[True, True, True, False],
        kind='mergesort'
    )

    best = sub.groupby('input_taxid', as_index=False).head(1).copy()

    keep_cols = [
        'input_taxid',
        'taxid',
        'species_taxid',
        'organism_name',
        'assembly_accession',
        'refseq_category',
        'assembly_level',
        'seq_rel_date',
        'ftp_path',
    ]
    keep_cols = [c for c in keep_cols if c in best.columns]
    return best[keep_cols].copy()

# prepare the proteome download URL
def proteome_url_from_ftp_path(ftp_path: str) -> str:
    # force HTTPS because 'requests' will crash on 'ftp://'
    https_path = ftp_path.replace('ftp://', 'https://')
    base = https_path.rstrip('/').split('/')[-1]
    return f'{https_path}/{base}_protein.faa.gz'

# parallelized proteome download from NCBI
def download_proteomes(best_map: pd.DataFrame, out_dir: Path, max_workers: int = 4) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _download_one(r):
        input_taxid = int(r.input_taxid)
        acc = r.assembly_accession
        url = proteome_url_from_ftp_path(r.ftp_path)
        out_path = out_dir / f'{input_taxid}_{acc}_protein.faa.gz'
        if out_path.exists() and out_path.stat().st_size > 0:
            return dict(input_taxid=input_taxid, assembly_taxid=int(r.taxid),
                        assembly_accession=acc, proteome_url=url,
                        local_path=str(out_path), status='skipped_exists', error='')
        try:
            download_file(url, out_path)
            status, err = 'ok', ''
        except Exception as e:
            status, err = 'failed', f'{type(e).__name__}: {e}'
        return dict(input_taxid=input_taxid, assembly_taxid=int(r.taxid),
                    assembly_accession=acc, proteome_url=url,
                    local_path=str(out_path), status=status, error=err)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_download_one, r) for r in best_map.itertuples(index=False)]
        rows = [f.result() for f in as_completed(futures)]

    return pd.DataFrame(rows)

# update the NCBI taxonomy ID until everything is resolved
def resolve_merge_chain(taxid: int, merge_map: dict) -> int:
    seen = set()
    while taxid in merge_map and taxid not in seen:
        seen.add(taxid)
        taxid = merge_map[taxid]
    return taxid