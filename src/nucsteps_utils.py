from Bio import SeqIO, motifs
import pybedtools as pbt
import pandas as pd


def bedtools_getfasta(df, genome_path, weights=False):
    if isinstance(df, pd.DataFrame):
        bed_file = pbt.BedTool.from_dataframe(df)
    elif isinstance(df, pbt.bedtool.BedTool):
        bed_file = df
    else:
        raise TypeError(f"format of df: {type(df).__name__} is not supported")
    sequences = bed_file.sequence(
        fi=genome_path,
        name=True,
    )
    if weights:
        fasta_records = [
            [record.seq.upper()] * weights * int(record.name.split("::")[0])
            for record in SeqIO.parse(sequences.seqfn, "fasta")
        ]
        fasta_records = [record for records in fasta_records for record in records]
        return fasta_records

    fasta_records = [record.seq.upper() for record in SeqIO.parse(sequences.seqfn, "fasta")]
    return fasta_records


def make_pwm(df, genome_path, weights=False):
    fasta_records = bedtools_getfasta(df, genome_path, weights)
    motif = motifs.create(fasta_records)
    pwm = pd.DataFrame(motif.pwm)
    return pwm


def make_nucleotide_steps(pwm):
    nuc_steps = dict()
        
    ww = [
        row.A * pwm.iloc[i + 1].A
        + row.A * pwm.iloc[i + 1]["T"]
        + row["T"] * pwm.iloc[i + 1]["T"]
        + row["T"] * pwm.iloc[i + 1].A
        for i, row in pwm.iloc[:-1].iterrows()
    ]
    nuc_steps['ww'] = ww

    yy = [
        row.C * pwm.iloc[i + 1].C
        + row.C * pwm.iloc[i + 1]["T"]
        + row["T"] * pwm.iloc[i + 1]["T"]
        + row["T"] * pwm.iloc[i + 1].C
        for i, row in pwm.iloc[:-1].iterrows()
    ]
    nuc_steps['yy'] = yy

    yr = [
        row.C * pwm.iloc[i + 1].A
        + row.C * pwm.iloc[i + 1]["G"]
        + row["T"] * pwm.iloc[i + 1]["G"]
        + row["T"] * pwm.iloc[i + 1].A
        for i, row in pwm.iloc[:-1].iterrows()
    ]
    nuc_steps['yr'] = yr
    
    return nuc_steps