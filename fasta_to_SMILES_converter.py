import pandas as pd
from rdkit import Chem

def parse_fasta(file_path):
    """
    Parses a FASTA file and returns a list of peptide sequences.
    The FASTA file is expected to have headers starting with '>'
    followed by the peptide sequence on one or more lines.
    """
    peptides = []
    with open(file_path, 'r') as f:
        sequence = ""
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith(">"):
                if sequence:
                    peptides.append(sequence)
                    sequence = ""
                # Header line is skipped
            else:
                sequence += line  # Append sequence lines (handle multi-line sequences)
        if sequence:
            peptides.append(sequence)
    return peptides

def peptide_to_smiles(peptide):
    """
    Converts a peptide sequence to a SMILES string using RDKit.
    Returns None if conversion fails.
    """
    try:
        mol = Chem.MolFromSequence(peptide)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Exception converting peptide (first 50 chars: {peptide[:50]}...): {e}")
        return None

def main():
    # Use your provided file names
    toxic_file = "toxic_fasta_peptides.txt"
    nontoxic_file = "non_toxic_fasta_peptides.txt"

    # Parse FASTA files
    toxic_peptides = parse_fasta(toxic_file)
    nontoxic_peptides = parse_fasta(nontoxic_file)

    smiles_list = []
    toxicity_list = []

    # Set a cap for peptide length (in amino acids)
    max_length = 650

    # Process toxic peptides (label 1)
    for pep in toxic_peptides:
        print("Processing FASTA peptide:")
        print(pep[:100] + "..." if len(pep) > 100 else pep)
        
        if len(pep) > max_length:
            print(f"Warning: Peptide length ({len(pep)}) exceeds cap of {max_length} amino acids; skipping.")
            continue

        smi = peptide_to_smiles(pep)
        if smi is None:
            print(f"Warning: Failed to convert toxic peptide (first 50 chars: {pep[:50]}...); skipping.")
            continue
        
        print("Converted SMILES:")
        print(smi)
        smiles_list.append(smi)
        toxicity_list.append(1)

    # Process non-toxic peptides (label 0)
    for pep in nontoxic_peptides:
        if len(pep) > max_length:
            print(f"Warning: Peptide length ({len(pep)}) exceeds cap of {max_length} amino acids; skipping.")
            continue

        smi = peptide_to_smiles(pep)
        if smi is None:
            print(f"Warning: Failed to convert non-toxic peptide (first 50 chars: {pep[:50]}...); skipping.")
            continue
        
        smiles_list.append(smi)
        toxicity_list.append(0)

    # Create a DataFrame and export to an Excel file
    df = pd.DataFrame({
        "SMILES": smiles_list,
        "toxicity": toxicity_list
    })
    output_filename = "ToxinPeptideV2.xlsx"
    df.to_excel(output_filename, index=False)
    print(f"Excel file created: {output_filename}")

if __name__ == "__main__":
    main()
