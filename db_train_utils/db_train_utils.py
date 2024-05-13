import re
import numpy as np
from Bio import SeqIO

def is_valid_dna_sequence(sequence):
    pattern = re.compile('^[AGCTN]+$')
    if not pattern.match(str(sequence)):
        return False
    return True
	# assert pattern.match(str(sequence)), f"Invalid DNA sequence - {sequence}! DNA sequence can only contain A, G, C, T"

def get_input_shape_from_fasta(fasta_path):
    # read one sequence to get the shape
    fasta_sequences = SeqIO.parse(open(fasta_path), 'fasta')
    seq = next(fasta_sequences)
    return len(seq.seq), 4



def oneHot_encode(record):
    string = record.upper().replace('\n', '').replace('U', 'T').replace(' ', '')
    if not is_valid_dna_sequence(string):
        # replace invalid sequences with C #todo - check if this is the best way to handle invalid sequences
        string = re.sub('[^AGCTN]', 'C', string)
        print(f"Invalid DNA sequence - {record}! Replaced invalid characters with N")

    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3} 
    data = [mapping[char] for char in string.upper()]
    return np.eye(4)[data]

def oneHot_encode(record):
    string = record.upper().replace('\n', '').replace('U', 'T').replace(' ', '')
    if not is_valid_dna_sequence(string):
        # replace invalid sequences with C #todo - check if this is the best way to handle invalid sequences
        string = re.sub('[^AGCTN]', 'C', string)
        print(f"Invalid DNA sequence - {record}! Replaced invalid characters with N")

    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4} 
    data = [mapping[char] for char in string.upper()]
    one_hot = np.eye(5)[data]

    # remove the last column (N)
    one_hot = one_hot[:, :4]
    # return it as type bool to save memory
    return one_hot.astype(bool)