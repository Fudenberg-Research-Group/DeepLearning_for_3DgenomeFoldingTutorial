import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
import json
from basenji import dataset, seqnn

def set_diag(arr, x, i=0, copy=False):
    """
    copy from cooltools
    Rewrite the i-th diagonal of a matrix with a value or an array of values.
    Supports 2D arrays, square or rectangular. In-place by default.

    Parameters
    ----------
    arr : 2-D array
        Array whose diagonal is to be filled.
    x : scalar or 1-D vector of correct length
        Values to be written on the diagonal.
    i : int, optional
        Which diagonal to write to. Default is 0.
        Main diagonal is 0; upper diagonals are positive and
        lower diagonals are negative.
    copy : bool, optional
        Return a copy. Diagonal is written in-place if false.
        Default is False.

    Returns
    -------
    Array with diagonal filled.

    Notes
    -----
    Similar to numpy.fill_diagonal, but allows for kth diagonals as well.
    This solution was borrowed from
    http://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy

    """
    if copy:
        arr = arr.copy()
    start = max(i, -arr.shape[1] * i)
    stop = max(0, (arr.shape[1] - i)) * arr.shape[1]
    step = arr.shape[1] + 1
    arr.flat[start:stop:step] = x
    return arr

def print_partial_model_summary(model, num_layers=5):
    """
    Print the summary of the first few layers of a Keras model.
    
    Parameters:
    model (tf.keras.Model): The Keras model.
    num_layers (int): The number of layers to include in the summary.
    """
    print(f"Model: {model.name}")
    print("_" * 65)
    print(f"Layer (type)                 Output Shape              Param # ")
    print("=" * 65)
    
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for i, layer in enumerate(model.layers[:num_layers]):
        output_shape = layer.output_shape
        param_count = layer.count_params()
        
        total_params += param_count
        if layer.trainable:
            trainable_params += param_count
        else:
            non_trainable_params += param_count
        
        name = layer.name
        class_name = layer.__class__.__name__
        
        print(f"{name:<25} ({class_name:<15}) {str(output_shape):<25} {param_count:<10,}")
    
    print("=" * 65)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("_" * 65)


def permute_seq_k(seq_1hot, k=2):
    """
    Permute a 1hot encoded sequence by k-mers.

    Parameters
    ----------
    seq_1hot : numpy.array
        n_bases x 4 array
    k : int
        number of bases kept together in permutations.
    """

    seq_length = len(seq_1hot)
    if seq_length % k != 0:
        raise ValueError("Sequence length must be divisible by k")

    seq_1hot_perm = np.zeros_like(seq_1hot)

    num_permutations = seq_length // k
    perm_inds = np.arange(num_permutations) * k
    np.random.shuffle(perm_inds)

    for i in range(k):
        seq_1hot_perm[i::k] = seq_1hot[perm_inds + i, :].copy()

    return seq_1hot_perm


def get_relative_window_coordinates(s, shift=0, seq_length=1310720):
    """
    Calculates the relative coordinates of a genomic window within an expanded and optionally shifted sequence.

    This function takes a genomic window (defined by start and end positions) and calculates its relative
    position within an expanded sequence of a specified length. The expansion is symmetric around the original
    window, and an optional shift can be applied. The function returns the relative start and end positions
    of the original window within this expanded sequence.

    Parameters:
    - s (Series): A pandas Series or similar object containing the original genomic window, with 'start'
      and 'end' fields.
    - shift (int, optional): The number of base pairs to shift the center of the window. Default is 0.
    - seq_length (int, optional): The total length of the expanded sequence. Default is 1310720.

    Returns:
    - tuple: A tuple (relative_start, relative_end) representing the relative start and end positions
      of the original window within the expanded sequence.
    """
    start, end = s.start, s.end
    if abs(end - start) % 2 != 0:
        start = start - 1

    span_length = abs(end - start)
    length_diff = seq_length - span_length
    up_length = length_diff // 2

    # relative start and end of the span of interest in the prediction window
    relative_start = up_length + 1 + shift
    relative_end = relative_start + span_length

    return relative_start, relative_end


def expand_and_check_window(s, chrom_sizes_table, shift=0, seq_length=1310720):
    """
    Expands a genomic window to a specified sequence length and checks its validity against chromosome sizes.

    Given a genomic window (defined by chromosome, start, and end), this function expands the window to a specified
    sequence length. It ensures that the expanded window is symmetric around the original window and optionally
    applies a shift. It then checks if the expanded window is valid (i.e., within the bounds of the chromosome).

    Parameters:
    - s (Series): A pandas Series or similar object containing the original genomic window, with 'chrom', 'start',
      and 'end' fields.
    - chrom_sizes_table (DataFrame): A pandas DataFrame containing chromosome sizes with 'chrom' and 'size' columns.
    - shift (int, optional): The number of base pairs to shift the center of the window. Default is 0.
    - seq_length (int, optional): The desired total length of the expanded window. Default is 1310720.

    Returns:
    - tuple: A tuple (chrom, up_start, down_end) representing the chromosome, the start, and the end of the
      expanded and potentially shifted window.

    Note: If the shift is too large or if the expanded window extends beyond the chromosome boundaries, an
    Exception is raised.
    """
    chrom, start, end = s.chrom, s.start, s.end
    if abs(end - start) % 2 != 0:
        start = start - 1

    span_length = abs(end - start)
    length_diff = seq_length - span_length
    up_length = down_length = length_diff // 2

    if shift > up_length:
        raise Exception(
            "For the following window of interest: ",
            chrom,
            start,
            end,
            "shift excludes the window of interest from the prediction window.",
        )

    # start and end in genomic coordinates (optionally shifted)
    up_start, down_end = start - up_length - shift, end + down_length - shift

    # checking if a genomic prediction can be centered around the span
    chr_size = int(
        chrom_sizes_table.loc[chrom_sizes_table["chrom"] == chrom, "size"].iloc[0]
    )

    if up_start < 0 or down_end > chr_size:
        raise Exception(
            "The prediction window for the following window of interest: ",
            chrom,
            start,
            end,
            "cannot be centered.",
        )
    return chrom, up_start, down_end


def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    """dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
      n_sample:  sample ACGT for N
    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype="float16")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="bool")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                elif n_sample:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1

    return seq_code


def central_permutation_seqs_gen(
    seq_coords_df,
    genome_open,
    chrom_sizes_table,
    permutation_window_shift=0,
    revcomp=False,
    seq_length=1310720,
):
    """
    Generates sequences for a set of genomic coordinates, applying central permutations and optionally
    operating on reverse complements, with an additional option for shifting the permutation window.

    This generator function takes a DataFrame `seq_coords_df` containing genomic coordinates
    (chromosome, start, end, strand), a genome file handler `genome_open` to fetch sequences, and
    a table of chromosome sizes `chrom_sizes_table`. It yields sequences with central permutations
    around the coordinates specified in `seq_coords_df`, considering an optional shift for the
    permutation window. If `rc` is True, the reverse complement of these sequences is generated.

    Parameters:
    - seq_coords_df (pandas.DataFrame): DataFrame with columns 'chrom', 'start', 'end', 'strand',
                                        representing genomic coordinates of interest.
    - genome_open (GenomeFileHandler): A file handler for the genome to fetch sequences.
    - chrom_sizes_table (pandas.DataFrame): DataFrame with columns 'chrom' and 'size', representing
                                            the sizes of chromosomes in the genome.
    - permutation_window_shift (int, optional): The number of base pairs to shift the center of the
                                                 permutation window. Default is 0.
    - rc (bool, optional): If True, operates on reverse complement of the sequences. Default is False.
    - seq_length (int, optional): The total length of the sequence to be generated. Default is 1310720.

    Yields:
    numpy.ndarray: One-hot encoded DNA sequences. Each sequence is either the original or its central
                   permutation, with or without reverse complement as specified by `rc`.

    Raises:
    Exception: If the prediction window for a given span cannot be centered within the chromosome.
    """

    for s in seq_coords_df.itertuples():
        list_1hot = []

        chrom, window_start, window_end = expand_and_check_window(
            s, chrom_sizes_table, shift=permutation_window_shift
        )
        permutation_start, permutation_end = get_relative_window_coordinates(
            s, shift=permutation_window_shift
        )

        wt_seq_1hot = dna_1hot(
            genome_open.fetch(chrom, window_start, window_end).upper()
        )
        if revcomp:
            rc_wt_seq_1hot = hot1_rc(wt_seq_1hot)
            list_1hot.append(rc_wt_seq_1hot.copy())
        else:
            list_1hot.append(wt_seq_1hot.copy())

        alt_seq_1hot = wt_seq_1hot.copy()
        permuted_span = permute_seq_k(
            alt_seq_1hot[permutation_start:permutation_end], k=1
        )
        alt_seq_1hot[permutation_start:permutation_end] = permuted_span

        if revcomp:
            rc_alt_seq_1hot = hot1_rc(alt_seq_1hot.copy())
            list_1hot.append(rc_alt_seq_1hot)
        else:
            list_1hot.append(alt_seq_1hot)

        # yielding first the reference, then the permuted sequence
        for sequence in list_1hot:
            yield sequence


def ut_dense(preds_ut, diagonal_offset=2):
    """Construct symmetric dense prediction matrices from upper triangular vectors.

    Parameters
    -----------
    preds_ut : ( M x O) numpy array
        Upper triangular matrix to convert. M is the number of upper triangular entries,
        and O corresponds to the number of different targets.
    diagonal_offset : int
        Number of diagonals that are added as zeros in the conversion.
        Typically 2 diagonals are ignored in Hi-C data processing.

    Returns
    --------
    preds_dense : (D x D x O) numpy array
        Each output upper-triangular vector is converted to a symmetric D x D matrix.
        Output matrices have zeros at the diagonal for `diagonal_offset` number of diagonals.

    """
    ut_len, num_targets = preds_ut.shape

    # infer original sequence length
    seq_len = int(np.sqrt(2 * ut_len + 0.25) - 0.5)
    seq_len += diagonal_offset

    # get triu indexes
    ut_indexes = np.triu_indices(seq_len, diagonal_offset)
    assert len(ut_indexes[0]) == ut_len

    # assign to dense matrix
    preds_dense = np.zeros(shape=(seq_len, seq_len, num_targets), dtype=preds_ut.dtype)
    preds_dense[ut_indexes] = preds_ut

    # symmetrize
    preds_dense += np.transpose(preds_dense, axes=[1, 0, 2])

    return preds_dense

def from_upper_triu(vector_repr, matrix_len, num_diags):
        z = np.zeros((matrix_len,matrix_len))
        triu_tup = np.triu_indices(matrix_len,num_diags)
        z[triu_tup] = vector_repr
        for i in range(-num_diags+1,num_diags):
            set_diag(z, np.nan, i)
        return z + z.T

def get_data(data_dir, split_label='train'):
    # read data parameters
    data_stats_file = '%s/statistics.json' % data_dir
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)
    hic_diags =  data_stats['diagonal_offset']
    target_crop = data_stats['crop_bp'] // data_stats['pool_width']
    target_length1 = data_stats['seq_length'] // data_stats['pool_width']
    target_length1_cropped = target_length1 - 2*target_crop

    data = dataset.SeqDataset(data_dir, split_label, batch_size=8)
    inputs, targets = data.numpy(return_inputs=True, return_outputs=True)

    return inputs, targets, hic_diags, target_length1_cropped

def show_targets(data_dir, split_label='train', sample_indices=[]):
    inputs, targets, hic_diags, target_length1_cropped = get_data(data_dir, split_label)

    # plot target 
    vmin=-2; vmax=2

    num_plots = len(sample_indices)
    num_rows = np.ceil(num_plots / 4).astype(int) 

    for i, sample_index in enumerate(sample_indices):
        plt.subplot(num_rows, 4, i+1)
        mat = from_upper_triu(targets[sample_index:sample_index+1,:,:][:,:,0], target_length1_cropped, hic_diags)
        im = plt.matshow(mat, fignum=False, cmap='RdBu_r', vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad=0.05, ticks=[-2, -1, 0, 1, 2])
        plt.title(f'target{sample_index+1}')

    plt.tight_layout()

def show_predictions(predictions, hic_diags, target_length1_cropped):
    vmin=-2; vmax=2

    num_plots = len(predictions)
    num_rows = np.ceil(num_plots / 4).astype(int) 

    for i, pred in enumerate(predictions):
        plt.subplot(num_rows, 4, i+1)
        mat = from_upper_triu(pred[:,:,0], target_length1_cropped, hic_diags)
        im = plt.matshow(mat, fignum=False, cmap='RdBu_r', vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad=0.05, ticks=[-2, -1, 0, 1, 2])
        plt.title(f'pred{i+1}')

    plt.tight_layout()


def symmertic_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open, nproc=1, map=map):
    """
    Generate sequences with symmetric insertions for a given set of coordinates.

    This generator function takes a DataFrame `seq_coords_df` containing genomic
    coordinates, a list of background sequences `background_seqs`, and a genome file
    handler `genome_open`. It yields one-hot encoded DNA sequences with symmetric
    insertions based on the specified coordinates.

    Parameters:
    - seq_coords_df (pandas.DataFrame): DataFrame with columns 'chrom', 'start', 'end',
                                         'strand', 'flank_bp', 'spacer_bp', 'orientation'.
                                         Represents genomic coordinates and insertion parameters.
    - background_seqs (List[numpy.ndarray]): List of background sequences to be modified.
    - genome_open (GenomeFileHandler): A file handler for the genome to fetch sequences.

    Yields:
    numpy.ndarray: One-hot encoded DNA sequence with symmetric insertions.
    """
    
    for s in seq_coords_df.itertuples():

        flank_bp = s.flank_bp
        spacer_bp = s.spacer_bp
        orientation_string = s.orientation

        seq_1hot_insertion = dna_1hot(
            genome_open.fetch(
                s.chrom, s.start - flank_bp, s.end + flank_bp
            ).upper()
        )

        if s.strand == "-":
            seq_1hot_insertion = hot1_rc(seq_1hot_insertion)
            # now, all motifs are standarized to this orientation ">"

        seq_1hot = background_seqs[s.background_index].copy()

        seq_1hot = _insert_casette(
            seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string
        )

        yield seq_1hot    


def _insert_casette(
    seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string
    ):
    """
    Insert a casette sequence into a given one-hot encoded DNA sequence.

    This function takes a one-hot encoded DNA sequence `seq_1hot`, an insertion
    sequence `seq_1hot_insertion`, the number of base pairs for intert-insert spacers
    `spacer_bp`, and an orientation string `orientation_string` specifying the orientation
    and number of insertions. It inserts the given casette sequence into the original sequence
    based on the specified orientations and returns the modified sequence.

    Parameters:
    - seq_1hot (numpy.ndarray): One-hot encoded DNA sequence to be modified.
    - seq_1hot_insertion (numpy.ndarray): One-hot encoded DNA sequence to be inserted.
    - spacer_bp (int): Number of base pairs for intert-insert spacers.
    - orientation_string (str): String specifying the orientation and number of insertions.
                               '>' denotes forward orientation, and '<' denotes reverse.

    Returns:
    numpy.ndarray: One-hot encoded DNA sequence with the casette insertion.

    Raises:
    AssertionError: If the insertion offset is outside the valid range or if the length
                    of the insert and inter-insert spacing leads to an invalid offset.
    """
    seq_length = seq_1hot.shape[0]
    insert_bp = len(seq_1hot_insertion)
    num_inserts = len(orientation_string)

    insert_plus_spacer_bp = insert_bp + 2 * spacer_bp
    multi_insert_bp = num_inserts * insert_plus_spacer_bp
    insert_start_bp = seq_length // 2 - multi_insert_bp // 2

    output_seq = seq_1hot.copy()
    insertion_starting_positions = []
    for i in range(num_inserts):
        offset = insert_start_bp + i * insert_plus_spacer_bp + spacer_bp
        insertion_starting_positions.append(offset)

        assert (
            offset >= 0 and offset < seq_length - insert_bp
        ), f"offset = {offset}. Please, check length of insert and inter-insert spacing."

        for orientation_arrow in orientation_string[i]:
            if orientation_arrow == ">":
                output_seq[offset : offset + insert_bp] = seq_1hot_insertion
            else:
                output_seq[offset : offset + insert_bp] = hot1_rc(
                    seq_1hot_insertion
                )

    return output_seq


def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    """dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
      n_sample:  sample ACGT for N
    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype="float16")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="bool")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                elif n_sample:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1

    return seq_code


def hot1_rc(seqs_1hot):
    """Reverse complement a batch of one hot coded sequences,
    while being robust to additional tracks beyond the four
    nucleotides."""

    if seqs_1hot.ndim == 2:
        singleton = True
        seqs_1hot = np.expand_dims(seqs_1hot, axis=0)
    else:
        singleton = False

    seqs_1hot_rc = seqs_1hot.copy()

    # reverse
    seqs_1hot_rc = seqs_1hot_rc[:, ::-1, :]

    # swap A and T
    seqs_1hot_rc[:, :, [0, 3]] = seqs_1hot_rc[:, :, [3, 0]]

    # swap C and G
    seqs_1hot_rc[:, :, [1, 2]] = seqs_1hot_rc[:, :, [2, 1]]

    if singleton:
        seqs_1hot_rc = seqs_1hot_rc[0]

    return seqs_1hot_rc

