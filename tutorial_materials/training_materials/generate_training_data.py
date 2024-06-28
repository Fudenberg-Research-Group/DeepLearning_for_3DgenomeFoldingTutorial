#!/usr/bin/env python
from optparse import OptionParser
import os
from tqdm import tqdm

import numpy as np
from scipy.stats import beta, zscore
import tensorflow as tf

from basenji.dna_io import dna_1hot
from cooltools.lib.numutils import interp_nan

np.random.seed(39)

'''
Name

Description...
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    # out_dir = 'squares_valid'
    out_dir = '/content/akita_tutorial/tutorial_materials/training_materials/tfrecords'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    seq_length = 32768 #2^15
    bin_size = 512
    seq_bins = seq_length // bin_size
    split_label = 'valid'
    diagonal_offset = 2

    triu_tup = np.triu_indices(seq_bins, diagonal_offset)


    seqs_per_tfr = 32 #batch size
    if split_label == 'train': 
        num_seqs = 32 * 80
    if split_label == 'valid': 
        num_seqs = 32 * 20        
    num_tfr = num_seqs // seqs_per_tfr

    ### define motif 
    ctcf_consensus = ['C','C','G','C','G','A','G','G','T','G','G','C','A','G']
    ctcf_revcomp   = ['C','T','G','C','C','A','C','C','T','C','G','C','G','G']
    ctcf_consensus= np.array(ctcf_consensus)
    ctcf_revcomp = np.array(ctcf_revcomp)
    motif_len = len(ctcf_consensus)
    spacer_len = 10

    # define options
    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

    for ti in tqdm(range(num_tfr)):
        tfr_file = '%s/%s-%d.tfr' % (out_dir, split_label, ti)

        with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:

            for si in range(seqs_per_tfr):

                num_boundaries = np.random.randint(4,8)
                boundary_positions = np.sort(np.random.choice(np.arange(
                                           motif_len +spacer_len//2 +1, seq_length -motif_len -spacer_len//2), num_boundaries,replace=False) )
                boundary_positions = np.array( [0] + list(boundary_positions) + [seq_length])

                # create a random mask
                maskMatrix = np.ones((seq_bins,seq_bins))
                maskMatrix = maskMatrix.astype('float16')
                maskMatrix = maskMatrix[triu_tup].reshape((-1, 1))

                

                targetMatrix = np.zeros((seq_bins,seq_bins))
                for i in range(len(boundary_positions)-1):
                    s = boundary_positions[i] //bin_size
                    e = boundary_positions[i+1] //bin_size
                    targetMatrix[ s:e,s:e] = 1
                    if s + 4 < seq_bins and e - 4 > 0:
                        targetMatrix[(s+2):(e-2),(s+2):(e-2)] = 0.75
                        targetMatrix[(s+4):(e-4),(s+4):(e-4)] = 0.5
                
                seq_dna = np.random.choice(['A','C','G','T'], size=seq_length, p= [.25,.25,.25,.25])
                for i in range(1,len(boundary_positions)-1):
                    seq_dna[boundary_positions[i]-motif_len - spacer_len//2:boundary_positions[i]- spacer_len//2 ]  = ctcf_consensus
                    seq_dna[boundary_positions[i] + spacer_len//2: boundary_positions[i] + motif_len + spacer_len//2 ]  =ctcf_revcomp


                # collapse list
                seq_dna = ''.join(seq_dna)

                # 1 hot code
                seq_1hot = dna_1hot(seq_dna)


                # compute targets
                seq_targets = targetMatrix.astype('float16')
                seq_targets = seq_targets[triu_tup].reshape((-1, 1))


                # make example
                example = tf.train.Example(features=tf.train.Features(feature={
                  'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
                  'target': _bytes_feature(seq_targets[:, :].flatten().tostring()),
                  'mask': _bytes_feature(maskMatrix[:, :].flatten().tostring())
                  }))

                # write example
                writer.write(example.SerializeToString())

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main() 