#!/usr/bin/python

import subprocess
import numpy as np
import os


def main():

	gen_dir = '../Data Generator/'
	train_dir = '../RNN/'
	data_dir = '../../Data/Generated Data/'

    generation_script = gen_dir + 'generate_sequence_data.py'
    train_script = train_dir + 'char_feature_rnn.py'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # generate and run xentropy models with default parameters
    max_k=200
    num_samples=100000
    validate_samples=100
    max_epochs=2
    num_iter=1
    num_layers=2
    num_features=100
    
    p=0.5
    for k in range(100, max_k, 10):
        for iter in range(num_iter):
            print  'Processing for k: ', str(k)
            markovity = k
            file_name = data_dir + 'input_xentropy_iter_' + str(iter) + '_markovity_' + str(markovity) + '.txt'
            info_file = data_dir + 'info_xentropy_iter_' + str(iter) + '_markovity_' + str(markovity) + '.txt'
            val_name = data_dir + 'validate_xentropy_iter_' + str(iter) + '_markovity_' + str(markovity) + '.txt'
            
            ### Generate validation data first
            arg_string  = '  --num_samples ' + str(validate_samples)
            arg_string += '  --data_type '   + 'xentropy'
            arg_string += '  --markovity '   + str(markovity)
            arg_string += '  --file_name '   + val_name
            arg_string += '  --info_file '   + info_file    
            arg_string += '  --p '           + str(p)

            generation_command = 'python2 ' + generation_script + arg_string
            subprocess.call([generation_command] , shell=True)

            ### Generate the complete data
            arg_string  = '  --num_samples ' + str(num_samples)
            arg_string += '  --data_type '   + '0entropy'
            arg_string += '  --markovity '   + str(markovity)
            arg_string += '  --file_name '   + file_name
            arg_string += '  --info_file '   + info_file 
            arg_string += '  --p1 '          + str(p1)

            generation_command = 'python2 ' + generation_script + arg_string
            subprocess.call([generation_command] , shell=True)
            
            assert os.path.isfile(file_name), 'The complete data did not get generated'
            assert os.path.isfile(val_name), 'The validation data did not get generated'
            assert os.path.isfile(info_file), 'The info file did not get created'


            #### Prepare for training
            for _size in [128]:
                summary_dir = '.summary_azure/big_model/'
                summary_dir = os.path.join(summary_dir, 'size_' + str(_size))
                summary_dir = os.path.join(summary_dir, 'num_layers_' + str(num_layers))
                summary_dir = os.path.join(summary_dir, 'markovity_' + str(k))
                summary_dir = os.path.join(summary_dir, 'features_' + str(num_features))
                summary_dir = os.path.join(summary_dir, 'run_' + str(iter))
                arg_string  = ' --data_path '   + file_name
                arg_string += ' --info_path '   + info_file
                arg_string += ' --validate_path '    + val_name
                #arg_string += ' --output_path ' + output_file
                arg_string += ' --num_epochs '  + str(max_epochs)
                arg_string += ' --num_layers '  + str(num_layers)
                arg_string += ' --hidden_size '    + str(_size)
                arg_string += ' --feature_size '    + str(num_features)
                arg_string += ' --summary_path ' + str(summary_dir)
                # Run the training
                train_command = 'python2 ' + train_script + arg_string
                subprocess.call([train_command], shell=True) 

if __name__ == '__main__':
    main()

