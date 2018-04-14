
# coding: utf-8

# In[56]:


import numpy as np
import argparse
import json


# In[57]:


#returns argument parser
def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='iid',
                        help='the type of data that needs to be generated')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='length of the sequence to be generated')
    parser.add_argument('--markovity', type=int, default=30,
                        help='steps for xentropy or hmm data')
    parser.add_argument('--file_name', type=str, default='../../Data/Original Data/input.txt',
                        help='output file name')
    parser.add_argument('--info_file', type=str, default='../../Data/Original Data/input_info.txt',
                        help='info file name')
    parser.add_argument('--p', type=float, default=0.5,
                        help='the probability for the entire sequence (for iid)\
                        or the base (for xentropy or hmm)')
    parser.add_argument('--n', type=float, default=0.0,
                        help='the probability for the base for hmm')
    
    return parser


# In[58]:


# Computes the binary entropy
def entropy_iid(prob):
    p1 = prob
    p0 = 1.0 - prob
    try: #in case p1 is 1.0 or 0.0, except block handles
        H = -(p1*np.log2(p1) + p0*np.log2(p0))
    except:
        H = 0
    return H


# In[61]:


def main():
    parser = get_argument_parser()
    FLAGS = parser.parse_known_args()[0]
    print FLAGS
    _keys = ['data_type', 'p', 'entropy'] #for the info file

    #initialise empty data
    data = np.empty(FLAGS.num_samples, dtype='S1')
    
    #Generate data
    if FLAGS.data_type=='iid':
        data = np.random.choice(['a', 'b'], size=FLAGS.num_samples, p=[1-FLAGS.p, FLAGS.p])
        FLAGS.entropy = entropy_iid(FLAGS.p)
 
    elif FLAGS.data_type=='xentropy':
        data[:FLAGS.markovity] = np.random.choice(['a', 'b'], size=FLAGS.markovity, p=[1-FLAGS.p, FLAGS.p])
        for i in range(FLAGS.markovity, FLAGS.num_samples):
            if data[i-1] == data[i-FLAGS.markovity]:
                data[i] = 'a'
            else:
                data[i] = 'b'
        FLAGS.entropy = 0
        _keys.append('markovity')
    
    elif FLAGS.data_type=='hmm':
        data[:FLAGS.markovity,:] = np.random.choice(['a', 'b'], size=(FLAGS.markovity,1), p=[1-FLAGS.p, FLAGS.p])
        for i in range(FLAGS.markovity, FLAGS.num_samples):
            if data[i-1] == data[i-FLAGS.markovity]:
                data[i] = np.random.choice(['a','b'], p=[1-FLAGS.n, FLAGS.n])
            else:
                data[i] = np.random.choice(['b','a'], p=[1-FLAGS.n, FLAGS.n])
  
        FLAGS.entropy = entropy_iid(FLAGS.n) 
        _keys.append('n')
        _keys.append('markovity')
        
    print 'Data Generated'
    np.savetxt(FLAGS.file_name, data, delimiter='', fmt='%s', newline='');
    print 'Data saved in file: ' + FLAGS.file_name
    
    args = vars(FLAGS)
    info = {key : args[key] for key in _keys }
    with open(FLAGS.info_file, 'wb') as f:
        json.dump(info,f)
    print 'Info file generated: ' + FLAGS.info_file


# In[60]:


main()

