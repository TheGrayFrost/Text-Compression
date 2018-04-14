''' Based partially on CS224n assignment 3
'''
import argparse
import tensorflow as tf
from sequence_pred_feature_model import SequencePredFeature
from feature_model_trainer import ModelTrainer

def get_argument_parser():
    
    parser = argparse.ArgumentParser()

    #file io parameters
    parser.add_argument('--data_path', type=str, default='../../Data/Original Data/input.txt',
                        help='data directory containing input.txt')
    parser.add_argument('--info_path', type=str, default=None,
                       help='Information about the input file')
    parser.add_argument('--validate_path', type=str, default='../../Data/Original Data/validate.txt')                         

    #data parameters
    parser.add_argument('--vocab', type=str, default='ab')
    parser.add_argument('--entropy', type=float, default=0)

    #rnn parameters
    parser.add_argument('--feature_size', type=int, default=10,
                       help='feature size')
    parser.add_argument('--hidden_size', type=int, default=32,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model_type', type=str, default='gru')
    parser.add_argument('--regularization', type=str, default='dropout')
    parser.add_argument('--drop_prob', type=float, default=0.0)

    #training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for the training')
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--validate_every', type=int, default=1000)
    parser.add_argument('--summary_path', type=str, default='../../Summary',
                       help='directory to store tf summary')

    return parser

def get_config_args():
    parser = get_argument_parser()
    config = parser.parse_args()
    config.max_length = config.batch_size
    config.num_classes = len(config.vocab)
    return config

def main():
    config = get_config_args()
    GRUModel = SequencePredFeature(config);
    Trainer = ModelTrainer(config, GRUModel)
    Trainer.do_training();
    
if __name__ == '__main__':
    main()
