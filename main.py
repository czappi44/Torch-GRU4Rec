import os
import argparse
from Modules import Tools, Dataset, Evaluation
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="")
parser.add_argument("--train_path", type=str, default="")
parser.add_argument("--valid_path", type=str, default="")
parser.add_argument("--test_path", type=str, default="")
parser.add_argument("--test", action="store_true")
parser.add_argument('--m', '--measure', type=int, nargs='+', default=[20])
parser.add_argument("--model_path", type=str, default="")

parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--loss", type=str, default='nll') #'bprmax'
parser.add_argument("--optimizer", type=str, default='adagrad')
parser.add_argument("--lr", type=float)

parser.add_argument("--embedding_size", type=int, default=-1)
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--n_layers", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--n_sample", type=int, default=2048)
parser.add_argument("--valid_n_sample", type=int, default=0)
parser.add_argument("--dropout_p_embed", type=float, default=0.0)
parser.add_argument("--dropout_p_hidden", type=float, default=0.0)
parser.add_argument("--final_act", type=str, default='softmaxlogit')#'elu-0.5'
parser.add_argument("--bpreg", type=float, default=1.0)
parser.add_argument("--sample_alpha", type=float, default=1.0)
parser.add_argument("--init_as_normal", type=bool, default=False)
parser.add_argument("--sigma", type=float, default=0.0)

parser.add_argument("--session_key", type=str, default="SessionId")
parser.add_argument("--item_key", type=str, default="ItemId")
parser.add_argument("--time_key", type=str, default="Time")
parser.add_argument("--sep", type=str, default='\t')
parser.add_argument("--use_cuda", type=bool, default=True)
######
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--time_sort", type=bool, default=True)
parser.add_argument("--train_random_order", type=bool, default=False)
######


if __name__ == "__main__":
    args = parser.parse_args()
    print(pd.DataFrame({'Args':list(args.__dict__.keys()), 'Values':list(args.__dict__.values())}))
    
    if not args.test:
        train_dataset = Dataset.Dataset(args.train_path, sessionKey=args.session_key, itemKey=args.item_key, timeKey=args.time_key, sep=args.sep)
        valid_dataset = None
        
        input_size = train_dataset.nItems 
        output_size = input_size 
        Tools.fitAndEvalute(train_dataset, valid_dataset, args.save_path, input_size, output_size, args.loss, args.final_act, args.n_layers, args.hidden_size, args.n_epochs, args.batch_size, True, args.dropout_p_hidden, args.dropout_p_embed, args.lr, args.momentum, args.weight_decay, args.embedding_size, args.n_sample, args.valid_n_sample, args.sample_alpha, args.optimizer, args.bpreg, args.sigma, args.init_as_normal, args.train_random_order, args.time_sort, None, args.session_key, args.item_key, args.time_key, args.use_cuda)
    else:
        train_dataset = Dataset.Dataset(args.train_path, sessionKey=args.session_key, itemKey=args.item_key, timeKey=args.time_key, sep=args.sep)
        test_dataset = Dataset.Dataset(args.test_path, sessionKey=args.session_key, itemKey=args.item_key, timeKey=args.time_key, sep=args.sep, itemMap=train_dataset.itemMap)
        test_data_generator = Dataset.DataGenerator(test_dataset, batchSize=64, nSample=0, sampleAlpha=0.0, timeSort=False, trainRandomOrder=False)
        checkpoint = torch.load(args.model_path)

        print("Test results")
        print(f"recall\tmrr\t@{args.m}")
        for k in args.m:
            model = checkpoint["model"]
            model.gru.flatten_parameters()
            evaluation = Evaluation.Evaluation(model, k=k)
            _, recall, mrr = evaluation.evalute(test_data_generator)
            print(f'Recall@{k}: {recall:.8} MRR@{k}: {mrr:.8}')