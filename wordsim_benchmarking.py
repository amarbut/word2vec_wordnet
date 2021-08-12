# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:32:07 2021

@author: Anna
"""

import numpy as np
from scipy import stats
import argparse

def make_emb_dict(emb_file, word_file):
    emb_dict = dict()
    
    with open(emb_file, "r") as ef:
        with open(word_file, "r") as wf:
            for emb in ef.readlines():
                w = word_file.readline().strip()
                emb_split = emb.strip().split("\t")
                emb_dict[w] = emb_split
    return emb_dict
                
def make_sim_dict(wordsim_file, wordsim_task):
    sim_dict = dict()
    if wordsim_task = 'ws353':
        with open(wordsimfile, 'r') as f:
            for i,line in enumerate(f.readlines()):
                if i <11:
                    continue
                else:
                    cl, w1, w2, rank = line.lower().split()
                    sim_dict[(w1,w2)] = rank
    elif wordsim_task = 'simlex':
        with open (wordsim_file, 'r') as f:
            for i,line in enumerate(f.readlines()):
                if i<2:
                    continue
                else:
                    row = line.split()
                    w1 = row[0]
                    w2 = row[1]
                    score = row[3]
                    sim_dict[(w1,w2)] = score
    else:
        raise Exception("Similarity task not supported")
        
    return sim_dict

def compare_similarity(emb_dict, sim_dict):
    cos_sim = []
    gold_sim = []
    
    for w1,w2 in sim_dict:
        if w1 in emb_dict and w2 in emb_dict:
            A = emb_dict[w1]
            B = emb_dict[w2]
            cos_sim.append(np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B)))
            gold_sim.append(simdict[(w1,w2)])
        
    return stats.spearmanr(cos_sim, gold_sim)

def benchmark_embeddings(emb_file, word_file, wordsim_file, wordsim_task):
    
    emb_dict = make_emb_dict(emb_file, word_file)
    sim_dict = make_sim_dict(wordsim_file, wordsim_task)
    return compare_similarity(emb_dict, sim_dict)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file',  help = '.txt file with trained embeddings', required = True)
    parser.add_argument('--word_file',  help = '.txt file with word list for trained embeddings', required = True)
    parser.add_argument('--wordsim_file',  help = 'benchmarking task file', required = True)
    parser.add_argument('--wordsim_task',  help = 'benchmarking task to perform; either ws353 or simlex', required = True)
    
    args = vars(parser.parse_args())
    benchmark_embeddings(**args)
    
    
        
    
                

    