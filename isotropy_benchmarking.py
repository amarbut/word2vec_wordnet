#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:07:34 2022

@author: anna
"""
from sklearn.decomposition import PCA
import numpy as np
from math import exp
import argparse


def pc_isotropy(embeddings):
    #per Mu 2017, take ratio of dot product of max/min principal component with all words in vocab as a measure of isotropy
    pc = PCA()
    pc.fit(embeddings)
    max_pc = pc.components_[0]
    min_pc = pc.components_[-1]
    max_sum = 0
    min_sum = 0
    for emb in embeddings:
        max_sum += exp(np.dot(max_pc, emb))
        min_sum += exp(np.dot(min_pc, emb))
        
    return min_sum/max_sum
        
    
def avg_cos(embeddings, num_sample = 1000):
    #per Bihani 2021, avg cosine sim over 1k uniformly randomly chosen words from vocab--let user choose number to sample
    np.random.shuffle(embeddings)
    samples = embeddings[0:num_sample]
    cos_dist = []
    for i in range(num_sample):
        for j in range(i,num_sample):
            cos_dist.append(np.dot(samples[i],samples[j]))
            
    return np.mean(cos_dist)

def load_embeddings(embedding_file):
    emb = np.genfromtxt(embedding_file)
    return emb

def compute_isotropy_measures(embedding_file, num_sample = 1000):
    model_name = embedding_file.split("/")[-2]
    embeddings = load_embeddings(embedding_file)
    avg_cos_dist = avg_cos(embeddings)
    pc_iso = pc_isotropy(embeddings)
    print(model_name+"\t"+str(avg_cos_dist)+"\t"+str(pc_iso)+"\n")
    

#--------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', help = 'tab delimited file with word embeddings', default = None,required = False)
    parser.add_argument('--num_sample',  help = 'number of words to sample for avg cosine similarity', default = 1000, required = False)
    args = vars(parser.parse_args())
    compute_isotropy_measures(**args)
    
    
        
                