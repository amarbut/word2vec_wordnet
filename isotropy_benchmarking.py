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
    num = len(embeddings[0])
    sums = []
    for c in pc.components_:
        c_sum = 0
        for emb in embeddings:
            c_sum += exp(np.dot(c,emb)/((np.lingalg.norm(c)+1e-5)*(np.linalg.norm(emb)+1e-5)))
        sums.append(c_sum)
    
    sums.sort()
    max_sum = sums[-1]
    min_sum = sums[0]
    sum_75 = sums[round(num*0.25)-1]
    sum_50 = sums[round(num*0.5)-1]
    sum_25 = sums[round(num*0.75)-1]
    
        
    pcr_100 = min_sum/max_sum
    pcr_75 = sum_75/max_sum
    pcr_50 = sum_50/max_sum
    pcr_25 = sum_25/max_sum
        
    return [pcr_100, pcr_75, pcr_50, pcr_25]
        
    
def avg_cos(embeddings, num_sample = 1000):
    #per Bihani 2021, avg cosine sim over 1k uniformly randomly chosen words from vocab--let user choose number to sample
    np.random.shuffle(embeddings)
    samples = embeddings[0:num_sample]
    cos_dist = []
    for i in range(num_sample):
        for j in range(i,num_sample):
            cos_dist.append(np.dot(samples[i],samples[j])/((np.linalg.norm(samples[i])+0.00001)*(np.linalg.norm(samples[j])+0.00001)))
            
    return np.mean(cos_dist)

def load_embeddings(embedding_file):
    emb = np.genfromtxt(embedding_file)
    return emb

def compute_isotropy_measures(embedding_file, dir_name, num_sample = 1000):
    embeddings = load_embeddings(embedding_file)
    avg_cos_dist = avg_cos(embeddings)
    pc_100, pc_75, pc_50, pc_25 = pc_isotropy(embeddings)
    model_name = dir_name.split("/")[-2]
    print("\t".join([model_name, str(avg_cos_dist), str(pc_100), str(pc_75), str(pc_50), str(pc_25)])+"\n")
    

#--------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', help = 'tab delimited file with word embeddings', default = None,required = True)
    parser.add_argument('--dir_name', help = 'name of directory holding embeddings, used to extract model name', default = None, required = True)
    parser.add_argument('--num_sample',  help = 'number of words to sample for avg cosine similarity', default = 1000, required = False)
    args = vars(parser.parse_args())
    compute_isotropy_measures(**args)
    
    
        
                