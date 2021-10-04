# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:29:59 2021

@author: Anna
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import argparse
import os
import numpy as np
import pickle
import json
from nltk.corpus import wordnet as wn

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, train_dir, input_file_name, wn, wn_depth):


        self.word2id = dict()
        self.id2word = dict()
        self.token_count = 0
        self.sentence_count = 0
        self.word_frequency = dict()
        
        self.negative_probs = []
        self.subsample_probs = []
        
        self.wn = wn
        self.wn_depth = wn_depth
        self.wn_synset2id = dict()
        self.wn_id2synset = dict()
        self.wn_synset2word = dict()
        self.wn_word2synset = dict()        
        self.wn_synset_frequency = dict()
        
        self.wn_negative_probs = []

        self.input_file_name = os.path.join(train_dir, input_file_name)
        self.read_words()
        self.init_negative_probs()
        self.init_subsample_probs()
        if self.wn == True:
            self.init_wn_negative_probs()
        

    #get term counts over entire dataset    
    def read_words(self, ):
        
        word_frequency = dict()
        for line in open(self.input_file_name, encoding = 'utf8'):
            line = line.split()
            self.sentence_count += 1
            if len(line) > 1:
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print(str(int(self.token_count / 1000000)), "M tokens ingested.")

        
        #start word ids at 1 so that 0 can be used for tensor padding
        wid = int(1)
        self.id2word[0] = ''
        self.word_frequency[0] = 0
        
        
        for w, c in word_frequency.items():
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total unique words: ", str(len(self.word2id)))
        
        if self.wn == True:
            print("Creating Wordnet Synsets")
        #CREATE WORDNET LOOKUP FOR VOCABULARY 
            #Include synset members and two levels of hyponym members, if in vocabulary
            #One dict for {synset: [members + hyponyms]}, one dict for {word:[synsets]}
            sid = 0
            for word in self.word2id:
                ss = wn.synsets(word)
                for synset in ss:
                    s = synset.name()
                    sw = s.split(".")[0]
                    #skip synset if already added or if not in vocab
                    if not s in self.wn_synset2id and sw in self.word2id and len(s.split(".")) == 3:
                        syns = [self.word2id[w.lower()] for w in synset.lemma_names() if w.lower() in self.word2id]
                        
                        #add 1st level of hyponyms
                        if self.wn_depth > 0: 
                            syns.extend([self.word2id[w.lower()] for h in s.hyponyms() for w in h.lemma_names() if w.lower() in self.word2id])
                        
                        #add 2nd level of hyponyms
                        if self.wn_depth > 1:
                            syns.extend([self.word2id[w.lower()] for h in s.hyponyms() for hh in h.hyponyms() for w in hh.lemma_names if w.lower() in self.word2id])
                        
                        self.wn_synset2id[s] = sid
                        self.wn_synset2word[sid] = set(syns)
                        self.wn_id2synset[sid] = s
                        for syn in syns:
                            if not syn in self.wn_word2synset:
                                self.wn_word2synset[syn] = set([sid])
                            else:
                                self.wn_word2synset[syn].add(sid)
                        sid +=1
                        
            self.wn_synset_frequency = {k:sum([self.word_frequency[w] for w in self.wn_synset2word[k]])/self.token_count for k in self.wn_synset2word}
        print("Total unique synsets: ", str(len(self.wn_synset2word)))

    #calculate subsample probability according to 2nd Word2Vec paper equation
    def init_subsample_probs(self):
        print("Initiating Subsample Probabilities")
        t = 0.00001
        f = np.array(list(self.word_frequency.values()))/ self.token_count
        self.subsample_probs = np.concatenate((np.array([0]),1-np.sqrt(t / f[1:])))     

    #create array of words to be used in negative samples of size NEGATIVE_TABLE_SIZE; sample with adjusted unigram method
    def init_negative_probs(self):
        print("Initiating Negative Sample Probabilities")
        #use adjusted unigram sampling--raise the word frequencies to the 3/4 power to make less frequent words appear more often
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        
        #calculate how many times ea word (rep by its wid) should appear in the neg sample array based on adjusted freq above
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negative_probs += [wid] * int(c)
        self.negative_probs = np.array(self.negative_probs)
        
        #randomize array of words to be sampled from in data set creation
        np.random.shuffle(self.negative_probs)
            

    #create array of words to be used in negative samples of size NEGATIVE_TABLE_SIZE; sample with adjusted unigram method
    def init_wn_negative_probs(self):
        print("Initiating Wordnet Negative Sample Probabilities")
        #use adjusted unigram sampling at synset level--raise the word frequencies to the 3/4 power to make less frequent words appear more often
        pow_freq = {ss:self.wn_synset_frequency[ss]**0.75 for ss in self.wn_synset_frequency}
        ss_pow = sum(list(pow_freq.values()))
        count_ratio = {k:np.round((v/ss_pow)*DataReader.NEGATIVE_TABLE_SIZE) for k,v in pow_freq.items()}
        
        #calculate how many times ea word (rep by its wid) should appear in the neg sample array based on adjusted freq above
        for k in count_ratio:
            self.wn_negative_probs += [k] * int(count_ratio[k])
        self.wn_negative_probs = np.array(self.wn_negative_probs)
        
        #randomize array of words to be sampled from in data set creation
        np.random.shuffle(self.wn_negative_probs)
        
    #function to sample from negative sampling list        
    def get_negatives(self, pos, size, boundary):
        u,v = pos
        num_negs = size*boundary
        #collect list of neg sample words
        response = []
        while len(response) < num_negs:
            response.extend([i for i in np.random.choice(self.negative_probs, num_negs) if i != u and (i not in v)])
        #return list of neg sample words
        return response[0:num_negs] 
    
    def get_wn_negatives(self, syn, num_negs):
        response = []
        syn_name = self.wn_id2synset[syn]
        pos = syn_name.split('.')[1]
        while len(response) < num_negs:
            #populate wn negative samples with synsets that are not equal to the target, but are of the same pos
            response.extend([i for i in np.random.choice(self.wn_negative_probs,num_negs) if i != syn and
                             self.wn_id2synset[i].split('.')[1] == pos])
        return response[0:num_negs]
    
    
# -------------------------------------------------------------------------------------------------------------------------
        
class Word2VecWordnetDataset:
    def __init__(self, data, window_size, wn_negative_sample = False, wn_positive_sample = False):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.input_file_name, encoding = 'utf-8')
        self.wn_negative_sample = wn_negative_sample
        self.wn_positive_sample = wn_positive_sample

    def __len__(self):
        return self.data.sentence_count
    
    #function used to retrieve samples for each line (target (or wordnet syn), [context], [negative context], [wordnet syn]) 
    def __getitem__(self, idx):
        self.input_file.seek(idx)
        line = self.input_file.readline()
        
        if len(line) > 1:
            words = line.strip().split()

            if len(words) > 1:
                #collect word ids for sentence, ignoring words w/ subsample probability
                word_ids = [self.data.word2id[w] for w in words if w in self.data.word2id]
                
                boundary = int((self.window_size-1)/2)
                
                
                #collect list of all target/positive context pairs
                #PAD LIST SO THAT ALL THE SAME LENGTH WHEN CONVERT TO TENSOR
                pos_pairs = [(u,word_ids[int(max(i - boundary, 0)):int(i + boundary+1)]) for i, u in enumerate(word_ids) if np.random.rand() > self.data.subsample_probs[int(u)]]
                pos_pairs = [(u,v+([0]*((boundary*2)-len(v)+1))) for u,v in pos_pairs]
                if len(pos_pairs) == 0:
                    return([])
                
                #do vanilla word2vec neg sampling without wordnet input
                if not self.wn_negative_sample and not self.wn_positive_sample:
                    
                    negs = [self.data.get_negatives(pos,5, boundary*2) for pos in pos_pairs]
                    wn_pos = [[] for _ in range(len(negs))]
                    sims = [[] for _ in range(len(negs))]
                    not_sims = [[] for _ in range(len(negs))]
                    mismatches = ([[] for _ in range(len(negs))])
                
                #add wordnet to sample as target word or negative sample
                else:
                     
                    if self.wn_positive_sample: #TODO: randomize whether this happens for every word? How many new pos examples are added?
                    #replace target word with wordnet similar words and add to positive examples
                        wn_pairs = []
                        for pos in pos_pairs:
                            u,v = pos
                            syns = [s for ss in self.data.wn_word2synset[u] for s in self.data.wn_synset2word[ss]]
                            
                            #subsample wordnet similar words before adding
                            wn_pairs.extend([(s,v) for s in syns
                                                 if np.random.rand() > self.data.subsample_probs[s]])
                        pos_pairs.extend(wn_pairs)
        
                    if self.wn_negative_sample:
                    #include one wordnet similar word per positive context to provide positive wordnet similarity samples
                    #create wn contrastive loss lists
                        negs = []
                        wn_pos = []
                        sims = []
                        not_sims = []
                        mismatches = []
                        for pos in pos_pairs:
                            u,v = pos
                            
                            #skip if target word doesn't have synset in vocabulary
                            if u in self.data.wn_word2synset:
                                #gets similar words
                                syns = [s for ss in self.data.wn_word2synset[u] for s in self.data.wn_synset2word[ss] if s != u]
                                #skip if no synonyms in vocabulary
                                if len(syns) >0:
                                    #select similar words based on negative sampling probs
                                    syn_probs = np.array([self.data.negative_probs[i] for i in syns])
                                    syn_norm = syn_probs/sum(syn_probs)
                                    
                                    wn_pos_ids = np.random.choice(syns, boundary*2, p = syn_norm)
                                    neg_ids = self.data.get_negatives(pos, 4, boundary*2)
        
                                    wn_pos.append(wn_pos_ids)
                                    negs.append(neg_ids)
                           
                                #sort context and neg sample words by wordnet similarity to target
                                    sim = list(wn_pos_ids)
                                    w2v_mismatch = []
                                    not_sim = []
                                    for j in v:
                                        #skip padded cells
                                        if j != 0 and j in self.data.wn_word2synset:
                                            #mark as similar if u and j share any synset groups
                                            if len(self.data.wn_word2synset[u]-self.data.wn_word2synset[j]) < len(self.data.wn_word2synset[u]):
                                                sim.append(j)
                                            else:
                                                w2v_mismatch.append(j)
                                    for k in neg_ids:
                                        #skip padded cells in tensors
                                        if k != 0 and k in self.data.wn_word2synset:
                                            #mark as similar if u and k share any synset groups
                                            if len(self.data.wn_word2synset[u]-self.data.wn_word2synset[k]) < len(self.data.wn_word2synset[u]):
                                                sim.append(k)
                                            else:
                                                not_sim.append(k)
                                    
                                    sims.append(sim)
                                    not_sims.append(not_sim)
                                    mismatches.append(w2v_mismatch)
                                else:
                                    negs.append(self.data.get_negatives(pos, 4, boundary*2))
                                    wn_pos.append([0]*boundary*2)
                                    sims.append([])
                                    not_sims.append([])
                                    mismatches.append([])
                            else:
                                negs.append(self.data.get_negatives(pos, 4, boundary*2))
                                wn_pos.append([0]*boundary*2)
                                sims.append([])
                                not_sims.append([])
                                mismatches.append([])
                    else:
                    #vanilla negative sampling method from 2nd word2vec paper
                        negs = [self.data.get_negatives(pos,5, boundary*2) for pos in pos_pairs]
                        wn_pos = [[] for _ in range(len(negs))]
                        sims = [[] for _ in range(len(negs))]
                        not_sims = [[] for _ in range(len(negs))]
                        mismatches = ([[] for _ in range(len(negs))])
                return([(pair[0],pair[1],negs[i], wn_pos[i], sims[i], not_sims[i], mismatches[i]) for i,pair in enumerate(pos_pairs)])
                

    @staticmethod
    #combine all target, context, and negative samples into tensors for each batch
    def collate_fn(batches):
        if batches:
            batches = [batch for batch in batches if batch and len(batch) > 0]
            
            all_u = [u for batch in batches for u, _, _, _, _, _, _ in batch]
            all_v = [v for batch in batches for _, v, _, _, _, _, _ in batch]
            all_neg = [neg for batch in batches for _, _, neg, _, _, _, _ in batch]
            all_wn = [wn_pos for batch in batches for _, _, _, wn_pos, _, _, _ in batch ]
            
            #pad with zeros for conversion to tensors,must be at least length 2 to preserve tensor math
            all_sim = [sim for batch in batches for _, _, _, _, sim, _, _ in batch]
            sim_len = max([2,max([len(i) for i in all_sim])])
            all_sim = [i+([0]*(sim_len-len(i))) for i in all_sim]
            
            all_not_sim = [not_sim for batch in batches for _, _, _, _, _, not_sim, _ in batch]
            not_sim_len = max([2,max([len(i) for i in all_not_sim])])
            all_not_sim = [i+([0]*(not_sim_len-len(i))) for i in all_not_sim]
            
            all_mismatch = [mismatch for batch in batches for _, _, _, _, _, _, mismatch in batch]
            mismatch_len = max([2,max([len(i) for i in all_mismatch])])
            all_mismatch = [i+([0]*(mismatch_len-len(i))) for i in all_mismatch]
            
            
            t_all_u=torch.LongTensor(all_u)
            t_all_v=torch.LongTensor(all_v)
            t_all_neg=torch.LongTensor(all_neg)
            t_all_wn=torch.LongTensor(all_wn)
            t_all_sim=torch.LongTensor(all_sim)
            t_all_not_sim=torch.LongTensor(all_not_sim)
            t_all_mismatch=torch.LongTensor(all_mismatch)
        
            return(t_all_u, t_all_v, t_all_neg, t_all_wn, t_all_sim, t_all_not_sim, t_all_mismatch)
        
        else:
            return(None)

class WordnetFineTuningDataset:
    def __init__(self, data, num_negs, margin_weight):
        self.data = data
        self.synset_list = list(data.wn_synset2word.keys())
        self.num_negs = num_negs
        self.wn_id2synset = data.wn_id2synset
        self.wn_synset2word = data.wn_synset2word
        self.margin_weight = margin_weight
    
    def __len__(self):
        return len(self.synset_list)
    
    #function used to retrieve contrastive synset samples
    def __getitem__(self,idx):
        
        target = self.synset_list[idx]
        
        #get specified # of other synsets for comparison in contrastive loss function
        negs = self.data.get_wn_negatives(target, self.num_negs)
        
        #get synset group word ids
        syn_words = list(self.wn_synset2word[target])
        #get neg synset group word ids
        neg_words = [list(self.wn_synset2word[neg]) for neg in negs]
        
        #get actual synset name to calculate wn distance
        syn = self.wn_id2synset[target]
        #get actual synset name to calculate wn distance    
        neg_syns = [self.wn_id2synset[n] for n in negs]
        
        #use wn distance between target synset and neg synset to set contrastive loss margins
        margins = [int(self.margin_weight*(min((wn.synset(syn).shortest_path_distance(wn.synset(neg)) or 10),10))) for neg in neg_syns]
        
        return {'margins': margins, 'syn_words':syn_words, 'neg_words':neg_words}
    
    @staticmethod
    def collate_fn(batches):
        #pad batched word lists and convert to tensors
        all_syn_words = [batch['syn_words'] for batch in batches]
        syn_len = max([2,max([len(i) for i in all_syn_words])])
        all_syn_words = [i+([0]*(syn_len-len(i))) for i in all_syn_words]
        
        all_neg_words = [batch['neg_words'] for batch in batches]
        neg_len = max([2,max([len(j) for i in all_neg_words for j in i])])
        all_neg_words = [[j+([0]*(neg_len-len(j))) for j in i] for i in all_neg_words]
        
        all_margins = [batch['margins'] for batch in batches]
        margin_len = max([2,max([len(i) for i in all_margins])])
        all_margins = [i+([0]*(margin_len-len(i))) for i in all_margins]
        
        t_all_syn_words = torch.LongTensor(all_syn_words)
        t_all_neg_words = torch.LongTensor(all_neg_words)
        t_all_margins = torch.LongTensor(all_margins)
        
        return (t_all_syn_words, t_all_neg_words, t_all_margins)

#---------------------------------------------------------------------------------------------------------------------------

class SkipGramWordnetModel(nn.Module):
    def __init__(self, vocab_size, emb_dimension, wn_negative_sample):
        super(SkipGramWordnetModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.wn_negative_sample = wn_negative_sample
        

        #initialize target and context embeddings with vocab size and embedding dimension
        self.u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True, padding_idx = 0)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True, padding_idx = 0)
        
        initrange = 1.0 / self.emb_dimension
        #initialize target embeddings with random nums between -/+ initrange
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        #reset padding index (0) to all 0 values
        self.u_embeddings.weight.data[0] = 0
        #initialize context embeddings with zeros--first values will be determined by loss update after first batch
        init.constant_(self.v_embeddings.weight.data, 0)
        
    def forward(self, u, v, neg, wn, sim, not_sim, mismatch, mismatch_weight=1, w2v_loss_weight=1, wn_loss_weight=1, margin = 1):
        #get input embeddings for target, context, and negative samples
        emb_u = self.u_embeddings(u)
        emb_v = self.v_embeddings(v)
        emb_neg = self.v_embeddings(neg)
        emb_wn = self.v_embeddings(wn)
        
       
        #calculate dot product for target and all context words
        w2v_pos_loss = torch.bmm(emb_v,emb_u.unsqueeze(2)).squeeze()
        #disinclude padding "word" loss
        w2v_pos_loss[v != 0] = -F.logsigmoid(w2v_pos_loss[v != 0])
        #use loss function from w2v paper
        w2v_pos_loss = torch.sum(w2v_pos_loss, dim = 1).unsqueeze(1)

        #normalize by number of non-padding "words" in context
        v_mask = v != 0
        w2v_pos_loss /= v_mask.sum(dim = 1).unsqueeze(1)
        
        #calculate dot product for target and all negative sample words
        w2v_neg_loss = torch.bmm(emb_neg,emb_u.unsqueeze(2)).squeeze()
        #use loss function from w2v paper 
        #normalized to match with pos_loss
        w2v_neg_loss = torch.sum(-F.logsigmoid(-w2v_neg_loss), dim = 1).unsqueeze(1)/len(neg[0])
        
        if self.wn_negative_sample:
            #calculate dot product for target and all wn_positive(w2v negative) sample words
            w2v_mismatch_loss = torch.bmm(emb_wn,emb_u.unsqueeze(2)).squeeze()
            #exclude padding "word" loss
            w2v_mismatch_loss[wn != 0] = -F.logsigmoid(-w2v_mismatch_loss[wn != 0])
            # use loss function from w2v paper, downweighted so not in competition with positive wn similarity loss
            # normalized to match pos_loss
            w2v_mismatch_loss = mismatch_weight*(torch.sum(w2v_mismatch_loss,dim = 1)).unsqueeze(1)/len(wn[0])
        else:
            #emb_wn should be empty, produce empty loss            
            w2v_mismatch_loss = torch.bmm(emb_wn,emb_u.unsqueeze(2)).squeeze()
            #can't normalize to avoid division by 0
            w2v_mismatch_loss = mismatch_weight*(torch.sum(-F.logsigmoid(-w2v_mismatch_loss),dim = 1)).unsqueeze(1)
        
        #total w2v loss for all target words
        w2v_loss = w2v_pos_loss + w2v_neg_loss + w2v_mismatch_loss
        
        #add in wordnet similarity contrastive loss
        if self.wn_negative_sample:   
            #get embeddings for wordnet similarity groups
            emb_sim = self.v_embeddings(sim)
            emb_not_sim = self.v_embeddings(not_sim)
            emb_mismatch = self.v_embeddings(mismatch)
            
            #compute euclidean distance
            wn_pos_dist = torch.sqrt(torch.sum(((emb_u.unsqueeze(1)-emb_sim.unsqueeze(0))**2).squeeze(1), dim =3) + 1e-9).squeeze()
            wn_neg_dist = torch.sqrt(torch.sum((emb_u.unsqueeze(1)-emb_not_sim.unsqueeze(0))**2, dim =3) + 1e-9).squeeze()
            wn_mismatch_dist = torch.sqrt(torch.sum((emb_u.unsqueeze(1)-emb_mismatch.unsqueeze(0))**2, dim =3) + 1e-9).squeeze()
            
            #ignore padding "word" distances
            wn_pos_dist2 = torch.clone(wn_pos_dist)
            wn_neg_dist2 = torch.clone(wn_neg_dist)
            wn_mismatch_dist2 = torch.clone(wn_mismatch_dist)
            
            
            wn_pos_dist2[sim == 0] -= wn_pos_dist[sim == 0]
            wn_neg_dist2[not_sim == 0] -= wn_neg_dist[not_sim == 0]
            wn_mismatch_dist2[mismatch == 0] -= wn_mismatch_dist[mismatch == 0]
                       
            #for dissimilar words: replace distance with 0 if larger than contrastive loss margin, or difference from margin if not 
            wn_neg_dist2 = margin - wn_neg_dist2
            wn_neg_dist2[wn_neg_dist2<0] = 0
            
            wn_mismatch_dist2 = margin - wn_mismatch_dist2
            wn_mismatch_dist2[wn_mismatch_dist2<0] = 0
            
            #use distances in contrastive loss function
            wn_pos_loss = torch.sum(0.5*(wn_pos_dist2**2),1).unsqueeze(1)
            wn_neg_loss = torch.sum(0.5*(wn_neg_dist2**2),1).unsqueeze(1)
            #downweight loss from dissimilar words in word2vec context so not competing
            wn_mismatch_loss = mismatch_weight*(torch.sum(0.5*(wn_mismatch_dist2**2),1).unsqueeze(1))
            
            #normalize all loss by number of non-padding "words"
            sim_mask= (sim != 0).sum(dim = 1).unsqueeze(1)
            not_sim_mask = (not_sim != 0).sum(dim = 1).unsqueeze(1)
            mismatch_mask = (mismatch != 0).sum(dim = 1).unsqueeze(1)
        
            
            wn_pos_loss[sim_mask != 0] /= sim_mask[sim_mask != 0]
            wn_neg_loss[not_sim_mask != 0] /= not_sim_mask[not_sim_mask != 0]
            wn_mismatch_loss[mismatch_mask != 0] /= mismatch_mask[mismatch_mask != 0]
            
            #combine wn_pos_loss, wn_neg_loss, wn_mismatch
            wn_loss = wn_pos_loss + wn_neg_loss + wn_mismatch_loss
            
            #return average total loss (wordnet and word2vec), weighted according to parameters
            return torch.mean((wn_loss_weight*wn_loss)+(w2v_loss_weight*w2v_loss))
        
    
        #return average word2vec loss
        return torch.mean(w2v_loss)
    
    def save_embeddings(self, id2word, model_dir):
        emb_file = model_dir+"/w2v_embeddings.txt"
        wordlist_file = model_dir+"/w2v_wordlist.txt"
        model_file = model_dir+"/w2v_model.pth"
        
        embeddings = self.u_embeddings.weight.cpu().data.numpy()
        with open(emb_file, 'w') as ef:
            with open(wordlist_file, "w") as wlf:
                for wid, w in id2word.items():
                    wlf.write(w+'\n')
                    e = '\t'.join([str(emb) for emb in embeddings[wid]])
                    ef.write(e+'\n')
        torch.save(self.state_dict(), model_file)

    
class WordnetFineTuning(nn.Module):
    def __init__(self, wn_id2synset, wn_synset2word, num_negs, margin_weight, vocab_size, emb_dimension):
        super(WordnetFineTuning, self).__init__()
        self.num_negs = num_negs
        self.embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True, padding_idx = 0)
        
    def forward(self, syn_words, neg_words, margins):
        #get word embeddings
        syn_embeddings = self.embeddings(syn_words)
        neg_embeddings = self.embeddings(neg_words)
      
        #use mean of all synset group member embeddings as synset centroid
        #create mask to ignore padding "words" (all 0 values)
        syn_mask1 = syn_embeddings != 0
        syn_mask2 = syn_words != 0
        
        #use average embedding for non-padding "words" as synset centroid
        syn_centroids = torch.sum(syn_embeddings, dim = 1)/syn_mask1.sum(dim = 1)
        
        #compute distance between centroid and synset group members
        syn_dist = torch.sqrt(torch.sum((syn_centroids.unsqueeze(1)-syn_embeddings.unsqueeze(0))**2, dim = 3).squeeze() + 1e-9)
        #ignore padding "word" distances
        syn_dist2 = torch.clone(syn_dist)
        syn_dist2[syn_words == 0] -= syn_dist[syn_words == 0]
        #use contrastive loss to move all words in synset group closer; normalized for number of non-padding words
        syn_pos_loss = torch.sum(0.5*(syn_dist2**2),1)/syn_mask2.sum(dim=1)
        
        
        #use mask to ignore padding "words" (all 0 values)
        neg_mask = neg_embeddings != 0
        
        #use mean of all synset group member embeddings as neg synset group centroid (ignoring padding)
        neg_centroids = torch.sum(neg_embeddings, dim = 2)/neg_mask.sum(dim = 2)
        
        #compute distance between neg group centroids and target group centroids
        neg_dist = torch.sqrt(torch.sum((syn_centroids.unsqueeze(1) - neg_centroids.unsqueeze(0))**2, dim = 3).squeeze() + 1e-9)
        
        #replace neg_dist with difference from margin or 0 if outside of margin
        neg_dist = torch.sub(margins, neg_dist)
        neg_dist[neg_dist<0] = 0
        #use distances in contrastive loss function; normalized by number of neg samples (to better match pos loss)
        neg_loss = torch.sum(0.5*(neg_dist**2),1)/len(neg_centroids[0])
        
        loss = syn_pos_loss + neg_loss
        return(torch.mean(loss))
        

    def save_embeddings(self, id2word, model_dir):
        emb_file = model_dir+"/ft_embeddings.txt"
        wordlist_file = model_dir+"/ft_wordlist.txt"
        model_file = model_dir+"/ft_model.pth"
        
        embeddings = self.embeddings.weight.cpu().data.numpy()
        with open(emb_file, 'w') as ef:
            with open(wordlist_file, "w") as wlf:
                for wid, w in id2word.items():
                    wlf.write(w+'\n')
                    e = '\t'.join([str(emb) for emb in embeddings[wid]])
                    ef.write(e+'\n')
        torch.save(self.state_dict(), model_file)
#--------------------------------------------------------------------------------------------------------------------------------                    

class Word2VecWordnetTrainer:
    def __init__(self, datareader = None, train_dir = None, input_file_name= None, model_dir = None, model_state_dict = None,
                 emb_dimension = 100, batch_size = 32, num_workers = 0, epochs = 3, initial_lr = 0.001, 
                 window_size = 5, wn_negative_sample = False, wn_positive_sample = False, wn_depth = 0,
                 mismatch_weight=1, w2v_loss_weight=1, wn_loss_weight=1, margin = 1,
                 wn_fine_tune = False, ft_margin_weight = 0.1, ft_num_negs = 4, ft_epochs = 15):
        
        self.wn = (wn_negative_sample or wn_positive_sample or wn_fine_tune)
        self.wn_depth = wn_depth
        
        if datareader is not None:
            self.data = pickle.load(open(datareader, "rb"))
        else:
            print("Reading data file")
            self.data = DataReader(train_dir, input_file_name, self.wn, self.wn_depth)
            dr_location = model_dir +"/data_reader.pkl"
            pickle.dump(self.data, open(dr_location, "wb"))
                    
        self.model_dir = model_dir
        self.vocab_size = len(self.data.id2word)
        self.emb_dimension = emb_dimension
        self.initial_lr = initial_lr
        self.model = SkipGramWordnetModel(self.vocab_size, self.emb_dimension, wn_negative_sample)
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print("cuda available:", self.use_cuda)
        if self.use_cuda:
            self.model.cuda()
        
        if model_state_dict is not None:
            self.pretrained = True
            self.model.load_state_dict(torch.load(model_state_dict, map_location = self.device))
        else:
            self.pretrained = False
            dataset = Word2VecWordnetDataset(self.data, window_size, wn_negative_sample, wn_positive_sample)
            self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers,
                                         collate_fn = dataset.collate_fn)
            
            self.batch_size = batch_size
            self.epochs = epochs
            
            self.mismatch_weight = mismatch_weight
            self.w2v_loss_weight = w2v_loss_weight
            self.wn_loss_weight = wn_loss_weight
            self.margin = margin
        
            
        self.wn_fine_tune = wn_fine_tune
        if self.wn_fine_tune:
            self.ft_margin_weight = ft_margin_weight
            self.ft_num_negs = ft_num_negs
            self.ft_epochs = ft_epochs
            
            self.ft_dataset = WordnetFineTuningDataset(self.data, self.ft_num_negs, self.ft_margin_weight)
            self.ft_batch_size = int(len(self.data.wn_synset2id)/1000)
            self.ft_dataloader = DataLoader(self.ft_dataset, batch_size = self.ft_batch_size, shuffle = True, num_workers = num_workers,
                                   collate_fn = self.ft_dataset.collate_fn)
            self.ft_model = WordnetFineTuning(self.data.wn_id2synset, self.data.wn_synset2word,
                                              self.ft_num_negs, self.ft_margin_weight, self.vocab_size, self.emb_dimension)
            if self.use_cuda:
                self.ft_model.cuda()
        
             
            
    def w2v_train(self):
        print("training on cuda: ", self.use_cuda)
        
        optimizer = optim.SparseAdam(self.model.parameters(), lr = self.initial_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 25000, gamma = 0.5)
        #torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.epochs):
            
            print('\nStarting Epoch', (epoch+1))
            
            for i, batch in enumerate(self.dataloader):
                if batch and len(batch[0])>1:
                    u = batch[0].to(self.device)
                    v = batch[1].to(self.device)
                    neg = batch[2].to(self.device)
                    wn = batch[3].to(self.device)
                    sim = batch[4].to(self.device)
                    not_sim = batch[5].to(self.device)
                    mismatch = batch[6].to(self.device)
                    
                    optimizer.zero_grad()
                    loss = self.model.forward(u, v, neg, wn, sim, not_sim, mismatch, self.mismatch_weight, self.w2v_loss_weight, self.wn_loss_weight, self.margin)
                    loss.backward()
                    optimizer.step()
                    
                    if i % 100 == 0:
                        print((i/len(self.dataloader))*100,"% Loss:", loss.item())
                    # if i % 25000 == 0: #based on loss leveling ~5% on full dataset
                    #     scheduler.step()
                    scheduler.step()

        self.model.save_embeddings(self.data.id2word, self.model_dir)        

                    
    def wn_ft(self):
        self.ft_model.embeddings.weight.data.copy_(self.model.u_embeddings.weight.data)
        self.ft_model.to(self.device)
        # set initial learning rate at 1/10 skipgram rate--encourage small adjustments to pre-trained embeddings
        ft_optimizer = optim.SparseAdam(self.ft_model.parameters(), lr = self.initial_lr)
        ft_scheduler = optim.lr_scheduler.StepLR(ft_optimizer, step_size = 5, gamma = 0.1)
        
        for epoch in range(self.ft_epochs):
            print("Starting Epoch:", (epoch+1))
                        
            for i, batch in enumerate(self.ft_dataloader):
                syn_words = batch[0].to(self.device)
                neg_words = batch[1].to(self.device)
                margins = batch[2].to(self.device)

                ft_optimizer.zero_grad()
                loss = self.ft_model.forward(syn_words, neg_words, margins)
                loss.backward()
                ft_optimizer.step()
                                
                if i % 25 == 0:
                    print((i/len(self.ft_dataloader))*100,"% Loss:", loss.item())
            
            #update learning rate every 5 epochs (based on leveling of loss w/ no scheduler ~5 epochs)
            # if (epoch +1) % 5 == 0:
            #     ft_scheduler.step()  
            ft_scheduler.step()                  
        

             
        self.ft_model.save_embeddings(self.data.id2word, self.model_dir)
            

#--------------------------------------------------------------------------------------------------------------------------
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datareader', help = 'pickled pre-built datareader object', default = None,required = False)
    parser.add_argument('--train_dir',  help = 'location of training data', default = None, required = False)
    parser.add_argument('--input_file_name', help = 'text file for training data', default = None, required = False)
    
    parser.add_argument('--model_dir', help = 'location for trained embeddings to be saved', required = True)
    parser.add_argument('--emb_dimension', help = 'dimension of trained embedding', type = int, default = 100, required = False)
    parser.add_argument('--model_state_dict', help = 'state dict for pre-trained w2v model', default = None, required = False)
    
    parser.add_argument('--batch_size', type = int, default = 1024, required = False)
    parser.add_argument('--num_workers', help = 'if using cuda, how many cpu cores to use in data loader', type = int, default = 0, required = False)            
    parser.add_argument('--epochs',  help = 'number of full runs through data set', type = int, default = 3, required = False)
    parser.add_argument('--initial_lr', help = 'starting learning rate, to be updated by sparse_adam optimizer', type = float, default = 0.001, required = False)
    parser.add_argument('--window_size', help = 'training window size', type = int, default = 5, required = False)
    parser.add_argument('--wn_negative_sample', help = 'integrate wn knowledge in w2v loss function with additional contrastive loss', type = bool, default = False, required = False)
    parser.add_argument('--wn_positive_sample', help = 'integrate wn knowledge in w2v loss by extending with target word-synset member replacement', type = bool, default = False, required = False)
    parser.add_argument('--wn_depth', help = 'how many levels of hyponyms to include in a wn synset group, between 0 (synset members only) and 2', type = int, default = 0, required = False)
    parser.add_argument('--mismatch_weight', help = 'if wn_negative_sample = True: weight for wn similar words in w2v loss function, and w2v positive context words in wn loss function', type = float, default = 1.0, required = False)
    parser.add_argument('--w2v_loss_weight', help = 'if wn_negative_sample = True: weight for w2v loss function', type = float, default = 1.0, required = False)
    parser.add_argument('--wn_loss_weight', help = 'if wn_negative_sample = True: weight for wn loss function', type = float, default = 1.0, required = False)
    parser.add_argument('--margin', help = 'if wn_negative_sample = True: wn contrastive loss margin', type = float, default = 1.0, required = False)
    
    parser.add_argument('--wn_fine_tune', help = 'integrate wn knowledge with second model using w2v-trained embeddings; uses contrastive loss to move synset group centroids', type = bool, default = False, required = False)
    parser.add_argument('--ft_margin_weight', help = 'if wn_fine_tune = True: weight for calculating contrastive loss margins based on wn synset path distance', type = float, default = 0.1, required = False)
    parser.add_argument('--ft_num_negs', help = 'if wn_fine_tune = True: number of negative synset groups to be sampled', type = int, default = 4, required = False)
    parser.add_argument('--ft_epochs', help = 'if wn_fine_tune = True: number of full runs through wn dataset', type = int, default = 15, required = False)
    
    args = vars(parser.parse_args())
    w2v_wn = Word2VecWordnetTrainer(**args)
    
    if args["model_state_dict"] is None:
        w2v_wn.w2v_train()
    if args["wn_fine_tune"] is True:
        w2v_wn.wn_ft()
           
