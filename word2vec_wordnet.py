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
            if len(line) > 1:
                self.sentence_count += 1
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
                    #skip synset if already added
                    if not s in self.wn_synset2id:
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
        while True:
        
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
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
                                
                                #gets similar words
                                syns = [s for ss in self.data.wn_word2synset[u] for s in self.data.wn_synset2word[ss]]
                                
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
                                    if j != 0:
                                        #mark as similar if u and j share any synset groups
                                        if len(self.data.wn_word2synset[u]-self.data.wn_word2synset[j]) < len(self.data.wn_word2synset[u]):
                                            sim.append(j)
                                        else:
                                            w2v_mismatch.append(j)
                                for k in neg_ids:
                                    #skip padded cells in tensors
                                    if k != 0:
                                        #mark as similar if u and k share any synset groups
                                        if len(self.data.wn_word2synset[u]-self.data.wn_word2synset[k]) < len(self.data.wn_word2synset[u]):
                                            sim.append(k)
                                        else:
                                            not_sim.append(k)
                                
                                sims.append(sim)
                                not_sims.append(not_sim)
                                mismatches.append(w2v_mismatch)
                                                      
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
        all_u = [u for batch in batches for u, _, _, _, _, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _, _, _, _, _ in batch if len(batch) > 0]
        all_neg = [neg for batch in batches for _, _, neg, _, _, _, _ in batch if len(batch) > 0]
        all_wn = [wn_pos for batch in batches for _, _, _, wn_pos, _, _, _ in batch if len(batch) > 0]
        
        #pad with zeros for conversion to tensors,must be at least length 2 to preserve tensor math
        all_sim = [sim for batch in batches for _, _, _, _, sim, _, _ in batch if len(batch) > 0]
        sim_len = max([2,max([len(i) for i in all_sim])])
        all_sim = [i+([0]*(sim_len-len(i))) for i in all_sim]
        
        all_not_sim = [not_sim for batch in batches for _, _, _, _, _, not_sim, _ in batch if len(batch) > 0]
        not_sim_len = max([2,max([len(i) for i in all_not_sim])])
        all_not_sim = [i+([0]*(not_sim_len-len(i))) for i in all_not_sim]
        
        all_mismatch = [mismatch for batch in batches for _, _, _, _, _, _, mismatch in batch if len(batch) > 0]
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

class WordnetFineTuningDataset:
    def __init__(self, data, num_negs):
        self.data = data
        self.synset_list = list(data.wn_synset2word.keys())
        self.num_negs = num_negs
    
    def __len__(self):
        return len(self.synset_list)
    
    #function used to retrieve contrastive synset samples
    def __getitem__(self,idx):
        target = self.synset_list[idx]
        
        #get specified # of other synsets for comparison in contrastive loss function
        negs = self.data.get_wn_negatives(target, self.num_negs)
        return {'syn':target, 'negs': negs}
            
        

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
        #use loss function from w2v paper
        w2v_pos_loss = torch.sum(-F.logsigmoid(w2v_pos_loss), dim = 1).unsqueeze(1)
        
        #calculate dot product for target and all negative sample words
        w2v_neg_loss = torch.bmm(emb_neg,emb_u.unsqueeze(2)).squeeze()
        #use loss function from w2v paper
        w2v_neg_loss = torch.sum(-F.logsigmoid(-w2v_neg_loss), dim = 1).unsqueeze(1)
        
        #calculate dot product for target and all wn_positive(w2v negative) sample words
        w2v_mismatch_loss = torch.bmm(emb_wn,emb_u.unsqueeze(2)).squeeze()
        # use loss function from w2v paper, downweighted so not in competition with positive wn similarity loss
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
            wn_pos_dist = torch.sqrt(torch.sum(((emb_u.unsqueeze(1)-emb_sim.unsqueeze(0))**2).squeeze(1), dim =3)).squeeze()
            wn_neg_dist = torch.sqrt(torch.sum((emb_u.unsqueeze(1)-emb_not_sim.unsqueeze(0))**2, dim =3)).squeeze()
            wn_mismatch_dist = torch.sqrt(torch.sum((emb_u.unsqueeze(1)-emb_mismatch.unsqueeze(0))**2, dim =3)).squeeze()
            
                       
            #for dissimilar words: replace distance with 0 if larger than contrastive loss margin, or difference from margin if not 
            wn_neg_dist = margin - wn_neg_dist
            wn_neg_dist[wn_neg_dist<0] = 0
            
            wn_mismatch_dist = margin - wn_mismatch_dist
            wn_mismatch_dist[wn_mismatch_dist<0] = 0
            
            #use distances in contrastive loss function
            wn_pos_loss = torch.sum(0.5*(wn_pos_dist**2),1).unsqueeze(1)
            wn_neg_loss = torch.sum(0.5*(wn_neg_dist**2),1).unsqueeze(1)
            #downweight loss from dissimilar words in word2vec context so not competing
            wn_mismatch_loss = mismatch_weight*(torch.sum(0.5*(wn_mismatch_dist**2),1).unsqueeze(1))
            
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
    def __init__(self, skip_gram_model, wn_id2synset, wn_synset2word, num_negs, margin_weight):
        super(WordnetFineTuning, self).__init__()
        self.embeddings = skip_gram_model.u_embeddings
        self.num_negs = num_negs
        self.margin_weight = margin_weight
        self.wn_id2synset = wn_id2synset
        self.wn_synset2word = wn_synset2word
        
        
    def forward(self, targets, negs, margin_weight=1):
        
        #get actual synset name to calculate wn distance
        syns = [self.wn_id2synset[t] for t in targets]
        
        #get embeddings for all words in wn syn group
        syn_words = [torch.LongTensor(self.wn_synset2word[t]) for t in targets]
        syn_embeddings = [self.embeddings(words) for words in syn_words]
        
        #use mean of all synset group member embeddings as synset centroid
        syn_centroids = torch.stack([torch.mean(emb, dim = 0) for emb in syn_embeddings])
        #compute distance between centroid and synset group members
        syn_dist = [torch.sqrt(torch.sum((syn_centroids[i]-syn_embeddings[i])**2, dim = 1)) for i in range(len(syn_centroids))]
        #use contrastive loss to move all words in synset group closer
        syn_pos_loss = torch.stack([torch.sum(0.5*(dist**2)) for dist in syn_dist])
        
        
        #get actual synset name to calculate wn distance    
        neg_syns = [[self.id2synset[n] for n in neg] for neg in negs]
        
        #get embeddings for all words in each wn neg sample syn group    
        neg_word_groups = [[torch.LongTensor(self.wn_synset2word[n]) for n in neg] for neg in negs]
        neg_embedding_groups = [[self.embeddings(words) for words in group] for group in neg_word_groups]
        
        #use mean of all synset group member embeddings as neg synset group centroid
        neg_centroids = torch.stack([torch.stack([torch.mean(emb, dim = 0) for emb in group]) for group in neg_embedding_groups])
        #compute distance between neg group centroids and target group centroids
        neg_dist = torch.sqrt(torch.sum((syn_centroids-neg_centroids)**2, dim = 2))
        
        #use wn distance between target synset and neg synset to set contrastive loss margins
        margins = torch.stack([torch.LongTensor([margin_weight*(wn.synset(syns[i]).shortest_path_distance(wn.synset(neg))) for neg in neg_syns[i]]) for i in range(len(syns))])
        
        #replace neg_dist with difference from margin or 0 if outside of margin
        neg_dist = torch.sub(margins, neg_dist)
        neg_dist[neg_dist<0] = 0
        
        #use distances in contrastive loss function
        neg_loss = torch.sum(0.5*(neg_dist**2),1)

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
                 wn_fine_tune = False, ft_margin_weight = 1, ft_num_negs = 4, ft_epochs = 5):
        
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
        
        if model_state_dict is not None:
            self.pretrained = True
            self.model.load_state_dict(torch.load(model_state_dict))
        else:
            self.pretrained = False
        
        dataset = Word2VecWordnetDataset(self.data, window_size, wn_negative_sample, wn_positive_sample)
        self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers,
                                     collate_fn = dataset.collate_fn)
        
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.mismatch_weight = mismatch_weight
        self.w2v_loss_weight = w2v_loss_weight
        self.wn_loss_weight = wn_loss_weight
        self.margin = margin
        
            
        self.wn_fine_tune = wn_fine_tune
        self.ft_margin_weight = ft_margin_weight
        self.ft_num_negs = ft_num_negs
        self.ft_epochs = ft_epochs
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print("cuda available:", self.use_cuda)
        if self.use_cuda:
            self.model.cuda()
            
            
    def w2v_train(self):
        print("training on cuda: ", self.use_cuda)
        for epoch in range(self.epochs):
            
            print('\nStarting Epoch', (epoch+1))
            optimizer = optim.SparseAdam(self.model.parameters(), lr = self.initial_lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
            
            for i, batch in enumerate(self.dataloader):
                if len(batch[0])>1:
                    u = batch[0].to(self.device)
                    v = batch[1].to(self.device)
                    neg = batch[2].to(self.device)
                    wn = batch[3].to(self.device)
                    sim = batch[4].to(self.device)
                    not_sim = batch[5].to(self.device)
                    mismatch = batch[6].to(self.device)
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.model.forward(u, v, neg, wn, sim, not_sim, mismatch, self.mismatch_weight, self.w2v_loss_weight, self.wn_loss_weight, self.margin)
                    loss.backward()
                    optimizer.step()
                    
                    if i > 0 and i % 5000 == 0:
                        print((i/len(self.dataloader))*100,"% Loss:", loss.item())

        
        self.model.save_embeddings(self.data.id2word, self.model_dir)        

                    
    def wn_ft(self):

        ft_dataset = WordnetFineTuningDataset(self.data, self.ft_num_negs)
        ft_batch_size = int(len(self.data.wn_synset2id)/300)
        ft_dataloader = DataLoader(ft_dataset, batch_size = ft_batch_size, shuffle = False, num_workers = 0)

        ft_model = WordnetFineTuning(self.model, self.data.wn_id2synset, self.data.wn_synset2word, self.ft_num_negs, self.ft_margin_weight)
        
        for epoch in range(self.ft_epochs):
            print("Starting Epoch:", (epoch+1))
            
            # set initial learning rate at 1/10 skipgram rate--encourage small adjustments to pre-trained embeddings
            ft_optimizer = optim.SparseAdam(ft_model.parameters(), lr = (self.initial_lr/10))
            ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, len(ft_dataloader))
            
            for i, batch in enumerate(ft_dataloader):
                targets = batch['syn']
                negs = batch['negs']
                    
                ft_scheduler.step()
                ft_optimizer.zero_grad()
                loss = ft_model.forward(targets, negs, self.ft_margin_weight)
                loss.backward()
                ft_optimizer.step()
                
                if i > 0 and i % 500 == 0:
                    print((i/len(ft_dataloader))*100,"% Loss:", loss.item())
                
        ft_model.save_embeddings(self.data.id2word, self.model_dir)
            

#--------------------------------------------------------------------------------------------------------------------------
# #TODO: work out arguments so databuild, skipgram, and fine tune can be performed separately       
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--datareader', help = 'pickled pre-built datareader object', default = None,required = False)
#     parser.add_argument('--train_dir',  help = 'location of training data', default = None, required = False)
#     parser.add_argument('--input_file_name', help = 'text file for training data', default = None, required = False)
    
#     parser.add_argument('--model_dir', help = 'location for trained embeddings to be saved', required = True)
#     parser.add_argument('--emb_dimension', help = 'dimension of trained embedding', type = int, default = 100, required = False)
#     parser.add_argument('--model_state_dict', help = 'state dict for pre-trained w2v model', default = None, required = False)
    
#     parser.add_argument('--batch_size', type = int, default = 1024, required = False)
#     parser.add_argument('--num_workers', help = 'if using cuda, how many cpu cores to use in data loader', type = int, default = 0, required = False)            
#     parser.add_argument('--epochs',  help = 'number of full runs through data set', type = int, default = 3, required = False)
#     parser.add_argument('--initial_lr', help = 'starting learning rate, to be updated by sparse_adam optimizer', type = float, default = 0.001, required = False)
#     parser.add_argument('--window_size', help = 'training window size', type = int, default = 5, required = False)
#     parser.add_argument('--wn_negative_sample', help = 'integrate wn knowledge in w2v loss function with additional contrastive loss', type = bool, default = False, required = False)
#     parser.add_argument('--wn_positive_sample', help = 'integrate wn knowledge in w2v loss by extending with target word-synset member replacement', type = bool, default = False, required = False)
#     parser.add_argument('--wn_depth', help = 'how many levels of hyponyms to include in a wn synset group, between 0 (synset members only) and 2', type = int, default = 0, required = False)
#     parser.add_argument('--mismatch_weight', help = 'if wn_negative_sample = True: weight for wn similar words in w2v loss function, and w2v positive context words in wn loss function', type = float, default = 1.0, required = False)
#     parser.add_argument('--w2v_loss_weight', help = 'if wn_negative_sample = True: weight for w2v loss function', type = float, default = 1.0, required = False)
#     parser.add_argument('--wn_loss_weight', help = 'if wn_negative_sample = True: weight for wn loss function', type = float, default = 1.0, required = False)
#     parser.add_argument('--margin', help = 'if wn_negative_sample = True: wn contrastive loss margin', type = float, default = 1.0, required = False)
    
#     parser.add_argument('--wn_fine_tune', help = 'integrate wn knowledge with second model using w2v-trained embeddings; uses contrastive loss to move synset group centroids', type = bool, default = False, required = False)
#     parser.add_argument('--ft_margin_weight', help = 'if wn_fine_tune = True: weight for calculating contrastive loss margins based on wn synset path distance', type = float, default = 1.0, required = False)
#     parser.add_argument('--ft_num_negs', help = 'if wn_fine_tune = True: number of negative synset groups to be sampled', type = int, default = 4, required = False)
#     parser.add_argument('--ft_epochs', help = 'if wn_fine_tune = True: number of full runs through wn dataset', type = int, default = 5, required = False)
    
#     args = vars(parser.parse_args())
#     w2v_wn = Word2VecWordnetTrainer(**args)
    
#     if args["model_state_dict"] is None:
#         w2v_wn.train()
#     if args["wn_fine_tune"] is True:
#         w2v_wn.wn_fine_tune()
           
            
