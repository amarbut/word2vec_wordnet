# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:11:04 2021

@author: Anna
"""

synset2word = {'a':[1,2,3], 'b':[3,4,5,6], 'c':[2,6,8], 'd':[1,9,10,11], 'e':[2,9], 'f':[12,11,1]}
num_negs = 3
wn_negative_probs = np.array(['a', 'a', 'a', 'b', 'b', 'c', 'c','c','c','c', 'd','d','e','e','e','f','f','f'])
np.random.shuffle(wn_negative_probs)

vocab_size = 13
emb_dimension = 10

u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)


def get_wn_negatives(syn, num_negs):
        response = []
        while len(response) < num_negs:
            response.extend([i for i in np.random.choice(wn_negative_probs,num_negs)])
        return response[0:num_negs]
    
    
class WordnetFineTuningDataset:
    def __init__(self, data, num_negs):
        self.data = data
        self.synset_list = list(data.keys())
        self.num_negs = num_negs
    
    def __len__(self):
        return len(self.synset_list)
    
    #function used to retrieve contrastive synset samples
    def __getitem__(self,idx):
        target = self.synset_list[idx]
        
        #get specified # of other synsets for comparison in contrastive loss function
        negs = get_wn_negatives(target, self.num_negs)
        return {'syn':target, 'negs': negs}
    
dataset = WordnetFineTuningDataset(synset2word, num_negs)

dataloader = DataLoader(dataset, batch_size = 3)

for i,batch in enumerate(dataloader):
    print(i, ": ")
    targets = batch['syn']
    negs = batch['negs']
    syn_words = [torch.LongTensor(synset2word[t]) for t in targets]
    syn_embeddings = [u_embeddings(words) for words in syn_words]
    
    syn_centroids = [torch.mean(emb, dim = 0) for emb in syn_embeddings]
    syn_dist = [torch.sqrt(torch.sum((syn_centroids[i]-syn_embeddings[i])**2, dim = 1)) for i in range(len(syn_centroids))]
    
    syn_pos_loss = sum([torch.sum(0.5*(dist**2)) for dist in syn_dist])
    
    #get actual synset name to calculate wn distance    
    #neg_syns = [[id2synset[n] for n in neg] for neg in negs]
    
    #get embeddings for all words in each wn neg sample syn group    
    neg_word_groups = [[torch.LongTensor(synset2word[n]) for n in neg] for neg in negs]
    neg_embedding_groups = [[u_embeddings(words) for words in group] for group in neg_word_groups]
    
    #use mean of all synset group member embeddings as neg synset group centroid
    neg_centroids = [torch.stack([torch.mean(emb, dim = 0) for emb in group]) for group in neg_embedding_groups]
    #compute distance between neg group centroids and target group centroids
    neg_dist = [torch.sqrt(torch.sum((syn_centroids[i]-neg_centroids[i])**2, dim = 1))  for i in range(len(syn_centroids))]

    
margins = [[margin_weight*(wn.synset(syns[i]).shortest_path_distance(wn.synset(neg))) for neg in neg_syns[i]] for i in range(len(syns))]

    
batches = [[(1,[2,3],[4,5,6,7],[8,9]),(10,[11,12],[13,14,15,16],[17,18])],[(21,[22,23],[24,25,26,27],[28,29]),(30,[31,32],[33,34,13,36],[37,38])]]    
all_u = [u for batch in batches for u, _, _, _ in batch if len(batch) > 0]
all_v = [v for batch in batches for _, v, _, _ in batch if len(batch) > 0]
all_neg = [neg for batch in batches for _, _, neg, _ in batch if len(batch) > 0]
all_wn = [wn_pos for batch in batches for _, _, _, wn_pos in batch if len(batch) > 0]

u = torch.LongTensor(all_u)
v = torch.LongTensor(all_v)
neg = torch.LongTensor(all_neg)
wn = torch.LongTensor(all_wn)

for i, batch in enumerate(w2v_wn.dataloader):
    while i <2:
        print(batch)
        
pos_pairs = [(u,word_ids[max(i - boundary, 0):i + boundary]) for i, u in enumerate(word_ids)]

input_file = open(w2v_wn.data.input_file_name, encoding = 'utf-8')
window_size = 5

while line:
    line = input_file.readline()
    # if not line:
    #     self.input_file.seek(0, 0)
    #     line = self.input_file.readline()
    
    if len(line) > 1:
        words = line.strip().split()
        print(words)
    
        if len(words) > 1:
            #collect word ids for sentence, ignoring words w/ subsample probability
            word_ids = [w2v_wn.data.word2id[w] for w in words if w in w2v_wn.data.word2id]
            
            #context window ranges from 1 to window size
            #skip for now because messes with tensor building
            #boundary = np.random.randint(1, (self.window_size-1)/2)
            boundary = int((window_size-1)/2)
            
            #collect list of all target/positive context pairs
            pos_pairs = [(u,word_ids[max(i - boundary, 0):i + boundary]) for i, u in enumerate(word_ids) if np.random.rand() > w2v_wn.data.subsample_probs[u]]
            
            
w2v_wn = Word2VecWordnetTrainer(datareader = "/home/anna/Documents/W2V_Data/moby_trials/data_reader.pkl",
                                #train_dir = "/home/anna/Documents/W2V_Data/", 
                                #input_file_name = "moby_clean.txt", 
                                #model_dir = "/home/anna/Documents/W2V_Data/moby_trials", 
                                model_state_dict = "/home/anna/Documents/W2V_Data/moby_trials/w2v_model.pth",
                                wn_negative_sample = True,
                                wn_fine_tune = True)

all_sim = wn.numpy().tolist()
all_not_sim = []
all_mismatch = []

#sort context and neg sample words by wordnet similarity to target
for idx,i in enumerate(u):
    sim = []
    not_sim = []
    w2v_mismatch = []
    for j in v[idx]:
        #skip padded cells in tensors
        if j != 0:
            #mark as similar if u and j share any synset groups
            if len(w2v_wn.data.wn_word2synset[i.item()]-w2v_wn.data.wn_word2synset[j.item()]) < len(w2v_wn.data.wn_word2synset[i.item()]):
                sim.append(j.item())
            else:
                w2v_mismatch.append(j.item())
    for k in neg[idx]:
        #skip padded cells in tensors
        if k != 0:
            #mark as similar if u and k share any synset groups
            if len(w2v_wn.data.wn_word2synset[i.item()]-w2v_wn.data.wn_word2synset[k.item()]) < len(w2v_wn.data.wn_word2synset[i.item()]):
                sim.append(k.item())
            else:
                not_sim.append(k.item())
    
    all_sim[idx].extend(sim)
    all_not_sim.append(not_sim)
    all_mismatch.append(w2v_mismatch)

#pad wn lists with zeros for conversion to tensors
all_sim_len = max([2,max([len(i) for i in all_sim])])
all_sim = [i+([0]*(all_sim_len-len(i))) for i in all_sim]

all_not_sim_len = max([2,max([len(i) for i in all_not_sim])])
all_not_sim = [i+([0]*(all_not_sim_len-len(i))) for i in all_not_sim]

all_mismatch_len = max([2,max([len(i) for i in all_mismatch])])
all_mismatch = [i+([0]*(all_mismatch_len-len(i))) for i in all_mismatch]

emb_u = w2v_wn.model.u_embeddings(u)

#get embeddings for wordnet similarity groups
emb_sim = w2v_wn.model.v_embeddings(torch.LongTensor(all_sim))
emb_not_sim = w2v_wn.model.v_embeddings(torch.LongTensor(all_not_sim))
emb_mismatch = w2v_wn.model.v_embeddings(torch.LongTensor(all_mismatch))


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