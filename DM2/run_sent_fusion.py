import numpy as np
import pickle
import keras.backend as K

from keras.layers import  GlobalAveragePooling1D, Input, Activation, MaxPooling1D, BatchNormalization, Dense, Dropout, Conv1D,GlobalMaxPooling1D
from keras.layers import GRU,AveragePooling1D,CuDNNGRU
from keras.layers.merge import Concatenate
from keras.models import Model 
from keras.callbacks import EarlyStopping,ModelCheckpoint


import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
seed = 777
np.random.seed(seed)
torch.manual_seed(seed)

def gen_embedding(words4sent, max_seq_len, feature_dim,   to_reverse=0):
    length = []
    output = []
    
    for words in words4sent:
        if to_reverse:
            words = np.flip(words, 0)
        length.append( words.shape[0])
        if  words.shape[0] < max_seq_len:
            wordList = np.concatenate([words,np.zeros([max_seq_len - words.shape[0],feature_dim])])
        output.append(wordList)
    return np.array(output),np.array(length) 

def mean_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_() # BxF
    for i in range(x.size(1)):
        out[i] = torch.mean(x[:lengths[i],i,:], 0)
    return out


class RandLSTM(nn.Module):

    def __init__(self, input_dim, num_layers, output_dim,  bidirectional=False):
        super(RandLSTM, self).__init__()
        

        self.bidirectional = bidirectional
        self.max_seq_len = 128
        self.input_dim = input_dim
         

        self.e_hid_init = torch.zeros(1, 1, output_dim)
        self.e_cell_init = torch.zeros(1, 1, output_dim)

        self.output_dim = output_dim
        self.num_layers = num_layers
        self.lm = nn.LSTM(input_dim, output_dim, num_layers=num_layers,
                          bidirectional= self.bidirectional, batch_first=True)

        self.bidirectional += 1
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

   

    def lstm(self, inputs, lengths):
        bsz, max_len, _ = inputs.size()
        in_embs = inputs
        lens, indices = torch.sort(lengths, 0, True)

        e_hid_init = self.e_hid_init.expand(1*self.num_layers*self.bidirectional, bsz, self.output_dim).contiguous()
        e_cell_init = self.e_cell_init.expand(1*self.num_layers*self.bidirectional, bsz, self.output_dim).contiguous()
        all_hids, (enc_last_hid, _) = self.lm(pack(in_embs[indices],
                                                        lens.tolist(), batch_first=True), (e_hid_init, e_cell_init))
        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]

        return all_hids

    def forward(self, words4sent):
        
        out, lengths = gen_embedding(words4sent, self.max_seq_len, self.input_dim)
        out = torch.from_numpy(out).float()
        lengths = torch.from_numpy(np.array(lengths))
        out = self.lstm(out, lengths)
#         print("output size:",out.size())
        out = out.transpose(1,0)
        out = mean_pool(out, lengths)
        return out

    def encode(self, batch):
        return self.forward(batch).cpu().detach().numpy()
    

from six.moves import cPickle as pickle #for performance

 
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

# extract w2v model
from gensim.models import Word2Vec 

w2vmodel =  Word2Vec.load('/home/xhh/PMC_model/PMC_model.txt')
def vector_name(name): 
    s = name.split(' ')
    vectors = [] 
    for word in s: 
        if w2vmodel.wv.__contains__(word):   
            vectors.append(w2vmodel.wv[word]) 
    else: 
        clear_words = re.sub('[^A-Za-z0-9]+', ' ', word) 
        clear_words = clear_words.lstrip().rstrip().split(' ')
        for w in clear_words: 
            if w2vmodel.wv.__contains__(w): 
                vectors.append(w2vmodel.wv[w]) 
    return vectors


# read go.obo obtain ontology type
 
obo_file = '../cross-species/go.obo'
fp=open(obo_file,'r')
obo_txt=fp.read()
fp.close()
obo_txt=obo_txt[obo_txt.find("[Term]")-1:]
obo_txt=obo_txt[:obo_txt.find("[Typedef]")]
# obo_dict=parse_obo_txt(obo_txt)
id_name_dicts = {}
for Term_txt in obo_txt.split("[Term]\n"):
    if not Term_txt.strip():
        continue
    name = ''
    ids = []
    for line in Term_txt.splitlines():
        if   line.startswith("id: "):
            ids.append(line[len("id: "):])     
        elif line.startswith("name: "):
             name=line[len("name: "):]
        elif line.startswith("alt_id: "):
            ids.append(line[len("alt_id: "):])
    
    for t_id in ids:
        id_name_dicts[t_id] = name
        
        
        
import re
protein2go =  load_dict('DM2prot2go.pkl')
prot2emb_w2v = {}
project_dim = 1024
num_layers = 1
max_go_len = 1024
max_protlen = 0
w2vlstm = RandLSTM(200,num_layers,  project_dim, bidirectional = False)


for key, value in protein2go.items(): 
    allgos = value.split(',') 
    allgos = list(set(allgos))
    count = 0
    words4sent = []
    for  go in  allgos:
        if len(go) > 2:
            feature = np.array(vector_name(id_name_dicts[go]))
            if feature.shape[0] > 0:
                words4sent.append(feature)
            
        count += feature.shape[0]
    if len(words4sent) > 0:
        sent_embedding = w2vlstm.encode(words4sent)
    else:
        sent_embedding = np.zeros((1, project_dim))
    if max_protlen < sent_embedding.shape[0]:
        max_protlen = sent_embedding.shape[0]
    prot2emb_w2v[key] = sent_embedding 

del w2vmodel


print(max_protlen)
w2v_len = 150
# w2v_len = 211
prot2emb_bert = {}
max_protlen = 0
input_dim = 768
 
bertlstm = RandLSTM(input_dim,num_layers,  project_dim, bidirectional = False)
for key, value in protein2go.items():
     
    allgos = value.split(',') 
    allgos = list(set(allgos))
    count = 0
    words4sent = []
    for  go in  allgos:
        if len(go) > 2:
            feature = np.load('../ncbi_allfeatures4go/'+go+'_0.npy')[1:-1]
            words4sent.append(feature)
        count += feature.shape[0] 
    if len(words4sent) > 0:
        sent_embedding = bertlstm.encode(words4sent)
    else:
        sent_embedding = np.zeros((1, project_dim))
    if max_protlen < sent_embedding.shape[0]:
        max_protlen = sent_embedding.shape[0]
    prot2emb_bert[key] = sent_embedding 
    
    
print(max_protlen)
bert_len = 150
import keras
feature_len = 768

max_seq_len = 1000
# max_protlen = 32
 




class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,  ppi_pair_file, batch_size=128):
        'Initialization' 
        self.batch_size = batch_size
        self.ppi_pair_file = ppi_pair_file
        input_dim = 768
        num_layers = 1
         
        
        
        self.projection_dim = project_dim
        self.bert_len = bert_len
        self.w2v_len = w2v_len
        self.max_seqlen = max_seq_len
        self.protein2seq = load_dict('DM2prot2seq.pkl')
        self.read_ppi()
         
        self.prot2emb_bert =  prot2emb_bert
        self.prot2emb_w2v = prot2emb_w2v
         
#         self.prot2embedding() 
         
        self.on_epoch_end()
    
    def read_ppi(self):
        with open(self.ppi_pair_file, 'r') as f:
            self.ppi_pairs  =  f.readlines()
    
    
    
   

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ppi_pairs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ppi_pairs))
         
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

         
#         X_seq1 = np.empty((self.batch_size, self.max_seqlen,20))
#         X_seq2 = np.empty((self.batch_size, self.max_seqlen,20))
        y = np.empty((self.batch_size))
        X_go1 = np.empty((self.batch_size, self.bert_len + self.w2v_len,self.projection_dim))
        X_go2 = np.empty((self.batch_size, self.bert_len + self.w2v_len,self.projection_dim))


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ppi_pair = self.ppi_pairs[ID]
            p1, p2, label = ppi_pair.rstrip().split('\t')
            if label == '+':
                y[i] = 1
            else:
                y[i] = 0
#             X_seq1[i] =  self.protein2onehot[p1]
#             X_seq2[i] =  self.protein2onehot[p2]
            
            prot1emb_bert = self.prot2emb_bert[p1]
            X_go1[i,:prot1emb_bert.shape[0]] = prot1emb_bert
            
            prot2emb_bert = self.prot2emb_bert[p2]
            X_go2[i,:prot2emb_bert.shape[0]] = prot2emb_bert
            
            prot1emb_w2v = self.prot2emb_w2v[p1]
            X_go1[i,prot1emb_bert.shape[0]:prot1emb_w2v.shape[0] + prot1emb_bert.shape[0]] = prot1emb_w2v
            
            prot2emb_w2v = self.prot2emb_w2v[p2]
            X_go2[i,prot2emb_bert.shape[0]:prot2emb_bert.shape[0] + prot2emb_w2v.shape[0] ] = prot2emb_w2v
            
             
            
            
        return [X_go1, X_go2] ,  y



    def all_data(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

         
#         X_seq1 = np.empty((len(list_IDs_temp), self.max_seqlen,20))  
#         X_seq2 = np.empty((len(list_IDs_temp), self.max_seqlen,20))
        y = np.empty((len(list_IDs_temp)))
        X_go1 = np.empty((len(list_IDs_temp), self.bert_len + self.w2v_len,self.projection_dim))
        X_go2 = np.empty((len(list_IDs_temp), self.bert_len + self.w2v_len,self.projection_dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ppi_pair = self.ppi_pairs[ID]
            p1, p2, label = ppi_pair.rstrip().split('\t')
            if label == '+':
                y[i] = 1
            else:
                y[i] = 0
#             X_seq1[i] =  self.protein2onehot[p1]
#             X_seq2[i] =  self.protein2onehot[p2]
            
            prot1emb_bert = self.prot2emb_bert[p1]
            X_go1[i,:prot1emb_bert.shape[0]] = prot1emb_bert
            
            prot2emb_bert = self.prot2emb_bert[p2]
            X_go2[i,:prot2emb_bert.shape[0]] = prot2emb_bert
            
            prot1emb_w2v = self.prot2emb_w2v[p1]
            X_go1[i,prot1emb_bert.shape[0]:prot1emb_w2v.shape[0] + prot1emb_bert.shape[0]] = prot1emb_w2v
            
            prot2emb_w2v = self.prot2emb_w2v[p2]
            X_go2[i,prot2emb_bert.shape[0]:prot2emb_bert.shape[0] + prot2emb_w2v.shape[0] ] = prot2emb_w2v
            
           
        return [X_go1, X_go2] ,  y
from keras.layers import   Embedding
from keras.layers import  GRU, Bidirectional, CuDNNGRU, Lambda, Flatten
from keras.utils import multi_gpu_model
from keras.layers.merge import concatenate
from keras_radam import RAdam
from keras_lookahead import Lookahead

def inception_block(input_tensor, output_size):
    """"""
    con1d_filters = int(output_size/4)
    y = Conv1D(con1d_filters, 3, activation="relu", padding='same')(input_tensor)
    x1 = Conv1D(con1d_filters, 5, activation="relu", padding='same')(y)

    y = Conv1D(con1d_filters, 1, activation="relu", padding='valid')(input_tensor)
    x2 = Conv1D(con1d_filters, 3, activation="relu", padding='same')(y)

    x3 = Conv1D(con1d_filters, 3, activation="relu", padding='same')(input_tensor)
    x4 = Conv1D(con1d_filters, 1, activation="relu", padding='same')(input_tensor)

    y = Concatenate()([x1, x2, x3, x4])
#     y = MaxPooling1D(4)(mix0)
    # y = AveragePooling1D()(mix0)
#     y = BatchNormalization()(y)

    return y


class ResidualConv1D:
    """
    ***ResidualConv1D for use with best performing classifier***
    """

    def __init__(self, filters, kernel_size, pool=False):
        self.pool = pool
        self.kernel_size = kernel_size
        self.params = {
            "padding": "same",
            "kernel_initializer": "he_uniform",
            "strides": 1,
            "filters": filters,
        }

    def build(self, x):

        res = x
        if self.pool:
            x = MaxPooling1D(1, padding="same")(x)
            res = Conv1D(kernel_size=1, **self.params)(res)

        out = Conv1D(kernel_size=1, **self.params)(x)

#         out = BatchNormalization(momentum=0.9)(out)
        out = Activation("relu")(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.params)(out)

#         out = BatchNormalization(momentum=0.9)(out)
        out = Activation("relu")(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.params)(out)

        out = keras.layers.add([res, out])

        return out

    def __call__(self, x):
        return self.build(x)


def build_residual_cnn(input_x):
    

    x = Conv1D(
        filters=32,
        kernel_size=16,
        padding="same",
        kernel_initializer="he_uniform",
    )(input_x)
#     x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # residual net part
    x = ResidualConv1D(filters=32, kernel_size=3, pool=True)(x)
    x = ResidualConv1D(filters=32, kernel_size=3)(x)
    x = ResidualConv1D(filters=32, kernel_size=3)(x)
    
    
    x = ResidualConv1D(filters=64, kernel_size=3, pool=True)(x)
    x = ResidualConv1D(filters=64, kernel_size=3)(x)
    x = ResidualConv1D(filters=64, kernel_size=3)(x)
    
    
    x = ResidualConv1D(filters=128, kernel_size=3, pool=True)(x)
    x = ResidualConv1D(filters=128, kernel_size=3)(x)
    x = ResidualConv1D(filters=128, kernel_size=3)(x)
    
#     x = BatchNormalization(momentum=0.9)(x)
    x = Activation("relu")(x)
#     x = MaxPooling1D(1, padding="same")(x)
    x  = GlobalAveragePooling1D()(x)
    return x
    

def build_fc_model(input_x):
     
    
    x_a = GlobalAveragePooling1D()(input_x)
    x_b = GlobalMaxPooling1D()(input_x)
#     x_c = Attention()(input_x)
    x = Concatenate()([x_a, x_b])
    x = Dense(1024)(x)
#     x = Dense(256)(x)
    return x 



def build_model():
    con_filters = 128
    left_input_go = Input(shape=(bert_len+w2v_len,project_dim))
    right_input_go = Input(shape=(bert_len+w2v_len,project_dim))
    NUM_FILTERS = 32
    FILTER_LENGTH1 = 5
    FILTER_LENGTH2 = 5
    
    
    left_x_go= Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(left_input_go)
    left_x_go = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(left_x_go)
    left_x_go = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(left_x_go)
    left_x_go_max = GlobalMaxPooling1D()(left_x_go) #pool_size=pool_length[i]
    left_x_go_avg = GlobalAveragePooling1D()(left_x_go) #pool_size=pool_length[i]


    right_x_go = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(right_input_go)
    right_x_go = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(right_x_go)
    right_x_go = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(right_x_go)
    right_x_go_max = GlobalMaxPooling1D()(right_x_go)
    right_x_go_avg = GlobalAveragePooling1D()(right_x_go)

 
     
    x =   Concatenate()([left_x_go_avg, left_x_go_max  , right_x_go_avg, right_x_go_max])
     
     
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
  
     
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)
    # model = Model([left_input_go, right_input_go], output)
    optimizer = Lookahead(RAdam())
  
    model = Model([left_input_go, right_input_go], output)
#     model = multi_gpu_model(model, gpus=2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = build_model()
print(model.summary())
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from keras.utils import multi_gpu_model
import os
import sys
dataset_name = 'AT'
 
def main():
    rep = sys.argv[1]
    split = sys.argv[2]
    
     
    train_pairs_file = 'CV/train'+str(rep)+'-'+str(split)
    test_pairs_file = 'CV/test'+str(rep)+'-'+str(split)
    valid_pairs_file = 'CV/valid'+str(rep)+'-'+str(split)

    batch_size = 32
    train_generator = DataGenerator(   train_pairs_file,batch_size = batch_size )
    test_generator = DataGenerator(   test_pairs_file,batch_size = batch_size)
    valid_generator = DataGenerator(   valid_pairs_file,batch_size = batch_size)

    # model = build_model_without_att()
    model = build_model()
    save_model_name = 'CV/fusion_sent_GoplusSeq'+str(rep)+'-'+str(split) + '.hdf5'

    earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='max')
    save_checkpoint = ModelCheckpoint(save_model_name, save_best_only=True, monitor='val_acc', mode='max', save_weights_only=True)

         
         
    hist = model.fit_generator(generator=train_generator,
                epochs = 100,verbose=1,validation_data=valid_generator,callbacks=[earlyStopping, save_checkpoint])

        
         
    model.load_weights(save_model_name)
    with open(test_pairs_file, 'r') as f:
        test_ppi_pairs  =  f.readlines()

    test_len = len(test_ppi_pairs) 
    list_IDs_temp = np.arange(test_len)

    test_x, y_test = test_generator.all_data(list_IDs_temp)

    y_pred_prob = model.predict(test_x)


    y_pred = (y_pred_prob > 0.5)
    auc = metrics.roc_auc_score(y_test, y_pred_prob) 
    f1 = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_pred_prob)
    pr_auc = metrics.auc(recall, precision)
    mcc = matthews_corrcoef(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total=tn+fp+fn+tp
    sen = float(tp)/float(tp+fn)
    sps = float(tn)/float((tn+fp))

    tpr = float(tp)/float(tp+fn)
    fpr = float(fp)/float((tn+fp))
    print('--------------------------\n')
    print ('AUC: %f' % auc)
    print ('ACC: %f' % acc) 
    # print("PRAUC: %f" % pr_auc)
    print ('MCC : %f' % mcc)
    # print ('SEN: %f' % sen)
    # print ('SEP: %f' % sps)
    print('TPR:%f'%tpr)
    print('FPR:%f'%fpr)
    print('Pre:%f'%pre)
    print('F1:%f'%f1)
    print('--------------------------\n')
    np.savez('CV/sent_fusion_'+rep+'-'+split, AUCs=auc, ACCs=acc, MCCs=mcc, TPRs = tpr, FPRs=fpr, Precs=pre, F1s=f1)
 
    
if __name__ == "__main__":
    main()
    
    