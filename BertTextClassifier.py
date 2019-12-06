#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# adaptation by Loreto Parisi https://twitter.com/loretoparisi
# @author: https://twitter.com/mroberti
# from: https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta
#

import os
import random
from pathlib import Path 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tables import *

# torch
import torch
import torch.optim as optim
import transformers

# fastai
import fastai
from fastai.text import *
from fastai.callbacks import *
from transformers import AdamW

# transformers
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

# progress util
def disable_progress():
    ''' disable fastai progress bar '''
    fastprogress.fastprogress.NO_BAR = True
    master_bar, progress_bar = fastprogress.force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
    
def enable_progress():
    ''' enable fastai progress bar '''
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = fastprogress.master_bar, fastprogress.progress_bar


def pd2hdf(path, train, test):
    ''' store to HDF '''
    with pd.HDFStore(path,  mode='w') as store:
        store.append('train', train, data_columns= train.columns, format='table')
        store.append('test', test, data_columns= test.columns, format='table')

def hdf2pd(path):
    ''' HDF to train, test '''
    with pd.HDFStore(path,  mode='r') as store:
        train = store.select('train')
        test = store.select('train')
        return train,test

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def get_preds_as_nparray(learner,databunch,ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]

class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
        return [CLS] + tokens + [SEP]
    
class TransformersVocab(Vocab):
    ''' Custom Numericalizer '''
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

class CustomTransformerModel(nn.Module):
    ''' custom transformer model architecture '''
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        #attention_mask = (input_ids!=1).type(input_ids.type()) # Test attention_mask for RoBERTa
        
        logits = self.transformer(input_ids,
                                attention_mask = attention_mask)[0]   
        return logits

def get_predictions(learner,databunch):
        '''
            Creating prediction
            Now that the model is trained, we want to generate predictions from the test dataset.
        '''
        test_preds = get_preds_as_nparray(learner,databunch,DatasetType.Test)
        sample_submission = pd.read_csv('https://raw.githubusercontent.com/loretoparisi/bert_text_classifier/master/data/imdb_kaggle/sampleSubmission.csv', quoting=csv.QUOTE_ALL, engine="python", quotechar='"', encoding="utf-8")
        
        sample_submission['Sentiment'] = np.argmax(test_preds,axis=1)
        sample_submission.to_csv("predictions.csv", index=False)
        
        return sample_submission

def train_learner(learner):
        # Train
        
        learner.save('untrain')
        seed_all(seed)
        learner.load('untrain')
        learner.freeze_to(-1)
        learner.summary()
        
        # To use our `one_cycle` we will need an optimum learning rate. 
        # We can find this learning rate by using a learning rate finder which can be called by using `lr_find.
        learner.lr_find()
        
        # We will pick a value a bit before the minimum, where the loss still improves. Here 2x10^-3 seems to be a good value.
        learner.fit_one_cycle(1,max_lr=2e-03,moms=(0.8,0.7))
        learner.save('first_cycle')
        seed_all(seed)
        learner.load('first_cycle')

        # We then unfreeze the second group of layers and repeat the operations.
        learner.freeze_to(-2)
        lr = 1e-5

        # Note here that we use slice to create separate learning rate for each group.
        learner.fit_one_cycle(1, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))
        learner.save('second_cycle')
        seed_all(seed)
        learner.load('second_cycle')
        learner.freeze_to(-3)
        learner.fit_one_cycle(1, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))
        learner.save('third_cycle')
        seed_all(seed)
        learner.load('third_cycle')

        # Here, we unfreeze all the groups.
        learner.unfreeze()
        learner.fit_one_cycle(2, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))

if __name__ == '__main__':

    print('fastai version :', fastai.__version__)
    print('transformers version :', transformers.__version__)

    PATH = os.path.dirname(os.path.realpath(__file__))
    DATA = os.path.join(PATH,'data','imdb_kaggle')

    # check store
    train = None
    test = None
    if os.path.isfile(os.path.join(DATA,'imdb.h5')):
        try:
            train, test = hdf2pd(os.path.join(DATA,'imdb.h5'))
        except: # store data corrupted
            pass
    
    if train is None:
        # read train test
        train = pd.read_csv('https://raw.githubusercontent.com/loretoparisi/bert_text_classifier/master/data/imdb_kaggle/train.tsv', sep="\t", quoting=csv.QUOTE_ALL, engine="python", quotechar='"', encoding="utf-8")
        test = pd.read_csv('https://raw.githubusercontent.com/loretoparisi/bert_text_classifier/master/data/imdb_kaggle/test.tsv', sep="\t", quoting=csv.QUOTE_ALL, engine="python", quotechar='"', encoding="utf-8")
        pd2hdf(os.path.join(DATA,'imdb.h5'), train, test)

    print(train.shape, test.shape)
    print( train.head() )
    print( test.head() )

    MODEL_CLASSES = {
        'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
        'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
        'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
        'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
        'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
    }

    # Parameters
    seed = 42
    use_fp16 = False
    bs = 16

    model_type = 'roberta'
    pretrained_model_name = 'roberta-base' # 'roberta-base-openai-detector'

    # model_type = 'bert'
    # pretrained_model_name='bert-base-uncased'

    # model_type = 'distilbert'
    # pretrained_model_name = 'distilbert-base-uncased-distilled-squad'#'distilbert-base-uncased'#'distilbert-base-uncased'

    #model_type = 'xlm'
    #pretrained_model_name = 'xlm-clm-enfr-1024'

    #model_type = 'xlnet'
    #pretrained_model_name = 'xlnet-base-cased'
    model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
    print( model_class.pretrained_model_archive_map.keys() )

    seed_all(seed)

    transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
    fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

    print(tokenizer_class.pretrained_vocab_files_map)

    # Custom processor
    transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)
    tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)
    transformer_processor = [tokenize_processor, numericalize_processor]

    # Setting up the Databunch
    pad_first = bool(model_type in ['xlnet'])
    pad_idx = transformer_tokenizer.pad_token_id

    databunch = (TextList.from_df(train, cols='Phrase', processor=transformer_processor)
             .split_by_rand_pct(0.1,seed=seed)
             .label_from_df(cols= 'Sentiment')
             .add_test(test)
             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))

    print('[CLS] token :', transformer_tokenizer.cls_token)
    print('[SEP] token :', transformer_tokenizer.sep_token)
    print('[PAD] token :', transformer_tokenizer.pad_token)
    
    databunch.show_batch()

    print('[CLS] id :', transformer_tokenizer.cls_token_id)
    print('[SEP] id :', transformer_tokenizer.sep_token_id)
    print('[PAD] id :', pad_idx)
    test_one_batch = databunch.one_batch()[0]
    print('Batch shape : ',test_one_batch.shape)
    print(test_one_batch)

    # defining our model architecture 
    config = config_class.from_pretrained(pretrained_model_name)
    config.num_labels = 5
    config.use_bfloat16 = use_fp16
    print(config)
    transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)
    # transformer_model = model_class.from_pretrained(pretrained_model_name, num_labels = 5)
    custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)

    # ### Learner : Custom Optimizer / Custom Metric
    learner = Learner(databunch, 
                  custom_transformer_model, 
                  opt_func = lambda input: AdamW(input,correct_bias=False), 
                  metrics=[accuracy])

    # Show graph of learner stats and metrics after each epoch.
    learner.callbacks.append(ShowGraph(learner))

    # Put learn in FP16 precision mode. --> Seems to not working
    if use_fp16: learner = learner.to_fp16()

    # Discriminative Fine-tuning and Gradual unfreezing (Optional)
    print(learner.model)

    # For DistilBERT
    # list_layers = [learner.model.transformer.distilbert.embeddings,
    #                learner.model.transformer.distilbert.transformer.layer[0],
    #                learner.model.transformer.distilbert.transformer.layer[1],
    #                learner.model.transformer.distilbert.transformer.layer[2],
    #                learner.model.transformer.distilbert.transformer.layer[3],
    #                learner.model.transformer.distilbert.transformer.layer[4],
    #                learner.model.transformer.distilbert.transformer.layer[5],
    #                learner.model.transformer.pre_classifier]

    # For roberta-base
    list_layers = [learner.model.transformer.roberta.embeddings,
                learner.model.transformer.roberta.encoder.layer[0],
                learner.model.transformer.roberta.encoder.layer[1],
                learner.model.transformer.roberta.encoder.layer[2],
                learner.model.transformer.roberta.encoder.layer[3],
                learner.model.transformer.roberta.encoder.layer[4],
                learner.model.transformer.roberta.encoder.layer[5],
                learner.model.transformer.roberta.encoder.layer[6],
                learner.model.transformer.roberta.encoder.layer[7],
                learner.model.transformer.roberta.encoder.layer[8],
                learner.model.transformer.roberta.encoder.layer[9],
                learner.model.transformer.roberta.encoder.layer[10],
                learner.model.transformer.roberta.encoder.layer[11],
                learner.model.transformer.roberta.pooler]

    learner.split(list_layers)
    num_groups = len(learner.layer_groups)
    print('Learner split in',num_groups,'groups')
    print(learner.layer_groups)

    # train
    train_learner(learner)

    # get predictions from sample submission
    sample_submission = get_predictions(learner,databunch)

    print(test.head())
    print(sample_submission.shape)
    print(sample_submission.head())



