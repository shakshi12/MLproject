import logging
import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from tqdm import tqdm
from  torchtext.datasets.text_classification import _csv_iterator, _create_data_from_iterator, TextClassificationDataset


NGRAMS=2

def _csv_iterator(data_path, ngrams, yield_cls=False, label=-1):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter="\t")
        for row in reader:
            tokens = ' '.join([row[5]])
            #print(row[5])
            tokens=tokenizer(tokens)

            if yield_cls:
                yield row[7], ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)


def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)


"""
train_csv_path = "./data/train.csv"
test_csv_path = "./data/test.csv"
"""
#-1: default
#0: am
#1: nam
def setup_datasets(train_csv_path, test_csv_path, include_unk=False):
    iterator=_csv_iterator(train_csv_path, NGRAMS)
    vocab = build_vocab_from_iterator(iterator)
    train_data, train_labels = _create_data_from_iterator(vocab, _csv_iterator(train_csv_path, NGRAMS, yield_cls=True, label=0), include_unk)
    test_data, test_labels = _create_data_from_iterator(vocab, _csv_iterator(test_csv_path, NGRAMS, yield_cls=True, label=0), include_unk)


    return TextClassificationDataset(vocab, train_data, train_labels), TextClassificationDataset(vocab, test_data, test_labels)



#print(setup_datasets(train_csv_path, test_csv_path, include_unk=False))
