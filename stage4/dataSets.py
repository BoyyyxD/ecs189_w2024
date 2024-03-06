import torch
from torch.utils.data import Dataset
import string
import re
import os
import pickle
from nltk import PorterStemmer
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot


class ClassificationDataset(Dataset):

    def __init__(self, process_data=False,test=False, file_path="/root/ucd/Interesno/ecs189_w2024/stage4/text_classification/data"):
        if process_data:
            self._classificationRearrange()
            return
        
        self.file_path = file_path
        self.test = test
        with open(self.file_path, "rb") as file:
            data = pickle.load(file)
        self.data = data["test" if test else "train"]
        self.unique = data["unique"] #120k unique values
        self.vocab_to_int = data["vocab_to_int"]
        self.int_to_vocab = data["int_to_vocab"]
        self.max_len = data["max_len"]
        self.seq_len = 50 # lol
    
        self.vocab_to_int[""] = len(self.unique)
        self.int_to_vocab[len(self.unique)] = ""
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        value = self.data[index]["value"]
        label = self.data[index]["label"]
        label = int(not label == "negative") 
        if len(value) < self.seq_len:
            value.extend([""] * (self.seq_len - len(value)))
        else:
            value = value[:self.seq_len]
        value = [self.vocab_to_int[i] for i in value]
        return torch.tensor(value), one_hot(torch.tensor(label), 2).float()
       
       
    def _classificationRearrange(self, p="/root/ucd/Interesno/ecs189_w2024/stage4/text_classification"):
        ps = PorterStemmer()
        
        def _adjust_data(d):
            data = d.translate(str.maketrans("", "", string.punctuation))
            data = re.sub(r"\W=", " ", data).lower()
            data = re.split(r"\W+", data)
            data = [ps.stem(i) for i in data if data != ""]
            return data      
        
        train = os.path.join(p, "train")
        data, db, full = [], {}, []
        max_len = 0
        
        
        unique = set()
        for file_name in os.listdir(os.path.join(train, "pos")):
            with open(os.path.join(train, "pos", file_name), "r", encoding="UTF-8") as file:
                f = _adjust_data(file.read())
                max_len = len(f) if len(f) > max_len else max_len
                
                for word in f:
                    unique.add(word)
                    full.append(word)
                data.append({"label": "positive", "value": f})
        for file_name in os.listdir(os.path.join(train, "neg")):
            with open(os.path.join(train, "neg", file_name), "r", encoding="UTF-8") as file:
                f = _adjust_data(file.read())
                max_len = len(f) if len(f) > max_len else max_len
                for word in f:
                    unique.add(word)
                    full.append(word)
                data.append({"label": "negative", "value": f})
        db["train"] = data
        data = []
        test = os.path.join(p, "test")
        for file_name in os.listdir(os.path.join(test, "pos")):
            with open(os.path.join(test, "pos", file_name), "r", encoding="UTF-8") as file:
                f = _adjust_data(file.read())
                max_len = len(f) if len(f) > max_len else max_len
                for word in f:
                    unique.add(word)
                    full.append(word)
                data.append({"label": "positive", "value": f})
        for file_name in os.listdir(os.path.join(test, "neg")):
            with open(os.path.join(test, "neg", file_name), "r", encoding="UTF-8") as file:
                f = _adjust_data(file.read())
                max_len = len(f) if len(f) > max_len else max_len
                for word in f:
                    unique.add(word)
                    full.append(word)
                data.append({"label": "negative", "value": f})
        
        word_count = Counter(full)
        sorted_vocab = sorted(word_count, key = word_count.get, reverse=True)
        int_to_vocab = {ii:word for ii, word in enumerate(sorted_vocab)}
        vocab_to_int = {word:ii for ii, word in int_to_vocab.items()}
        db["test"] = data
        db["unique"] = unique
        db["int_to_vocab"] = int_to_vocab
        db["vocab_to_int"] = vocab_to_int
        db["max_len"] = max_len
        
        with open(os.path.join(p, "data"), "wb") as file:
            pickle.dump(db, file)

class GenerationDataset(Dataset):


    def __init__(self, process_data=False, test=False) -> None:
        self.file_path_tmp = "/root/ucd/Interesno/ecs189_w2024/stage4/text_generation/data_split"
        
        
        with open(self.file_path_tmp, "rb") as file:
            data = pickle.load(file)
        self.data = data["test" if test else "train"]
        self.int_to_vocab = data["int_to_vocab"]
        self.vocab_to_int = data["vocab_to_int"]
        self.end_word = "0end0"
        self.length_of_seq = 3
    
    
    # def __init__(self) -> None:
    #     self.file_path_tmp = (
    #         "/root/ucd/Interesno/ecs189_w2024/stage4/text_generation"
    #     )
    #     with open(os.path.join(self.file_path_tmp, "data"), "r", encoding="UTF-8") as file:
    #         self.data = file.read()
    #     self.int_to_vocab = None
    #     self.vocab_to_int = None
    #     self.end_word = "0end0"
    #     self.length_of_seq = 3
    #     self._process_data_to_3(self.data) # 4644 unique values


    def _process_data(self):
        data = self.data
        data = data.translate(str.maketrans("", "", string.punctuation))
        data = re.sub(r"\d+", lambda x: x[0] + " ", data)
        data = re.sub(r"\W=", " ", data).lower()
        word_count = Counter(re.split(r"\W+", data))
        sorted_vocab = sorted(word_count, key = word_count.get, reverse=True)
        self.int_to_vocab = {ii:word for ii, word in enumerate(sorted_vocab)}
        self.vocab_to_int = {word:ii for ii, word in self.int_to_vocab.items()}
        data = data.split("\n")
        data = list(map(lambda x: re.split(r"\W+", x)[1:], data))
        self.data = list(map(lambda x: x.remove("") if "" in x else x, data))

    def _process_data_to_3(self, text):
        text = text.translate(str.maketrans("", "", string.punctuation)) # remove punctiatoin
        text = re.sub(r"\d+", "", text) # remove numbers and digits
        text = re.sub(r"\W=", " ", text).lower()
        text = re.sub(r"\n", " " + self.end_word + " ", text)
        word_count = Counter(re.split(r"\W+", text))
        sorted_vocab = sorted(word_count, key = word_count.get, reverse=True)
        self.int_to_vocab = {ii:word for ii, word in enumerate(sorted_vocab)}
        self.vocab_to_int = {word:ii for ii, word in self.int_to_vocab.items()}
        text = re.split(r"\W+", text)
        data = [] 
        for ind, _ in enumerate(text[:-(self.length_of_seq)]):
            if self.end_word not in (text[ind+i] for i in range(self.length_of_seq)):
                data.append([text[ind+i] for i in range(self.length_of_seq + 1)])
        
        train, test = train_test_split(data, test_size=0.3, random_state=43)
        db = {"train": train, "test": test, 
              "int_to_vocab": self.int_to_vocab, "vocab_to_int": self.vocab_to_int}
        with open(os.path.join(self.file_path_tmp, "data_split"), "wb") as file:
            pickle.dump(db, file)
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tmp = self.data[index]
        item = [self.vocab_to_int[i] for i in tmp[:-1]]
        label = self.vocab_to_int[tmp[-1]]
        return torch.tensor(item), one_hot(torch.tensor(label), len(self.int_to_vocab)).float()
        
        


        
# if __name__ == "__main__":
#     # classificationRearrange()
#     # a = GenerationDataset()
#     # print(f"{a.data=}")
#     # print(f"{a.vocab_to_int=}")


#     a = ClassificationDataset()
#     # a = GenerationDataset() 
    
#     r = torch.randint(len(a), size=(5,))
#     # print(f"F{a.unique=}")
    
#     # for i in r:
#     #     print(a[i][0].size())
#     #     print(a[i][1].size())
#     m = set()
#     for i in a:
#         m.add([i[0].size()][0])
#     # print(list(a[0][0].size()))    
#     print(m)
      
      
        
#     # print(len(a.unique))
#     # print(f"f{a}") 
#     # print(f"F{a.unique=}")
#     # print(os.listdir("/root/ucd/Interesno/ecs189_w2024/stage4/text_classification/train/neg/"))
