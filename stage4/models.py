import torch
from torch import nn
from dataSets import GenerationDataset, ClassificationDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, precision_score


class GenerationRNN(nn.Module):

    def __init__(self, num_of_words):
        super(GenerationRNN, self).__init__()
        # constants:
        # desciriptions:

        self.name = "Generation RNN"

        self.num_of_words = num_of_words  # total number of words
        self.seq_len = 3  # constant

        self.num_of_layers = 1

        self.hidden_size = 250  # poke around
        self.embed_size = 500  # (= input_size); can poke around

        self.embed = nn.Embedding(self.num_of_words, self.embed_size)
        self.rn = nn.RNN(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_of_layers,
            nonlinearity="relu",
            # dropout=0.2,
            batch_first=True,
        )

        self.lin = nn.Linear(
            self.hidden_size, self.num_of_words
        )  # convert output into prob vector
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x, prev):
        embed = self.embed(x)
        out, h_n = self.rn(embed, prev)
        out = out[:, -1, :]  # extract the last layer
        out = self.lin(out)  # convert to prob
        return out, h_n

    def init_state(self, batch_size):
        return torch.zeros((self.num_of_layers, batch_size, self.hidden_size))


class GenerationLSTM(nn.Module):

    def __init__(self, num_of_words):
        super(GenerationLSTM, self).__init__()
        self.name = "Generation LSTM"
        self.seq_len = 3  # constant
        self.num_of_words = num_of_words  # constant

        self.num_of_layers = 1

        self.hidden_size = 250  # can poke around
        self.embed_size = 500  # can poke around

        self.embed = nn.Embedding(self.num_of_words, self.embed_size)
        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_of_layers,
            # dropout=0.2,
            batch_first=True,
        )

        self.fc = nn.Linear(self.hidden_size, self.num_of_words)  # create output

    def forward(self, x, prev_state):
        embed = self.embed(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits[:, -1, :], state

    def init_state(self, batch_size):
        return (
            torch.zeros(self.num_of_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_of_layers, batch_size, self.hidden_size),
        )


class GenerationGRU(nn.Module):

    def __init__(self, num_of_words, configuration=0) -> None:
        """
        configuration: different variations of RNN
        -- 0: default, one layer, 250 embed size 500 hidden size, dropout 0
        -- 1: two layers, 250 embed, 500 hidden dropout 0.2
        -- 2: two layers, 50 embed, 500 hidden, dropout 0.2
        -- 3: two layers, 250 embed, 1000 hidden, dropout 0.2 (can't run on my machine)
        -- 4: 5 layers, 100 embed, 100 hidden, dropout 0.3
        
        """

        super(GenerationGRU, self).__init__()
        self.name = f"Generation GRU conf {configuration}"
        self.seq_len = 3  # constant
        self.num_of_words = num_of_words  # constant

        if configuration == 0:
            self.num_of_layers = 1
            self.dropout = 0
            self.embed_size = 250  # can poke around
            self.hidden_size = 500  # can poke around
        elif configuration == 1:
            self.num_of_layers = 2
            self.dropout = 0.2
            self.embed_size = 250
            self.hidden_size = 500
        elif configuration == 2:
            self.num_of_layers = 2
            self.dropout = 0.2
            self.embed_size = 50
            self.hidden_size = 500
        # elif configuration == 3:
        #     self.num_of_layers = 2
        #     self.dropout = 0.2
        #     self.embed_size = 250
        #     self.hidden_size = 1000
        elif configuration == 4:
            self.num_of_layers = 5
            self.dropout = 0.3
            self.embed_size = 100
            self.hidden_size = 100
        elif configuration == 5:
            self.num_of_layers = 5
            self.dropout = 0.2
            self.embed_size = 250
            self.hidden_size = 250


        # elif configuration == 2:

        self.embed = nn.Embedding(self.num_of_words, self.embed_size)
        self.gru = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_of_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, self.num_of_words)

    def forward(self, x, prev):
        tmp = self.embed(x)
        output, state = self.gru(tmp, prev)
        output = self.fc(output)
        return output[:, -1, :], state

    def init_state(self, batch_size):
        return torch.zeros(self.num_of_layers, batch_size, self.hidden_size)


class ClassificationRNN(nn.Module):

    def __init__(self, num_of_words) -> None:
        super(ClassificationRNN, self).__init__()
        self.name = "Classification RNN"
        self.seq_len = 100
        self.num_of_words = num_of_words

        self.num_of_layers = 1

        self.hidden_size = 250
        self.embed_size = 100

        self.embed = nn.Embedding(self.num_of_words, self.embed_size)
        self.rn = nn.RNN(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_of_layers,
            dropout=0.2,
            nonlinearity="relu",
            batch_first=True,
        )

        # self.fc = nn.Linear(self.hidden_size, 2)
        # self.sig = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.ReLU(),
        )

    def forward(self, x, prev_state):
        embed = self.embed(x)

        output, state = self.rn(embed, prev_state)
        logits = self.fc(output)
        # state = self.fc(state)
        return logits[:, -1, :], state

    def init_state(self, batch_size):
        return torch.zeros(self.num_of_layers, batch_size, self.hidden_size)


class ClassificationLSTM(nn.Module):

    def __init__(self, num_of_words) -> None:
        super(ClassificationLSTM, self).__init__()
        self.name = "Classification LSTM"
        self.seq_len = 100
        self.num_of_words = num_of_words

        self.num_of_layers = 1

        self.hidden_size = 250
        self.embed_size = 250

        self.embed = nn.Embedding(self.num_of_words, self.embed_size)
        self.rn = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_of_layers,
            # dropout=0.2,
            batch_first=True,
        )

        self.fc = nn.Linear(self.hidden_size, 2)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hidden_size, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 2),
        #     nn.ReLU(),
        # )

    def forward(self, x, prev_state):
        embed = self.embed(x)

        output, state = self.rn(embed, prev_state)
        logits = self.fc(output)
        return logits[:, -1, :], state

    def init_state(self, batch_size):
        return (
            torch.zeros(self.num_of_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_of_layers, batch_size, self.hidden_size),
        )


class ClassificationGRU(nn.Module):

    def __init__(self, num_of_words) -> None:
        super(ClassificationGRU, self).__init__()
        self.name = "Classification GRU"
        self.seq_len = 100
        self.num_of_words = num_of_words

        self.num_of_layers = 1

        self.hidden_size = 250
        self.embed_size = 250

        self.embed = nn.Embedding(self.num_of_words, self.embed_size)
        self.rn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_of_layers,
            dropout=0.2,
            batch_first=True,
        )

        self.fc = nn.Linear(self.hidden_size, 2)
        self.sig = nn.Sigmoid()

    def forward(self, x, prev_state):
        embed = self.embed(x)

        output, state = self.rn(embed, prev_state)
        logits = self.fc(output)
        state = self.sig(state)
        return logits[:, -1, :], state

    def init_state(self, batch_size):
        return torch.zeros(self.num_of_layers, batch_size, self.hidden_size)


def train(data_loader, model, optimizer, loss_function, hidden, is_lstm=False):
    model.train()
    avg_loss = 0
    for _, (X, y) in enumerate(data_loader):

        if is_lstm:
            hidden = hidden[0].detach(), hidden[1].detach()
        else:
            hidden = hidden.detach()  # the most important line of code xD

        y_pred, hidden = model(X, hidden)
        loss = loss_function(y_pred, y)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss / len(data_loader)


def test(data_loader, model, loss_function, bs, mode="Test"):
    model.eval()
    correct = 0
    num_of_batches = len(data_loader)
    size = len(data_loader.dataset)
    loss, recall, precision = 0, 0, 0
    with torch.no_grad():
        h = model.init_state(batch_size=bs)
        for _, (X, y) in enumerate(data_loader):
            prediction, _ = model(X, h)
            loss += loss_function(prediction, y).item()

            y_true = y.argmax(1)
            y_pred = prediction.argmax(1)
            correct += (
                (prediction.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()
            )
            # recall += recall_score(y_true, y_pred, average="macro", labels=range(len(data_loader.dataset.int_to_vocab)), zero_division=True)
            precision += precision_score(
                y_true, y_pred, average="macro", zero_division=True
            )
            recall += recall_score(
                y_true, y_pred, average="macro", labels=range(2), zero_division=True
            )

    accuracy = 100 * correct / len(data_loader.dataset)
    recall /= num_of_batches / 100
    precision /= num_of_batches / 100
    f1 = 2 * precision * recall / (precision + recall)
    loss /= num_of_batches

    print(
        f"""
        Metrics for model {model.name} on {mode} data:
        -- Accuracy: {accuracy:>.2f} %
        -- Recall: {recall:>.2f} %
        -- Precision: {precision:>.2f} %
        -- F1: {f1:>.2f} %
        -- Loss: {loss:>.2f}
        """
    )
    return accuracy


def create_sentence(words, model, dataloader):
    sentence = words
    words = torch.LongTensor([[dataloader.dataset.vocab_to_int[i] for i in words]])
    h0 = model.init_state(1)
    curr_word = ""
    i = 0
    while curr_word != dataloader.dataset.end_word and i < 100:
        prediction, _ = model(words[:, -3:], h0)
        prediction = prediction.argmax(1).item()
        words = torch.cat((words, torch.LongTensor([[prediction]])), 1)
        curr_word = dataloader.dataset.int_to_vocab[prediction]
        sentence.append(curr_word)
        i += 1
    # return sentence
    return " ".join(sentence[:-1]).replace("0end0", "")


if __name__ == "__main__":

    bs = 32

    # train_data = ClassificationDataset()
    # test_data = ClassificationDataset(test=True)
    # train_data = DataLoader(train_data, batch_size=bs, drop_last=True)
    # test_data = DataLoader(test_data, batch_size=bs, drop_last=True)

    train_data = GenerationDataset()
    test_data = GenerationDataset(test=True)

    train_data = DataLoader(train_data, batch_size=len(train_data))
    test_data = DataLoader(test_data, batch_size=len(test_data))

    # train_data = DataLoader(train_data, batch_size=bs, drop_last=True)
    # test_data = DataLoader(test_data, batch_size=bs, drop_last=True)

    # model = GenerationRNN(len(train_data.dataset.int_to_vocab))
    # model = GenerationLSTM(len(train_data.dataset.int_to_vocab))
    model = GenerationGRU(len(train_data.dataset.int_to_vocab), configuration=5)

    # model = ClassificationGRU(len(train_data.dataset.int_to_vocab))

    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 20

    train_loss = []
    for i in range(epochs):

        h = model.init_state(len(train_data.dataset))
        # h = model.init_state(bs)
        print(f"Epoch {i+1} ----------")
        l = train(train_data, model, opt, lf, h)
        print(f"Loss: {l}")
        train_loss.append(l)

    test(test_data, model, lf, len(test_data.dataset))
    test(train_data, model, lf, len(train_data.dataset), mode="Train")

    print(f"{create_sentence(['fish', 'walks', 'into'], model, train_data)} \n")
    # print(f"{create_sentence(['what', 'did', 'the'], model, train_data)} \n")

    plt.plot(train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (cross entropy)")
    plt.title(f"Loss convergence plot for {model}")
    plt.show()
