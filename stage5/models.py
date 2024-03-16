from torch_geometric.datasets import Planetoid as dataset
from torch_geometric.loader import DataLoader
import torch


from sklearn.metrics import recall_score, precision_score
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_labels, model_configuration = 0):
        super().__init__()
        self.name = "GCN" 
        self.num_of_labels = num_labels
        self.model_configuration = model_configuration
        if model_configuration in (0, 1):
            self.conv1 = GCNConv(num_features, 50)
            self.conv2 = GCNConv(50, 16)
            self.conv3 = GCNConv(16, num_labels)
        elif model_configuration == 2:
            self.conv1 = GCNConv(num_features, 17)
            self.conv2 = GCNConv(17, num_labels)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.model_configuration in (0, 2):        
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.log_softmax(x, dim=1) if self.model_configuration == 0 else F.softmax(x, dim=1)
        elif self.model_configuration == 1:
            x = self.conv1(x, edge_index)
            x = F.relu(x) 
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index)
            x = F.softmax(x, dim=1)
            
            
            
        return x


def train(d, model, optimizer, loss_function):
    model.train()
    avg_loss = 0

    
    y_pred = model(d)
    loss = loss_function(y_pred[d.train_mask], d.y[d.train_mask])
    avg_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return avg_loss

@torch.no_grad
def test(d, model, loss_function,  dataset_name, is_test=True, loss="ce"):
    model.eval()
    correct = 0
    num_of_batches = 1
    loss, recall, precision = 0, 0, 0
    prediction = model(d)
    
    
    y_true = d.y[d.test_mask if is_test else d.train_mask].float()
    y_pred = prediction[d.test_mask if is_test else d.train_mask].argmax(1).float()
    p = prediction[d.test_mask if is_test else d.train_mask]
    loss += loss_function(p.argmax(1).float() if loss == "ce" else p , y_true.long() if loss!="ce" else y_true).item()
    correct += (
        (y_pred == y_true).type(torch.float32).sum().item()
    )
    # recall += recall_score(y_true, y_pred, average="macro", labels=range(len(data_loader.dataset.int_to_vocab)), zero_division=True)
    precision += precision_score(
        y_true, y_pred, average="macro", zero_division=True
    )
    recall += recall_score(
        y_true, y_pred, average="macro", labels=range(model.num_of_labels), zero_division=True
    )


    accuracy = 100 * correct / len(y_true)
    recall /= num_of_batches / 100
    precision /= num_of_batches / 100
    f1 = 2 * precision * recall / (precision + recall)

    mode = "Test" if is_test else "Train"

    print(
        f"""
        Metrics for {model.name} model on {mode} data ({dataset_name}):
        -- Accuracy: {accuracy:>.2f} %
        -- Recall: {recall:>.2f} %
        -- Precision: {precision:>.2f} %
        -- F1: {f1:>.2f} %
        """
    )
    return accuracy


def run(model_name: str, loss_fn, lr=0.01, epochs=20, model_configuration=0 ,loss="ce", disable_msg=False):
    model_name = model_name.lower()
    if model_name == "cora":
        model = GCN(1433, 7, model_configuration=model_configuration)
        data = dataset(root="/Cora", name="Cora")
    elif model_name =="citeseer":
        model = GCN(3703, 6, model_configuration=model_configuration)
        data = dataset(root="/CiteSeer", name="CiteSeer")
    elif model_name == "pubmed":
        model = GCN(500, 3, model_configuration=model_configuration)
        data = dataset(root="/PubMed", name="PubMed")
    else:
        print(f"Error, model {model_name} is unknown")
        return
    data = data[0].to("cpu")
    model.to("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_loss = []
    for i in range(epochs):
        if not disable_msg:
            print(f"Epoch {i+1} ---------")
        
        l = train(data, model, opt, loss_fn) 
        train_loss.append(l)
        if not disable_msg:
            print(f"Training loss: {train_loss[i]}")
    test(data, model, loss_fn, model_name, loss=loss)
    test(data, model, loss_fn, model_name, loss=loss , is_test=False)
    
    plt.plot(train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (cross entropy)")
    plt.title(f"Loss convergence plot for {model_name}; configuration {model.model_configuration}")
    plt.show()    


if __name__ == "__main__":
    run("citeseer", torch.nn.NLLLoss(), loss="lol" )
    


























