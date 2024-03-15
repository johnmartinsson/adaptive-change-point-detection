import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn1  = nn.BatchNorm1d(input_size)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.bn1(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc2(x)
        return x
    
class MLPClassifier():
    def __init__(self, input_size, hidden_size, output_size, verbose=True, max_epochs=100, patience=10, batch_size=64):
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model     = MLP(input_size, hidden_size, output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        
        self.verbose    = verbose
        self.max_epochs = max_epochs
        self.patience   = patience
        self.batch_size = batch_size
    
    def fit(self, X, y):
        """Fit the model to the data using Adam optimizer"""

        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).long().to(self.device)

        # normalize the data to zero mean unit variance
        #X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-10)

        dataset = torch.utils.data.TensorDataset(X, y)
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

        best_valid_loss = float('inf')
        best_epoch = 0
        patience = 0
        for epoch in range(self.max_epochs):
            # train the model
            self.model.train()
            running_train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_dataset_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss = loss.item()
                running_train_loss += train_loss
            
            running_train_loss /= len(train_dataset_loader)
            if self.verbose:
                print('Epoch: {:d} | Training loss: {:.3f}'.format(epoch+1, running_train_loss))

            # validate the model
            with torch.no_grad():
                self.model.eval()
                running_valid_loss = 0.0
                for batch_idx, (data, target) in enumerate(valid_dataset_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    valid_loss = loss.item()
                    running_valid_loss += valid_loss
                
                running_valid_loss /= len(valid_dataset_loader)
                if self.verbose:
                    print('Epoch: {:d} | Validation loss: {:.3f}'.format(epoch+1, running_valid_loss))
                
                if running_valid_loss < best_valid_loss:
                    best_valid_loss = running_valid_loss
                    best_epoch      = epoch
                    patience        = 0
                else:
                    patience += 1

            if patience >= self.patience:
                print('early stopping ..., best validation loss: {:.3f}, best epoch {}'.format(best_valid_loss, best_epoch))
                break

    def predict_proba(self, x):
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.model(x).numpy()
        
    def predict(self, x):
        x = torch.from_numpy(x).float()
        return torch.argmax(self.predict_proba(x), dim=1).numpy()