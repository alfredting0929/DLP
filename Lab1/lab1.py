import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1)

class sigmoid():
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, output_grad):
        return output_grad * self.output * (1 - self.output)
        
class relu():
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, x):
        self.input = x
        self.output = np.maximum(self.input, 0)
        return self.output
    
    def backward(self, output_grad):
        relu_grad = float((self.input > 0))
        input_grad = output_grad * relu_grad
        return input_grad

class LinearLayer():
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features)
        self.bias = np.zeros((1, out_features))
        # Use when the optimizer is Momentum
        self.vw = np.zeros((in_features, out_features))
        self.vb = np.zeros((1, out_features))

    def forward(self, x):
        self.input = x
        output = np.dot(self.input, self.weight) + self.bias
        return output
    
    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = np.dot(self.input.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0, keepdims=True)
        return input_grad
    
    def update(self, learning_rate, optimizer):
        # Gradient descent optimizer
        if optimizer == 'GD':
            self.weight -= learning_rate * self.weight_grad
            self.bias -= learning_rate * self.bias_grad
        # Momentum optimizer
        if optimizer == 'Momentum':
            self.vw = 0.9 * self.vw - learning_rate * self.weight_grad
            self.weight += self.vw
            self.vb = 0.9 * self.vb - learning_rate * self.bias_grad
            self.bias += self.vb
    
class MSE():
    def __init__(self):
        self.pred = None
        self.label = None

    def forward(self, pred, label):
        self.pred = pred
        self.label = label
        output = np.mean((self.pred - self.label) ** 2)
        return output
    
    def backward(self):
        return 2 * (self.pred - self.label) / np.size(self.label)
    
class Network():
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, learning_rate, optimizer):
        self.fc1 = LinearLayer(input_dim, hidden1_dim)
        self.sigmoid1 = sigmoid()
        self.fc2 = LinearLayer(hidden1_dim, hidden2_dim)
        self.sigmoid2 = sigmoid()
        self.fc3 = LinearLayer(hidden2_dim, output_dim)
        self.sigmoid3 = sigmoid()
        self.loss = MSE()
        self.lr = learning_rate
        self.optimizer = optimizer

    def forward(self, x):
        # First hidden layer (Fully Connected Layer)
        x = self.fc1.forward(x)
        x = self.sigmoid1.forward(x)
        # Second hidden layer (Fully Connected Layer)
        x = self.fc2.forward(x)
        x = self.sigmoid2.forward(x)
        # Output layer
        x = self.fc3.forward(x)
        x = self.sigmoid3.forward(x)
        return x
    
    def compute_loss(self, pred, label):
        loss = self.loss.forward(pred, label)
        return loss
    
    def backward(self):
        grad = self.loss.backward()
        grad = self.sigmoid3.backward(grad)
        grad = self.fc3.backward(grad)
        grad = self.sigmoid2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.sigmoid1.backward(grad)
        grad = self.fc1.backward(grad)

    def update(self):
        self.fc1.update(learning_rate=self.lr, optimizer=self.optimizer)
        self.fc2.update(learning_rate=self.lr, optimizer=self.optimizer)
        self.fc3.update(learning_rate=self.lr, optimizer=self.optimizer)

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
    
def train(model, num_epoch, batch_size, train_data, train_label):
    train_acc_list = []
    train_loss_list = []

    for epoch in range(1, num_epoch + 1):
        train_acc = 0
        train_loss = 0

        for iter in range(len(train_data) // batch_size):
            # Forward process
            pred = model.forward(train_data[iter * batch_size:(iter + 1) * batch_size])
            train_acc += np.sum(np.round(pred) == train_label[iter * batch_size:(iter + 1) * batch_size])   
            # Compute the loss       
            batch_loss = model.compute_loss(pred, train_label[iter * batch_size:(iter + 1) * batch_size])
            train_loss += batch_loss
            # Backward process
            model.backward()
            # Update the weights
            model.update()
    
        if epoch % 5000 == 0:
            print('Epoch: {}    loss: {}'.format(epoch, train_loss / (iter + 1)))
            # print('Epoch: {}    Loss: {}    Acc: {}'.format(epoch, train_loss / (iter + 1), train_acc / len(train_data)))

        # Save the training history
        train_loss_list.append(train_loss / (iter + 1))
        train_acc_list.append(train_acc / len(train_data))

    # Plot the learning curve
    x = np.linspace(1, num_epoch, num_epoch)
    plt.suptitle('Learning curve')
    plt.subplot(1,2,1)
    plt.title('Loss')
    plt.plot(x, train_loss_list)
    plt.subplot(1,2,2)
    plt.title('Accuracy')
    plt.plot(x, train_acc_list)
    plt.show()

def test(model, x, y):
    pred_y = model.forward(x)
    # Print the ground truth and prediction
    for i in range(len(pred_y)):
        print('Ground truth: {} |   prediction: {}'.format(y[i], float(pred_y[i])))
    pred_y = np.round(pred_y)
    print('Accuracy: {}%'.format(np.sum(pred_y == y) / len(y) * 100))
    # Plot the figure
    show_result(x, y, pred_y)

if __name__ == '__main__':
    l = 1
    if l:
        # Task 1 for linear dataset
        # Initiate model weights and setting hyperparameters
        print('Task 1 for linear dataset')
        model1 = Network(input_dim=2, hidden1_dim=10, hidden2_dim=10, output_dim=1, learning_rate=0.01, optimizer='GD')
        num_epoch = 100000
        batch_size = 100

        # Generate training and testing dataset
        train_data, train_label = generate_linear(n=100)

        # Training part
        train(model=model1, num_epoch=num_epoch, batch_size=batch_size, train_data=train_data, train_label=train_label)
        # Testing part
        test(model=model1, x=train_data, y=train_label)



    if not l:
        # Task 2 for XOR dataset
        # Initiate model weights and setting hyperparameters
        print('Task 2 for XOR dataset')
        model2 = Network(input_dim=2, hidden1_dim=10, hidden2_dim=10, output_dim=1, learning_rate=0.01, optimizer='GD')
        num_epoch = 100000
        batch_size = 21

        # Generate training and testing dataset
        train_data, train_label = generate_XOR_easy()

        # Training part
        train(model=model2, num_epoch=num_epoch, batch_size=batch_size, train_data=train_data, train_label=train_label)
        # Testing part
        test(model=model2, x=train_data, y=train_label)


    # Evaluation part for new testing dataset
    if 0:
        test_data, test_label = generate_linear(n=100)
        acc = 0
        for i in range(len(test_data)):
            pred = model1.forward(test_data[i])
            if np.round(pred) == test_label[i]:
                acc += 1
        print('Acc: {}'.format(acc/len(test_data)))

    


