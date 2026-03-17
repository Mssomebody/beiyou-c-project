import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_derivative(x): return x*(1-x)

class MLP:
    def __init__(self, i, h, o, lr=0.5):
        self.lr, self.W1, self.b1 = lr, np.random.randn(i,h)*np.sqrt(2/i), np.zeros((1,h))
        self.W2, self.b2 = np.random.randn(h,o)*np.sqrt(2/h), np.zeros((1,o))
    
    def forward(self, X):
        self.h = sigmoid(np.dot(X,self.W1)+self.b1)
        return sigmoid(np.dot(self.h,self.W2)+self.b2)
    
    def backward(self, X, y, out):
        d2 = (y-out)*sigmoid_derivative(out)
        self.W2 += self.lr*np.dot(self.h.T,d2)
        d1 = np.dot(d2,self.W2.T)*sigmoid_derivative(self.h)
        self.W1 += self.lr*np.dot(X.T,d1)
        return np.mean((y-out)**2)

# 验证代码
X,y = np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]])
mlp = MLP(2,4,1)
print("开始训练XOR问题...")
for i in range(10000):
    loss = mlp.backward(X,y,mlp.forward(X))
    if i%2000==0: print(f"Epoch {i:5d}, Loss: {loss:.6f}")
pred = mlp.forward(X)
print(f"\n最终预测:\n{pred.round()}")
print(f"目标值:\n{y}")
print(f"准确率: {np.mean((pred.round()==y)*100):.0f}%")
input("按回车键退出...")