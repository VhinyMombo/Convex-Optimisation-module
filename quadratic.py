import numpy as np
import matplotlib.pyplot as plt
class quadratic :
    def __init__(self, gamma):
        self.gamma = gamma
    def loss(self,X):
        return (X**2 * self.gamma).sum()
    def grad(self,X):
        return 2*X*self.gamma
    def hessian(self,X):
        return np.diag(2*self.gamma)
    def backtracking(self, X, DX, alpha, beta):
        t = 1
        while self.loss(X + t * DX) >= self.loss(X) + alpha * t * self.grad(X).T.dot(DX):
            t = beta * t
        return t
        
    
    def grad_descent(self, X0 ,n_iter = 300, beta = 0.1, alpha = 0.1, tol = 1e-5):
        X = X0
        X_store = []
        X_store.append(X0)
        loss = [self.loss(X)]
        #print(n_iter)
        for i in range(n_iter):
            #print(i)
            DX = -self.grad(X)
            t = self.backtracking(X, DX, alpha, beta)
            X = X+t * DX
            X_store.append(X)
            loss.append(self.loss(X))
            if self.loss(X) < tol:
                break
        return X_store, loss
    
    def newton_method(self,X0,n_iter = 300,alpha = 0.1, beta = 0.1, tol =1e-3):
        X = X0
        X_store = []
        X_store.append(X0)
        loss = [self.loss(X)]
        for i in range(n_iter):
            print(i)
            DX = -np.linalg.inv(self.hessian(X)).dot(self.grad(X))
            lamba2 = self.grad(X).T.dot(np.linalg.inv(self.hessian(X)).dot(self.grad(X)))
            if lamba2/2 <= tol:
                return X_store, loss
            ### line search 
            t = self.backtracking(X, DX, alpha, beta)
            X = X+t * DX
            X_store.append(X)
            loss.append(self.loss(X))
        return X_store, loss
            
                