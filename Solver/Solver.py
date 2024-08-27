import numpy as np
from typing import Callable

class AdamMomentum:
    def __init__(self, dL:Callable, lr:float=0.001, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, 
                 weight_decay: float = 0, lambda_l2: float = 0, epochs: int = 1000):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.lambda_l2 = lambda_l2
        self.m = 0
        self.v = 0
        self.dL = dL
        self.iteration = 1
        self.epochs = epochs
        self.global_error_tolerance = 1e-5
        self.theta_result = []
        self.m_result = []
        self.v_result = []

    def __global_error__(self, theta_new: float , theta_old: float) -> float:
            diff = theta_new - theta_old
            return np.abs(diff)

    def solve(self, theta_initial):

        theta = theta_initial
        while self.iteration <= self.epochs:
            
            self.theta_result.append(theta)
            self.m_result.append(self.m)
            self.v_result.append(self.v)

            theta_old = theta
            dL_value = self.dL(theta)
            
            if self.lambda_l2 != 0:
                dL_value  += self.lambda_l2 / 2 * theta

            self.m = self.beta1 * self.m + (1 - self.beta1) * dL_value
            self.v = self.beta2 * self.v + (1 - self.beta2) * (dL_value ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.iteration)
            v_hat = self.v / (1 - self.beta2 ** self.iteration)
            
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            if self.weight_decay != 0:
                theta = (1 - self.weight_decay) * theta - update
            else:
                theta -= update
            
            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {global_error}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration}, Error: {global_error}.')

        return self.theta_result, self.m_result, self.v_result, self.iteration
    

class RMSPropMomentum:
    def __init__(self, dL:Callable, lr:float=0.001, beta:float=0.9, epsilon:float=1e-8, weight_decay: float = 0, lambda_l2: float = 0, epochs: int = 1000):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.lambda_l2 = lambda_l2
        self.v = 0
        self.dL = dL
        self.iteration = 1
        self.epochs = epochs
        self.global_error_tolerance = 1e-5
        self.theta_result = []
        self.v_result = []

    def __global_error__(self, theta_new: float , theta_old: float) -> float:
            diff = theta_new - theta_old
            return np.abs(diff)

    def solve(self, theta_initial):

        theta = theta_initial
        while self.iteration <= self.epochs:
            
            self.theta_result.append(theta)
            self.v_result.append(self.v)

            theta_old = theta
            dL_value = self.dL(theta)
            
            if self.lambda_l2 != 0:
                dL_value  += self.lambda_l2 / 2 * theta

            self.v = self.beta * self.v + (1 - self.beta) * (dL_value ** 2)            
            update = self.lr * dL_value / (np.sqrt(self.v) + self.epsilon)
            
            if self.weight_decay != 0:
                theta = (1 - self.weight_decay) * theta - update
            else:
                theta -= update
            
            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {global_error}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration}, Error: {global_error}.')

        return self.theta_result, self.v_result, self.iteration
    
class AdaGrad:
    def __init__(self, dL: Callable, lr: float = 0.01, epsilon: float = 1e-8, lr_decay: float = 0,
                 lambda_l2: float = 0, epochs: int = 1000):
        self.lr = lr
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.lambda_l2 = lambda_l2
        self.dL = dL
        self.accumulated_gradients = 0
        self.iteration = 1
        self.epochs = epochs
        self.global_error_tolerance = 1e-5
        self.theta_result = []
        self.accumulated_gradients_result = []

    def __global_error__(self, theta_new: float, theta_old: float) -> float:
        diff = theta_new - theta_old
        return np.abs(diff)

    def solve(self, theta_initial):
        theta = theta_initial
        while self.iteration <= self.epochs:
            
            self.theta_result.append(theta)
            self.accumulated_gradients_result.append(self.accumulated_gradients)

            theta_old = theta
            dL_value = self.dL(theta)

            if self.lambda_l2 != 0:
                dL_value += self.lambda_l2 / 2 * theta

            self.accumulated_gradients += dL_value ** 2
            update = self.lr * dL_value / (np.sqrt(self.accumulated_gradients) + self.epsilon)

            if self.lr_decay != 0:
                theta -= 1/(1 + (self.iteration - 1) * self.lr) * update
            else:
                theta -= update
            
            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {global_error}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration}, Error: {global_error}.')

        return self.theta_result, self.accumulated_gradients_result, self.iteration