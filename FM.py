from RMSE import RMSE
import numpy as np

class FM():
    def __init__(self, N, K, data, y, alpha, beta, train_ratio=0.75, iterations=100, tolerance=0.005, l2_reg=True, verbose=True):
        self.K = K          # Number of latent factors
        self.N = N          # Number of x (variables)
        self.n_cases = len(data)            # N of observations
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.l2_reg = l2_reg
        self.tolerance = tolerance
        self.verbose = verbose
        # w 초기화
        self.w = np.random.normal(scale=1./self.N, size=(self.N))
        # v 초기화
        self.v = np.random.normal(scale=1./self.K, size=(self.N, self.K))
        # Train/Test 분리
        cutoff = int(train_ratio * len(data))
        self.train_x = data[:cutoff]
        self.test_x = data[cutoff:]
        self.train_y = y[:cutoff]
        self.test_y = y[cutoff:]

    def test(self):                                     # Training 하면서 RMSE 계산 
        # SGD를 iterations 숫자만큼 수행
        best_RMSE = 10000
        best_iteration = 0
        training_process = []
        for i in range(self.iterations):
            rmse1 = self.sgd(self.train_x, self.train_y)        # SGD & Train RMSE 계산
            rmse2 = self.test_rmse(self.test_x, self.test_y)    # Test RMSE 계산     
            training_process.append((i, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.6f ; Test RMSE = %.6f" % (i+1, rmse1, rmse2))
            if best_RMSE > rmse2:                       # New best record
                best_RMSE = rmse2
                best_iteration = i
            elif (rmse2 - best_RMSE) > self.tolerance:  # RMSE is increasing over tolerance
                break
        print(best_iteration, best_RMSE)
        return training_process
        
    # w, v 업데이트를 위한 Stochastic gradient descent 
    def sgd(self, x_data, y_data):
        y_pred = []
        for data, y in zip(x_data, y_data):
            x_idx = data[0]
            x_0 = np.array(data[1])     # xi axis=0 [1, 2, 3]
            x_1 = x_0.reshape(-1, 1)    # xi axis=1 [[1], [2], [3]]
    
            # biases
            bias_score = np.sum(self.w[x_idx] * x_0)
    
            # score 계산
            vx = self.v[x_idx] * (x_1)
            sum_vx = np.sum(vx, axis=0)
            sum_vx_2 = np.sum(vx * vx, axis=0)
            latent_score = 0.5 * np.sum(np.square(sum_vx) - sum_vx_2)

            # 예측값 계산
            y_hat = bias_score + latent_score
            y_pred.append(y_hat)
            error = y - y_hat
            # w, v 업데이트
            if self.l2_reg:
                self.w[x_idx] += error * self.alpha * (x_0 - self.beta * self.w[x_idx])
                self.v[x_idx] += error * self.alpha * ((x_1) * sum(vx) - (vx * x_1) - self.beta * self.v[x_idx])
            else:
                self.w[x_idx] += error * self.alpha * x_0
                self.v[x_idx] += error * self.alpha * ((x_1) * sum(vx) - (vx * x_1))
        return RMSE(y_data, y_pred)
            
    def test_rmse(self, x_data, y_data):
        y_pred = []
        for data , y in zip(x_data, y_data):
            y_hat = self.predict(data[0], data[1])
            y_pred.append(y_hat)
        return RMSE(y_data, y_pred)

    def predict(self, idx, x):
        x_0 = np.array(x)
        x_1 = x_0.reshape(-1, 1)

        # biases
        bias_score = np.sum(self.w[idx] * x_0)

        # score 계산
        vx = self.v[idx] * (x_1)
        sum_vx = np.sum(vx, axis=0)
        sum_vx_2 = np.sum(vx * vx, axis=0)
        latent_score = 0.5 * np.sum(np.square(sum_vx) - sum_vx_2)

        # 예측값 계산
        y_hat = bias_score + latent_score
        return y_hat