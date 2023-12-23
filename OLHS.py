from scipy.spatial.distance import cdist
from datetime import datetime
import random
import numpy as np
import copy
import math

class OLHS:
    def __init__(self,bound:list,population:int=10,iteration:int=1,StratiType:str="center"):
        self.bound=np.array(bound)
        self.pop=population
        self.iteration=iteration
        self.type=StratiType
        self.initial_sample=None
        self.dim=self.bound.shape[0]
        self.m_bestfJ=[]
        self.m_bestfIn=[]
        self.m_bestfOuter=[]
    @staticmethod
    def ith_in_2_combinations(n, i):
        i = n * (n - 1) // 2 - i - 1
        t = int((math.sqrt(8 * i + 1) - 1) / 2)
        return (n - t - 2, n - 1 - (i - t * (t + 1) // 2))
    @staticmethod
    def unique_random_n_N_ref(n, N):
        seed = int((datetime.now() - datetime(1970, 1, 1)).total_seconds())
        np.random.seed(seed)
        index = []
        number = 0
        while n:
            probability = n / N
            p = np.random.uniform(0.0, 1.0)
            if p < probability:
                index.append(number)
                n -= 1
            N -= 1
            number += 1
        random.shuffle(index)
        return index
    def initial_lhs(self):
        if self.bound.shape[1] == 2:
            tp1 = (self.bound[:, 1] - self.bound[:, 0]) / self.pop
            tp2 = np.linspace(0, self.pop - 1, self.pop).reshape(1, self.pop)
            raw_1 = tp1.reshape(-1, 1) @ tp2 + np.tile(self.bound[:, 0], (self.pop, 1)).T
            if self.type == "random":
                raw_3 = np.abs(np.random.rand(self.bound.shape[0], self.pop))
            elif self.type == "center":
                raw_3 = np.ones((self.bound.shape[0], self.pop)) * 0.5
            else:
                raw_3 = np.ones((self.bound.shape[0], self.pop)) * 0.5
            samples = (raw_1 + np.tile(tp1, (self.pop, 1)).T * raw_3).T
            self.initial_sample=samples
            return self.initial_sample
        if self.bound.shape[1]==3:
            properties_list = np.array(self.bound)
            b = (properties_list[:, 1] - properties_list[:, 0]) / properties_list[:, 2]
            c = np.ceil(self.pop // b).astype('int')
            lhd = np.zeros((self.dim, self.pop))

            for i in range(self.dim):
                if(self.pop>b[i]):
                    lhd[i] = np.append(np.tile(np.append(
                    np.arange(properties_list[i][0], properties_list[i][1], properties_list[i][2]),
                    self.properties_list[i][1]), c[i]), np.random.permutation(
                    np.arange(properties_list[i][0], properties_list[i][1], properties_list[i][2])))[
                         0:self.pop]

                else:
                    # print("b[i]",b[i])
                    d = int(b[i]) // int(self.pop)
                    r = int(b[i]) % int(self.pop)
                    index = list(range(self.pop))
                    index1 = copy.deepcopy(index)
                    random.shuffle(index1)
                    subindex = index1[:r]
                    result = []
                    cum = 0
                    for j in index:
                        if (j in subindex):
                            p = random.randint(0, d + 1)
                            s = (j * d + cum + p) * properties_list[i][2]
                            cum += 1
                        else:
                            p = random.randint(0, d)
                            # print("i",j)
                            s = (j * d + cum + p) * properties_list[i][2]
                        result.append(s)
                    random.shuffle(result)
                    lhd[i]=result
            self.initial_sample=lhd.transpose()
            return self.initial_sample

    def mindis(self,A: np.ndarray) -> np.ndarray:
        SqED = cdist(A, A, 'sqeuclidean')
        SqED[SqED < 1e-4] = 0
        d = np.ravel(SqED)
        d = d[d != 0]
        s = pow(d, -1)
        Fx = pow(0.5*np.sum(s), 0.5)
        return Fx
    def sq2(self,A: np.ndarray,B:np.ndarray) -> np.ndarray:
        SqED = cdist(A, B, 'sqeuclidean')
        SqED[SqED < 1e-4] = 0
        d = np.ravel(SqED)
        d = d[d != 0]
        s = pow(d, -1)
        Fx = np.sum(s)
        return Fx

    def partialmin(self,origin_A_,past_evaluate,col,row1,row2):
        origin_A=copy.deepcopy(origin_A_)
        past_evaluate=past_evaluate**2
        p1=past_evaluate-self.sq2(origin_A,origin_A[row1:row1+1,:])-self.sq2(origin_A,origin_A[row2:row2+1,:])
        origin_A[row1,col],origin_A[row2,col]=origin_A[row2,col],origin_A[row1,col]
        p1+=self.sq2(origin_A,origin_A[row1:row1+1,:])+self.sq2(origin_A,origin_A[row2:row2+1,:])
        return np.power(p1,0.5)

    def sampling(self):
        if self.initial_sample is None:
            self.initial_lhs()
        X=self.initial_sample.copy()
        X_best=X.copy()
        m_xOldBest = X_best.copy()
        fX_oldbest=fX_best=fX=fX0=self.mindis(X)
        Th=0.05*fX0
        q=0
        not_improved=0
        ne = (self.pop * (self.pop - 1) / 2)
        J_t = np.ceil(ne / 5)
        temp1=None
        J_max=50
        M_max=50
        J = math.floor(min(J_max, max(J_t, ne)) if J_t < J_max else J_max)
        M = math.floor(min(M_max, int(np.ceil(2 * ne * self.dim / J))))
        bare=int(self.iteration/10)
        while(q<self.iteration):
            tol=0
            i=0
            n_acpt=0
            n_imp=0
            while(i<M):
                mod=i%self.dim
                X_try=X.copy()
                fX_try=10e9
                indexcombinations=self.unique_random_n_N_ref(J,ne)
                for j in range(J):
                    temp=self.ith_in_2_combinations(self.pop,indexcombinations[j])
                    currentValue = self.partialmin(X, fX, mod, temp[0], temp[1])
                    if (currentValue < fX_try):
                        fX_try = currentValue
                        temp1 = temp
                X_try[temp1[0],mod],X_try[temp1[1],mod]=X_try[temp1[1],mod],X_try[temp1[0],mod]
                if(fX_try - fX <= Th * random.random()):
                    X = X_try.copy()
                    fX = fX_try
                    n_acpt += 1
                    if (fX < fX_best):
                        X_best = X.copy()
                        fX_best = fX
                        n_imp += 1
                i+=1
            if(fX_oldbest - fX_best > tol):
                not_improved = 0
                m_xOldBest = X_best.copy()
                fX_oldbest = fX_best
                X = X_best.copy()
                fX = fX_best
                Th *= 0.8 if (n_acpt >= (0.1 * J)) and (n_imp < n_acpt) else 1.25
            else:
                not_improved+=1
                Th *= 1.43 if n_acpt < (0.1 * J) else 0.9 if n_acpt >= (0.8 * J) else 1
            if(not_improved>=bare):
                return m_xOldBest

            q+=1
        return m_xOldBest
if __name__=="__main__":
    bound = [[0, 1.0,], [0, 1.0]]
    a = OLHS(bound, 20, 500,"center")
    c = a.sampling()
    print(c)
    import matplotlib.pyplot as plt
    plt.scatter(c[:, 0], c[:, 1])
    plt.show()
