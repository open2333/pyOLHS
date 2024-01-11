from scipy.spatial.distance import cdist
from datetime import datetime
import random
import numpy as np
import copy
import math


class OLHS:
    def __init__(self,bound:list,population:int=10,iteration:int=1,StratiType:str="center",initseed=None,optseed=None):
        self.real_bound=np.array(bound,float)
        self.scaled_bound=self.scaleBound(bound)
        self.pop=population
        self.iteration=iteration
        self.type=StratiType
        self.initial_sample=None
        self.dim=self.real_bound.shape[0]
        self.m_bestfJ=[]
        self.m_bestfIn=[]
        self.m_bestfOuter=[]
        self.seed1=int((datetime.now() - datetime(1970, 1, 1)).total_seconds()) if initseed is None else initseed
        self.seed2=int((datetime.now() - datetime(1970, 1, 1)).total_seconds()) if optseed is None else optseed
    @staticmethod
    def restore_inputs(bound, scaled_bound, norm_inputs) -> np.ndarray:

        norm_inputs = np.array(norm_inputs,float)
        pro = np.array(bound)
        pro_t = np.transpose(pro)
        norm_pro = np.array(scaled_bound)
        norm_pro_t = np.transpose(norm_pro)
        unnorm_in = (norm_inputs - norm_pro_t[0]) / ((norm_pro_t[1] - norm_pro_t[0]) / (pro_t[1] - pro_t[0])) + pro_t[0]
        return unnorm_in

    @staticmethod
    def scaleBound(real_boundary: list, norm_low=0.0,
                   norm_up=1.0) :
        pro = np.array(real_boundary,float)
        pro_t = np.transpose(pro)
        norm_pro_t = copy.deepcopy(pro_t)
        if len(real_boundary[0]) == 3:
            norm_pro_t[2] = np.full_like(norm_pro_t[0], (norm_up - norm_low) ) / (
                        norm_pro_t[1] - norm_pro_t[0]) * (
                                norm_pro_t[2])
        else:
            pass
        norm_pro_t[0] = np.full_like(norm_pro_t[0], norm_low)
        norm_pro_t[1] = np.full_like(norm_pro_t[1], norm_up)
        norm_pro = np.transpose(norm_pro_t)
        return norm_pro

    @staticmethod
    def ith_in_2_combinations(n, i):
        i = n * (n - 1) // 2 - i - 1
        t = int((math.sqrt(8 * i + 1) - 1) / 2)
        return (n - t - 2, n - 1 - (i - t * (t + 1) // 2))
    @staticmethod
    def unique_random_n_N_ref(n, N,seed=None):
        np.random.seed(seed)
        random.seed(seed)
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
        random.seed(self.seed1)
        np.random.seed(self.seed1)
        if self.scaled_bound.shape[1] == 2:
            tp1 = (self.scaled_bound[:, 1] - self.scaled_bound[:, 0]) / self.pop
            tp2 = np.linspace(0, self.pop - 1, self.pop).reshape(1, self.pop)
            raw_1 = tp1.reshape(-1, 1) @ tp2 + np.tile(self.scaled_bound[:, 0], (self.pop, 1)).T
            if self.type == "random":
                raw_3 = np.abs(np.random.rand(self.scaled_bound.shape[0], self.pop))
            elif self.type == "center":
                raw_3 = np.ones((self.scaled_bound.shape[0], self.pop)) * 0.5
            else:
                raw_3 = np.ones((self.scaled_bound.shape[0], self.pop)) * 0.5
            samples = (raw_1 + np.tile(tp1, (self.pop, 1)).T * raw_3).T
            self.initial_sample=samples
            return self.initial_sample
        if self.scaled_bound.shape[1]==3:
            properties_list = np.array(self.scaled_bound)
            b = (properties_list[:, 1] - properties_list[:, 0]) / properties_list[:, 2]
            c = np.ceil(self.pop // b).astype('int')
            lhd = np.zeros((self.dim, self.pop))

            for i in range(self.dim):
                if(self.pop>b[i]):
                    lhd[i] = np.append(np.tile(np.append(
                    np.arange(properties_list[i][0], properties_list[i][1], properties_list[i][2]),
                    properties_list[i][1]), c[i]), np.random.permutation(
                    np.arange(properties_list[i][0], properties_list[i][1], properties_list[i][2])))[
                         0:self.pop]
                    random.shuffle(lhd[i])
                else:
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
                    random.shuffle(lhd[i])
            self.initial_sample=lhd.transpose()
            return self.initial_sample
    @staticmethod
    def mindis(A: np.ndarray) -> np.ndarray:
        SqED = cdist(A, A, 'sqeuclidean')
        SqED[SqED < 1e-4] = 0
        d = np.ravel(SqED)
        d = d[d != 0]
        s = pow(d, -1)
        Fx = pow(0.5*np.sum(s), 0.5)
        return Fx

    @staticmethod
    def sq2(A: np.ndarray,B:np.ndarray) -> np.ndarray:
        SqED = cdist(A, B,'sqeuclidean')
        SqED[SqED < 1e-4] = 0
        d = np.ravel(SqED)
        d = d[d != 0]
        s = pow(d, -1)
        Fx = np.sum(s)
        return Fx

    def partialmin(self,origin_A_,past_evaluate,col,row1,row2)-> np.ndarray:
        origin_A=copy.deepcopy(origin_A_)
        past_evaluate=past_evaluate**2
        p1=past_evaluate-self.sq2(origin_A,origin_A[row1:row1+1,:])-self.sq2(origin_A,origin_A[row2:row2+1,:])
        origin_A[row1,col],origin_A[row2,col]=origin_A[row2,col],origin_A[row1,col]
        p1+=self.sq2(origin_A,origin_A[row1:row1+1,:])+self.sq2(origin_A,origin_A[row2:row2+1,:])
        return np.power(p1,0.5)
    def sampling(self)-> np.ndarray:

        random.seed(self.seed2)
        if self.initial_sample is None:
            self.initial_lhs()
        #print("init", self.initial_sample)
        X=self.initial_sample.copy(order='K')
        X_best=X.copy(order='K')
        m_xOldBest = X_best.copy(order='K')
        fX_oldbest=fX_best=fX=fX0=self.mindis(X)
        Th=0.005*fX0
        q=0
        not_improved=0
        ne = (self.pop * (self.pop - 1) / 2)
        J_t = np.ceil(ne / 5)
        temp1=None
        J_max=50
        M_max=100
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
                X_try=X.copy(order='K')
                fX_try=10e9
                indexcombinations=self.unique_random_n_N_ref(J,ne,self.seed2)
                for j in range(J):
                    temp=self.ith_in_2_combinations(self.pop,indexcombinations[j])
                    currentValue = self.partialmin(X, fX, mod, temp[0], temp[1])
                    if (currentValue < fX_try):
                        fX_try = currentValue
                        temp1 = temp
                X_try[temp1[0],mod],X_try[temp1[1],mod]=X[temp1[1],mod],X[temp1[0],mod]
                if(fX_try - fX <= Th * random.random()):
                    X = X_try.copy(order='K')
                    fX = fX_try
                    n_acpt += 1
                    if (fX < fX_best):
                        X_best = X.copy(order='K')
                        fX_best = fX
                        n_imp += 1
                i+=1
            if(fX_oldbest - fX_best > tol):
                not_improved = 0
                m_xOldBest = X_best.copy(order='K')
                fX_oldbest = fX_best
                X = X_best.copy(order='K')
                fX = fX_best
                Th *= 0.8 if (n_acpt >= (0.1 * M)) and (n_imp < n_acpt) else 1.25
            else:
                not_improved+=1
                Th *= 1.43 if n_acpt < (0.1 * M) else 0.9 if n_acpt >= (0.8 * M) else 1
            if(not_improved>=bare):
                m_xOldBest_ = self.restore_inputs(self.real_bound, self.scaled_bound, m_xOldBest)
                return m_xOldBest_
            q+=1
        m_xOldBest_ = self.restore_inputs(self.real_bound, self.scaled_bound, m_xOldBest)
        return m_xOldBest_
if __name__=="__main__":
    bound = [[45,70,5],[15,25,1],[80,90,1]]
    a = OLHS(bound, 50, 500,"center",initseed=1)
    print(a.scaled_bound)
    c = a.sampling()
    print(c)
    import matplotlib.pyplot as plt
    plt.scatter(c[:, 0], c[:, 1])
    plt.show()
