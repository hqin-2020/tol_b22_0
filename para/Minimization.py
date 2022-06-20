import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
np.set_printoptions(suppress = True)

def minimization(start):

    start = start[0]
    
    obs_series = pd.read_csv('data.csv', delimiter=',')
    obs_series = np.array(obs_series.iloc[:,1:]).T

    def init_kf(θ):
    
        λ, η, b11, b21, As11, As12, As13, As14, As21, As22, As23, As24, As31, As32, As33, As34, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = θ
        ones = np.ones([3,1])
        
        Ass = np.array([[As11, As12, As13],\
                        [As21, As22, As23],\
                        [As31, As32, As33]])
        Aso = np.array([[As14],\
                        [As24],\
                        [As34]])
        Bs =  np.array([[Bs11, 0,    0],\
                        [Bs21, Bs22, 0],\
                        [Bs31, Bs32, Bs33]])
        
        μs = sp.linalg.solve(np.eye(3) - Ass, Aso) 
        Σs = sp.linalg.solve_discrete_lyapunov(Ass, Bs@Bs.T)
        
        β = sp.linalg.solve(np.hstack([Σs@np.array([[1,1],[0,-1],[-1,0]]), ones]).T, np.array([[0,0,1]]).T)                                     
        γ1 = np.array([[1],[0],[0]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
        γ2 = np.array([[0],[1],[0]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
        γ3 = np.array([[0],[0],[1]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
        Γ = np.hstack([γ1, γ2, γ3])
        
        Z01 = β.T@(obs_series[:,[0]] - μs)
        Σz01 = 0
        Z02 = η/(1-λ)
        Σz02 = b21**2/(1-λ**2)
        S0 = Γ.T@(obs_series[:,[0]] - μs) + μs
        Σs0 = (1/(ones.T@np.linalg.inv(Σs)@ones))[0][0]
        
        μ0 = np.array([[Z01[0][0]],\
                    [Z02],\
                    [S0[0][0]],\
                    [S0[1][0]],\
                    [S0[2][0]],\
                    [1]])
        Σ0 = np.array([[Σz01,0,    0,   0,   0,   0],\
                    [0,   Σz02, 0,   0,   0,   0],\
                    [0,   0,    Σs0, Σs0, Σs0, 0],\
                    [0,   0,    Σs0, Σs0, Σs0, 0],\
                    [0,   0,    Σs0, Σs0, Σs0, 0],\
                    [0,   0,    0,   0,   0,   0]])    

        return μ0, Σ0, Ass, Σs

    def Kalman_Filter(obs, D, F, A, B, μ0, Σ0):

        state_μ = np.zeros([A.shape[1], obs.shape[1]])
        state_μ[:,[0]] = μ0
        state_Σ = np.zeros([A.shape[1], A.shape[1], obs.shape[1]])
        state_Σ[:,:,0] = Σ0

        ll = 0

        for t in range(obs.shape[1]-1):
            μt = state_μ[:,[t]]
            Σt = state_Σ[:,:,t]
            KΣt = (A@Σt@D.T + B@F.T)@np.linalg.inv(D@Σt@D.T+F@F.T)
            state_μ[:,[t+1]] = A@μt + KΣt@(obs[:,[t+1]] - D@μt)
            state_Σ[:,:,t+1] = A@Σt@A.T + B@B.T - (A@Σt@D.T + B@F.T)@np.linalg.inv(D@Σt@D.T + F@F.T)@(D@Σt@A.T+F@B.T)

            Ω = D@Σt@D.T + F@F.T
            ll += (-0.5*obs_series.shape[0]*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(Ω)) - 0.5*(obs[:,[t+1]] - D@μt).T@np.linalg.inv(Ω)@(obs[:,[t+1]] - D@μt))

        return state_μ, state_Σ ,ll

    def ll(θ):

        λ, η, b11, b21, As11, As12, As13, As14, As21, As22, As23, As24, As31, As32, As33, As34, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = θ

        A = np.array([[1,   1,   0,     0,     0,     0],\
                    [0,   λ,   0,     0,     0,     η],\
                    [0,   0,   As11,  As12,  As13,  As14],\
                    [0,   0,   As21,  As22,  As23,  As24],\
                    [0,   0,   As31,  As32,  As33,  As34],\
                    [0,   0,   0,     0,     0,     1]])
        B = np.array([[b11, 0,     0,     0],\
                    [b21, 0,     0,     0],\
                    [0,   Bs11,  0,     0],\
                    [0,   Bs21,  Bs22,  0],\
                    [0,   Bs31,  Bs32,  Bs33],\
                    [0,   0,     0,   0]])
        D = np.array([[1,   1,   As11,  As12,  As13,  As14],\
                    [1,   1,   As21,  As22,  As23,  As24],\
                    [1,   1,   As31,  As32,  As33,  As34]])
        F = np.array([[b11, Bs11,  0,     0],\
                    [b11, Bs21,  Bs22,  0],\
                    [b11, Bs31,  Bs32,  Bs33]])

        μ0, Σ0, _, _ = init_kf(θ)
        
        _, _, ll = Kalman_Filter(obs_series, D, F, A, B, μ0, Σ0)

        return -ll[0][0]


    def callback(x):
        fobj = ll(x)
        θseries.append(x)
        llseries.append(fobj)

    bnds = ((0,1),(-5,5),\
            (-5,5),(-5,5),\
            (-5,5),(-5,5),(-5,5),(-5,5),\
            (-5,5),(-5,5),(-5,5),(-5,5),\
            (-5,5),(-5,5),(-5,5),(-5,5),\
            (-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5))

    θseries = []
    llseries = []
    
    try:
        μ0, Σ0, Ass, Σs = init_kf(start)
    except:
        return ['Singular', θseries, llseries, start]

    if (np.all(np.linalg.eigvals(Σ0)>=0)) & (np.all(abs(np.linalg.eigvals(Ass))<1) & (np.all(np.linalg.eigvals(Σs)>=0))):
        try:
            θ_opt = sp.optimize.minimize(ll, start, method = 'L-BFGS-B', bounds = bnds, callback = callback, tol=1e-9)  
            return [θ_opt, θseries, llseries, start]
        except:
            return ['Failure', θseries, llseries, start]
    else:
        return ['Unstable', θseries, llseries, start]

    
    
            
    