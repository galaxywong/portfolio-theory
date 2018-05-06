#%%
#modern portfolio theory
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

#%%
#3 stocks with prices of 10 days, same format of stock price stored as csv file.
#np.random.seed(0)
##price = np.random.rand(30).reshape((3,10))*20
#price = np.array([[1,2,3],[4,5,6]])
#stock = {}
#for i in range(len(price)):
#    stock['stock'+str(i)] = price[i]
#stock = pd.DataFrame(stock)
    
#%%
#caculate log rate of return;the argument is a DataFrame with everyday price of 
#all stocks,like the variable stock;return a DataFrame with everyday log rate 
def log_return(stockprice):
    return np.log(stockprice.shift(-1)/stockprice)*100 #which is the same datatype as stockprice

#%%
#calculate weighted variance
def weighted_variance(weight, logreturn):
    cov = logreturn.cov()
    #weight = weight.reshape(len(weight),1)
    return np.dot(np.dot(weight.T, cov), weight) #scail to 10000

#%%
#calculate the weighted average log return
def weighted_return(weight, logreturn, m):
    #weight = weight.reshape(len(weight),1)#make sure the dim of weight is(m,1)
    avlogreturn = np.mean(logreturn).values.reshape(m,1)
    
    return np.dot(weight.T, avlogreturn)

#%%
#constraint1:the sum of weights equals to 1
def constraint1(weight):
    return np.sum(weight) - 1
#constrain2 limits the variance
def constraint2(weight, logreturn, std):
    return -( weighted_variance(weight, logreturn) - std**2)
def constraint3(weight,logreturn,m):
    return weighted_return(weight, logreturn, m)
    
#%%
def objective(weight, logreturn,m):
   return -weighted_return(weight, logreturn, m)
#%%
#given the variance, calculate the weight to maximize return
def optimize_weight(stock, std):
    m = np.shape(stock)[1] #the number of stocks, every column is a stock
    weight0 = np.array([1/m for i in range(m)]).reshape(m,1)
    logreturn = log_return(stock)
    b = (0.0, 1.0)
    bnds = tuple([b for i in range(m)])
    con1 = {'type': 'eq', 'fun': constraint1} 
    con2 = {'type': 'ineq', 'fun': constraint2, 'args':[logreturn, std]}
    con3 = {'type':'ineq', 'fun':constraint3, \
            'args':(logreturn, m)}
    cons = ([con1,con2,con3])
    solution = minimize(objective,weight0,args =(logreturn,m), method='SLSQP',\
                    bounds=bnds,constraints=cons)
    if solution.success == True:    #optimizer succeeds
        return {'weight':solution.x,'expect_return':-solution.fun}
    else: #message : description of the cause of the termination
        print('The optimizer fails because:')
        print(solution.message)
        return {'message':solution.message}
#%%
def effective_set(stock, minstd, maxstd, step):
    std = np.arange(maxstd,minstd,step)
    effset = {}
    for i in range(len(std)):
        effset[str(std[i])] = optimize_weight(stock, std[i])
        if 'message' in effset[str(std[i])] :
            continue
        effset[str(std[i])]['std'] = std[i]
    return effset
#%%
def sharpe(effset, rf):
    sp = {}
    for i in effset:
        if 'message' in effset[i]: # exclude the ineffective items
            continue
        expect_return = effset[i]['expect_return']
        std = effset[i]['std']
        sp[str(i)] = (expect_return-rf)/std
    return sp  #a dict with the same key as effset
#%% 
def max_sharpe(sp):
    maxsp = max(zip(sp.values(),sp.keys()))
    return {'std':maxsp[1],'max_sharpe':maxsp[0]} #{'std':sharpe}
#%%
def plot_effset(effset):
    sns.regplot('std','expect_return',data=pd.DataFrame(effset).T, fit_reg=False)
    plt.title('The Effective Set')
#%%
#the main part
rf = 1 #free risk return,percent
minstd = 1.45
maxstd = 2
step = -0.05 #std,same as logreturn,percent
stock = pd.read_csv('stockprice.csv')
effset = effective_set(stock, minstd, maxstd,step)
sp = sharpe(effset, rf)
maxsp = max_sharpe(sp)
print('The weights of stocks are:')
print(effset[maxsp['std']]['weight'])

plot_effset(effset)
    
        