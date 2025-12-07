import numpy as np

e = np.log(1)

def exponential_regression(x,y):
    est = np.polyfit(x,np.log(y),1)

    res = np.sum((np.exp(np.polyval(est,x))-y)**2)

    return(np.exp(est),res)

def poly_regression(x,y):
    est = np.polyfit(x,y,6)
    res = np.sum((np.polyval(est,x)-y)**2)
    return(est, res)

def power_regression(x,y):
    est=np.polyfit(np.log(x),np.log(y),1)
    res = np.sum((np.polyval(est,np.log(x))-np.log(y))**2)
    est[-1] = np.exp(est[-1])
    return(est, res)

if __name__=="__main__":
    a = np.arange(1,11)
    b = 1+2*a**2
    c = a**3

    for k in [a,b,c]:
        print(power_regression(a,k))
        print("###########")
