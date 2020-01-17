import matplotlib.pyplot as plt 

def plotData(X, y):
    plt.figure()
    plt.plot(X, y, 'rx', marker='x', ms = 10)
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")
    plt.show(block=False)