import math
from math import sqrt

#Kernels
def gaussian(x):
    return math.exp(-x**2/2)/sqrt(2*math.pi)
def uniform(x):
    return 0 if abs(x) >= 1 else 1/2
def triangular(x):
    return 0 if abs(x) > 1 else 1-abs(x)
def epanechnikov(x):
    return 0 if abs(x) > 1 else 3/4*(1-x ** 2)
def quartic(x):
    return 0 if abs(x) > 1 else 15/16*(1-x ** 2)** 2
def triweight(x):
    return 0 if abs(x) > 1 else 35/32*(1-x ** 2)** 3
def custom(x):
    return math.exp(-x**2)/sqrt(math.pi)
def tricube(x):
    return 0 if abs(x) > 1 else 70/81*(1-abs(x) ** 3)** 3
def cosine(x):
    return 0 if abs(x) > 1 else math.pi/4*math.cos(math.pi/2*x)
def logistic(x):
    return  1/(math.e ** x +2 +math.e ** (-x))
def sigmoid(x):
    return 2/math.pi * 1/(math.e ** x + math.e ** (-x))
kernels = {'uniform':uniform, 'triangular':triangular, 'epanechnikov':epanechnikov, 
           'quartic':quartic, 'triweight':triweight, 'tricube':tricube, 'gaussian':gaussian, 
           'cosine':cosine, 'logistic':logistic, 'sigmoid':sigmoid}

def minkowskiDistance(X,Y,p):
    dist = 0
    for x,y in zip(X,Y):
        dist += abs(x-y) ** p
    return dist ** (1/p)
def manhattanDistance(x,y):
    return minkowskiDistance(x,y,1)
def euclideanDistance(x,y):
    return minkowskiDistance(x,y,2)
def cosDistance(x,y):
    dist = 0
    a,b,c = 0,0,0
    for i in range(len(x)):
        a+=x[i]*y[i]
        b+=x[i]*x[i]
        c+=y[i]*y[i]
    return a/(sqrt(b)*sqrt(c))
def chebyshevDistance(x,y):
    dist = 0
    for x,y in zip(X,Y):
        dist = max(dist,abs(x-y))
    return dist
distances = {'manhattan':manhattanDistance, 'euclidean':euclideanDistance, 'chebyshev':chebyshevDistance}

def predict(xtr, ytr, test,K = gaussian, dist = manhattanDistance, h = 0.1):
    a = 0
    b = 0
    for x,y in zip(xtr, ytr):
        a+=y*K(dist(x,test)/h)
        b+=K(dist(x,test)/h)
    return 0 if b == 0 else a/b
    

n, m = [int(x) for x in input().split(" ")]
xs = [[int(v) for v in input().split(" ")] for x in range(n)]
q = [int(v) for v in input().split(" ")]
ys = [x[-1] for x in xs]
xs = [x[:-1] for x in xs]
dist = distances[input()]
kernel = kernels[input()]
typeH = input()
num = int(input())

if (typeH == 'variable'):
    dists = [dist(x,q) for x in xs]
    dists.sort()
    num = dists[num]

print(predict(xs,ys,q, K = kernel, dist=dist, h = num))



