import sys, json
from random import randint
import numpy as np

#configures output np arrays into outputable string lists
def outstr(a):
	ow = []
	for n in a.tolist():
		ow.append(('%.4f'%n) + '\n')
	return ow

#basis function for linear regression (linear extraoplation)
def phij(x,j):
	return [1]+list(x)

#analytic solution to linear regression
def analytic(X,Y):
	PHI = []
	w = []
	for i in range(len(X)):
		PHI.append(phij(X[i],i))
	PHI = np.array(PHI)
	w = np.linalg.inv(PHI.transpose().dot(PHI)).dot(PHI.transpose()).dot(Y)
	return w

#stocastic gradient desent for linear regression
def grad_desent(X,Y,p):
	pX = []
	for i in range(len(X)):
		pX.append(phij(X[i],0))
	w = [1.]*(len(pX[0]))
	for n in range(int(p["num iter"])):
		r = randint(0,len(pX)-1)
		for i in range(len(w)):
			w[i] += float(p["learning rate"]) * ( Y[r] - float(np.array(w).transpose().dot(np.array(pX[r]))) ) * (pX[r][i])
	return np.array(w)


if __name__ == "__main__":
	data = []
	target = []
	hypp = {}
	w_an = []
	w_gd = []
	f_out = sys.argv[1][0:sys.argv[1].index(".")+1] + "out"
	
	#load in and sort data and target values
	with open(sys.argv[1]) as f:
		for s in f.readlines():
			n = s.split()
			data.append([float(x) for x in n[0:-1]])
			target.append(float(n[-1]))
	
	#load in gradient descent hyper parameters
	with open(sys.argv[2]) as j:
		hypp = json.load(j)
	
	#calculates the weights using each method
	w_an = outstr(analytic(data,target))
	w_gd = outstr(grad_desent(data,target,hypp))
	w_gd[-1] = w_gd[-1][0:-1]
	
	print(w_an)
	print(w_gd)
	
	#outputs both sets of weights to .out file
	with open(f_out,'w') as o:
		o.writelines(w_an)
		o.write("\n")
		o.writelines(w_gd)
	
