import numpy as np

def als(X,k,lambda_,max_iter,threshold):
    
    """
 Output:
           U --- n x k matrix
           V --- k x d matrix
 Input:
           X --- n x d input matrix
           W --- n x d binary matrix indicating known elements in X
           Lambda --- Ridge regularizer parameter 
    """
    def solve_V(X,W,U):
        X = X.values
        n,d = X.shape
        V = np.zeros((d,k))
        X = X.T
        W = W.T.values
        for j,x_j in enumerate(X):
            v_j = np.linalg.solve(U[W[j]].T.dot(U[W[j]])+lambda_*np.eye(k), U[W[j]].T.dot(x_j[W[j]]))
            V[j] = v_j
        return V

    def solve_U(X,W,V):
        X = X.values
        W = W.values
        n,d = X.shape
        U = np.zeros((n,k))
        for i,x_i in enumerate(X):
            u_i = np.linalg.solve(V[W[i]].T.dot(V[W[i]])+lambda_*np.eye(k), V[W[i]].T.dot(x_i[W[i]]))
            U[i] = u_i
        return U

    W = ~X.isnull()
    n,d = X.shape
    U = np.ones((n,k))
    V = solve_V(X,W,U)
    n_known = W.sum().sum()
    MSE = ((X - U.dot(V.T)).pow(2).sum().sum()*1.0)/n_known
    MSEs=[MSE]
    for i in range(max_iter):
        U_new = solve_U(X,W,V)
        V_new = solve_V(X,W,U)
        MSE_new = ((X - U_new.dot(V_new.T)).pow(2).sum().sum()*1.0)/n_known
        if (MSE - MSE_new) < MSE*threshold:
            #MSEs.append(MSE_new)
            break
        else:
            MSEs.append(MSE_new)
            MSE = MSE_new
            U = U_new
            V = V_new
    #print "Error history",MSEs
    return U,V.T

