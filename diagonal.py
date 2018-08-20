import numpy as np

class Diagonal:
    def __init__(self,n,d,precision='single',sparse=True):
        self.n = n
        self.d = d
        self.area = n*d
        self.dims = 2
        self.shape = [self.n*self.d,self.n*self.d]
        self.precision = np.float32 if 'single' else np.float64
        self.sparse= sparse
        self.permutation_flag = 1
        self.table = self._create_table()
        self.matrix = self._create_matrix()
        self.basis = self._create_basis()
        self.row_based = self._create_row_representation()


    def _create_basis(self):
        '''
            Create the basis vector utilized throughout the implementation
        '''
        return np.split(np.array(self.table[1][:self.area]),self.d)

    def _create_row_representation(self):
        row_wise = np.zeros((self.n,self.area))
        for ii,i in enumerate(np.repeat(range(self.matrix.shape[1]),self.d)):
            row_wise[:,ii] = self.matrix[:,i][ii%self.d::self.d]
        return row_wise
    def _create_table(self):
        '''
            This function will create a lookup table of indicies of where the values in the sparse
            representation exist in the full matrix
        '''
        rows = np.repeat(range(self.area),self.n)
        columns = [list(range(i,self.area,self.d)) for i in range(self.d)]
        columns = [i for c in columns for i in c] * self.n
        return rows,columns

    def _create_zero_matrix(self):
        return np.zeros((self.shape)).astype(self.precision)


    def _create_matrix(self):

        if 1 in[self.n,self.d]:
            raise AssertionError('n or d cannot be equal to 1')
        populated = self.n**2 * self.d
        values = np.random.randn(populated).astype(self.precision)
        sparse= np.reshape(values,(self.area, self.n))
        return sparse

    def from_dense(self,arr):
        '''
            This function will take a dense representation of the sparse matrix
            and recreate the sparse matrix
        '''
        matrix = self._create_zero_matrix()
        matrix[self.table] = arr.flatten()
        return matrix

    def _sparse_mm(self,y,left=False,batch = True):
        if batch:
            if left:
                batched = np.vstack([y[:,b] for b in self.basis])
                batched = np.reshape(batched,(self.d,y.shape[-1],self.n))
            else:
                batched = np.vstack([y[b,:] for b in self.basis])
                batched = np.reshape(batched,(self.d,self.n,y.shape[-1]))
        # For testing purposes, we can perform the mm operation on a pre indexed matrix
        else:
            batched=y
        if left:
            return np.vstack([np.dot(batched[ii%self.d],i) for ii,i in enumerate(self.row_based.T)]).T
        else:
            return np.vstack([np.dot(i,batched[ii%self.d]) for ii,i in enumerate(self.matrix)])

    @staticmethod
    def _swap_elements(l,i,j):
        new = l.copy()
        new[i],new[j] = j,i
        return new

    def _create_diag(self):
        return np.repeat(np.arange(0,self.n),self.d)

    def get_permutation_dense(self):
        ''' 
            To increase numerical stability and ensure that A is proper factorized during LU decomposition, we pivot and return 
            the pivoted dense matrix
        '''
        diags = np.repeat(np.arange(0,self.n),self.d)
        updated = self.row_based.copy()
        for c in range(updated.shape[1]):
            max_idx = np.argmax(np.abs(self.row_based[diags[c]:,c]))
            max_idx += diags[c]
            if (max_idx != diags[c]) and (max_idx > diags[c] ):
                self.permutation_flag *= -1

                order = self._swap_elements(list(range(updated.shape[0])),max_idx,diags[c])
                basis = self.basis[c%self.d]
                updated[:,basis] = updated[:,basis][[order],:]
        return updated


    def mm(self,y,x=None,sparse=True,left=False,batch=True):
        if sparse:
            return self._sparse_mm(y,left=left,batch=batch)
        else:
            assert x is not None
            return np.dot(x,y)


    def to_dense(self,mat):
        dense = np.zeros((self.area,self.n))
        for i in range(self.d):
            dense[self.basis[i%self.d],:] = mat[i%self.d::self.d,self.basis[i%self.d]]
        return dense
    def _backwards_solve(self,A,b):
        diags = self._create_diag()
        x = np.zeros(b.shape)
        for i in range(b.shape[0])[::-1]:
            basis = self.basis[i%self.d][::-1]
            subset = [s for s in basis if s >i]
            x[i] = (b[i] -(A[i][diags[i]+1:] *x[subset][::-1].T).sum()) / A[i,diags[i]]
        return x


    def backwards(self,A,b):
        return np.hstack([self._backwards_solve(A,b[:,i])[...,None] for i in range(len(b))])

    def _forward_solve(self,A,column):
            x = np.zeros(column.shape)
            diags = self._create_diag()
            for i in range(column.shape[0]):
                basis = self.basis[i%self.d]
                subset = [s for s in basis if s <i]
                x[i] = (column[i] - (A[i][:diags[i]] *x[subset].T).sum()) / A[i,diags[i]]
            return x
    def forward(self,A,b):
        return np.hstack([self._forward_solve(A,b[:,i])[...,None] for i in range(len(b))])

    def to_sparse(self,mat):
        '''
            Returns a sparse matrix from dense when representation is column based
        '''
        sparse = np.zeros((self.area,self.area))
        assert mat.shape == (self.n,self.area)
        for i in range(sparse.shape[0]):
            sparse[self.basis[i%self.d],i] = mat[:,i]
        return sparse



    def solve(self,b):
        ''' 
            Currently this solve function operates with dense L and U values, but orgingal sparsity in y and x.
            Further memory reductions are possible if the forward solve returns a dense y and the backwards
            solves operates on the dense y.  For the small scope of this implementation, it is probably safe to do so.
        '''
        pa,l,u = self.plu()
        if self.determinant(u) ==0.:
            raiseAssertionError("Matrix is Singular")
        # Ly=B
        y = self.forward(l,b)
        # Return Ux=y
        return self.backwards(u,y)
    def determinant (self,U=None):
        if U is None:
            _,_,U = self.plu()
        diagonals = []
        diags = self._create_diag()
        for i in range(U.shape[0]):
            diagonals.append(U[i,diags[i]])

        return np.prod(diagonals) * self.permutation_flag
    def plu(self):
        ''' 
            LU Decomposition of A with pivoting, following Doolittle Algorithm
            
            Operates on a dense representation and outputs dense LU matricies

        '''

        diags = self._create_diag()
        L = np.zeros((self.area,self.n))
        for i in range(self.area):
            L[i,diags[i]] = 1.
        U = np.zeros((self.area,self.n))
        PA = self.get_permutation_dense()

        for j in range(self.area):

            basis = self.basis[j%self.d]
            Uupdatedable = [b for b in basis if b<=j]
            Lupdatedable = [b for b in basis if b>j]
            
           
            # Iteration through only the basis vectors for the given column.  A reduction of d
            # Utilizing the basis vectors we also can perform per column updates, rather than iterating
            # through each value in the column, saving more computations.
            for ii,i in enumerate(basis):
                if i <= j:
                    if len(Uupdatedable) == 1:
                        value = PA[diags[ii],j]
                    else:
                        value=PA[ii,j] -(L[i,range(diags[j])] * U[basis,diags[j]][:diags[j]]).sum()
                    U[i,diags[j]] = value
                    
                if i > j:
                    if len(Lupdatedable)>self.d:
                        value = PA[ii,j] / U[j,diags[j]]
                    else:
                        value = (PA[ii,j] -(L[i,range(diags[j])] * U[basis,diags[j]][:diags[j]]).sum()) / U[j,diags[j]]
                    L[i,diags[j]] = value


        return(PA,L,U)
    