class Diagonal:
    def __init__(self,n,d,precision='single',sparse=True):
        self.n = n
        self.d = d
        self.area = n*d
        self.dims = 2
        self.shape = [self.n*self.d,self.n*self.d]
        self.precision = np.float32 if 'single' else np.float64
        self.sparse= sparse
        self.table = self._create_table()
        self.matrix = self._create_matrix()
        self.basis = self._create_basis()
        self.row_based = self._create_row_representation()
        
    
    def _create_basis(self):
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
    
    def _sparse_mm(self,y,left=False):
        if left:
            batched = np.vstack([y[:,b] for b in self.basis])
            batched = np.reshape(batched,(self.d,y.shape[-1],self.n))
        else:
            batched = np.vstack([y[b,:] for b in self.basis])
            batched = np.reshape(batched,(self.d,self.n,y.shape[-1]))
        #batched = np.vstack([batched for _ in range(self.n)])
        if left:
            return np.vstack([np.dot(batched[ii%self.d],i) for ii,i in enumerate(self.row_based.T)]).T
        else:
            return np.vstack([np.dot(i,batched[ii%self.d]) for ii,i in enumerate(diag.matrix)])
        
    @staticmethod
    def _swap_elements(l,i,j):
        new = l.copy()
        new[i],new[j] = j,i
        return new
    
    
    def get_permutation_dense(self):
        diags = np.repeat(np.arange(0,self.n),self.d)
        updated = self.row_based.copy()
        for c in range(updated.shape[1]):
            max_idx = np.argmax(np.abs(self.row_based[diags[c]:,c]))
            max_idx += diags[c]
            if (max_idx != diags[c]) and (max_idx > diags[c] ):

                order = self._swap_elements(list(range(updated.shape[0])),max_idx,diags[c])
                basis = self.basis[c%self.d]
                updated[:,basis] = updated[:,basis][[order],:]
        return updated
    
    @staticmethod
    def create_pivot(mat):
        mat_size = mat.shape[0]
        p_blank = np.eye(mat_size)
        for c in range(mat_size):
            row = max(range(c, mat_size), key=lambda i: abs(mat[i,c]))
            if c != row:
                p_blank[[c,row]] = p_blank[[row,c]]
        return p_blank
        
    
    def mm(self,y,x=None,sparse=True,left=False):
        if sparse:
            return self._sparse_mm(y,left=left)
        else:
            assert x is not None
            return np.dot(x,y)
        
    def plu(self):
        L = np.eye(self.area)
        U = np.zeros((self.area,self.area))
        P = self.create_pivot(self.from_dense(self.matrix))
        PA = self.get_permutation_dense()

        for j in range(self.area):

            basis = self.basis[j%self.d]
            updatedable = [b for b in basis if b<=j]

            for ii,i in enumerate(basis):
                if i <= j:
                    if len(updatedable) == 1:
                        value = PA[ii,j]
                    else:
                        value= PA[ii,j] - (U[updatedable[:ii+(1-j%2)],j] *L[basis[ii],updatedable[:ii+(1-j%2)]]).sum()
                    U[i,j] = value

            updatedable = [b for b in basis if b>j]
            for ii,i in enumerate(basis):
                if i > j:
                    if len(updatedable)>self.d:
                        value = PA[ii,j] / U[j,j]
                    else:
                        value = (PA[ii,j] -(L[basis[ii],range(j)] * U[range(j),j]).sum()) / U[j,j]
                    L[i,j] = value


        return(P,PA,L,U)
