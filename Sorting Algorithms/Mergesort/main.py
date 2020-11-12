class MergeSort:
	def __init__(self):
		self.B = None

	def merge(self, A, l, m, r):
		i = l
		j = m
		k = l
		while i < m and j <= r:
			if self.B[i] < self.B[j]:
				A[k] = self.B[i]
				i+=1
				k+=1
			else:
				A[k] = self.B[j]
				j+=1
				k+=1
		if i < m:
			A[k:r+1] = self.B[i:m]
		elif j <= r:
			A[k:r+1] = self.B[j:r+1]
		print("++++++++++++++++++++++++ i:"+str(l)+' m:'+str(m)+' r:'+str(r))
		print(self.B[l:r+1])
		print(A[l:r+1])


	def _mergeSort(self, A, l, r):
		if l < r:
			mid = int((l+r-1)/2)
			self.B[l:r+1] = A[l:r+1]
			self._mergeSort(A, l, mid)
			self._mergeSort(A, mid+1, r)
			#print(A)
			self.merge(A, l, mid+1, r)

	def mergeSort(self, A):
		self.B = [0]*len(A)
		self._mergeSort(A, 0, len(A)-1)

################# Test ##################
import numpy as np
A = np.random.permutation(10)
print(A)
MS = MergeSort()
MS.mergeSort(A)
print(A)
print('Can\'t help it')
