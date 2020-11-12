def m3(A, a, b, c):
	if (A[b] < A[a] < A[c]) or (A[c] < A[a] < A[b]):
		return a
	elif (A[a] < A[b] < A[c]) or (A[c] < A[b] < A[a]):
		return b
	return c

t = 0
def swap(A, i, j):
	global t
	t = A[i]
	A[i] = A[j]
	A[j] = t

v = 0
def p(A, l ,r):
	swap(A, l, m3(A, l, int((l+r)/2), r))
	v = A[l]
	i = l+1
	j = r
	while i <= j:
		while A[i] <= v:
			i += 1
			if i > j:
				break
		while A[j] > v:
			j -= 1
			if i > j:
				break
		if i < j:
			swap(A, i, j)
			i += 1
			j -= 1
	swap(A, l, j)
	return j

def _qs(A, l ,r):
	if r > l:
		pivot = p(A, l ,r)
		_qs(A, l, pivot-1)
		_qs(A, pivot+1, r)

def quickSort(A):
	_qs(A, 0, len(A)-1)

################# Test ##################
import numpy as np
A = np.random.permutation(50)
print(A)
quickSort(A)
print(A)
