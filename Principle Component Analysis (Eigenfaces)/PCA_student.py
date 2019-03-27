import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

######### Load the data ##########

infile = open('faces.csv','r')
img_data = infile.read().strip().split('\n')
img = [map(int,a.strip().split(',')) for a in img_data]
pixels = []
for p in img:
    pixels += p
faces = np.reshape(pixels,(400,4096))

######### Global Variable ##########

image_count = 0

######### Function that normalizes a vector x (i.e. |x|=1 ) #########

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.
#   Note: in the given functionm, U should be a vector, not a array. 
#         You can write your own normalize function for normalizing 
#         the colomns of an array.

def normalize(U):
	return U / LA.norm(U) 

######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: You need a matplotlib.pyplot.show() at the end to display all the figures.

first_face = np.reshape(faces[0],(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('First_face')
plt.imshow(first_face,cmap=plt.cm.gray)


########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.
#   Note: There are two ways to order the elements in an array: 
#         column-major order and row-major order. In np.reshape(), 
#         you can switch the order by order='C' for row-major(default), 
#         or by order='F' for column-major. 


#### Your Code Here ####

b = np.array([i for i in range(400)])
rand = np.random.choice(b)
random_face = np.reshape(faces[rand],(64,64),order='F')
print(random_face)
image_count+=1
plt.figure(image_count)
plt.title('Random_face')
plt.imshow(random_face,cmap=plt.cm.gray)

########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over 
#   the flattened array by default, otherwise over the specified axis. 
#   float64 intermediate and return values are used for integer inputs.

#### Your Code Here ####

faces = np.matrix(faces)
mean = np.mean(faces, axis=0)


mean_face = np.reshape(mean,(64,64),order='F')
plt.title('First_face')
plt.imshow(mean_face,cmap=plt.cm.gray)


######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

#### Your Code Here ####

centered = faces-mean


######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data. 
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations. 
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 

#### Your Code Here ####

A = np.matrix(centered)
L = A*A.transpose()

evals, evecs = LA.eig(L)
idx = np.argsort(evals)[::-1]
evecs_sorted = evecs[:,idx]
evals_sorted = evals[idx]

e_v = []
for i in range(len(evecs_sorted)):
    e_v.append(A.T*evecs_sorted[:,i])

for i in range(len(e_v)):
    e_v[i] = normalize(e_v[i])

zj = np.matrix(np.array(e_v))
U = zj.T


########## Display the first 16 principal components ##################

#### Your Code Here ####

eig_faces = np.reshape(np.array(U),(64,64,400), order='F')
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(eig_faces[:,:,i], cmap=plt.cm.gray)


########## Reconstruct the first face using the first two PCs #########

#### Your Code Here ####


def display(pcs, face):
    omega = (U[:,:pcs]).T*(A[face,:]).T
    disp = ((U[:,:pcs])*omega) + mean.T
    return disp

two_pcs = np.reshape(display(2,0),(64,64),order='F')
plt.imshow(two_pcs,cmap=plt.cm.gray)


########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

#### Your Code Here ####

ling = np.reshape(display(5,100),(64,64),order='F')
plt.imshow(ling, cmap=plt.cm.gray)

ling = np.reshape(display(10,100),(64,64),order='F')
plt.imshow(ling, cmap=plt.cm.gray)

ling = np.reshape(display(25,100),(64,64),order='F')
plt.imshow(ling, cmap=plt.cm.gray)

ling = np.reshape(display(50,100),(64,64),order='F')
plt.imshow(ling, cmap=plt.cm.gray)

ling = np.reshape(display(100,100),(64,64),order='F')
plt.imshow(ling, cmap=plt.cm.gray)

ling = np.reshape(display(200,100),(64,64),order='F')
plt.imshow(ling, cmap=plt.cm.gray)

ling = np.reshape(display(300,100),(64,64),order='F')
plt.imshow(ling, cmap=plt.cm.gray)

ling = np.reshape(display(399,100),(64,64),order='F')
plt.imshow(ling, cmap=plt.cm.gray)


######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 
#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.

#### Your Code Here ####

list2  = []
evals_sum = 0
for val in evals_sorted:
    evals_sum += val
for val in evals_sorted:
    list2.append(val/evals_sum)
plt.plot(list2)


