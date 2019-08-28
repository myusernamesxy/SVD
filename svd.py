import cv2
import numpy as np

img = cv2.imread("ruler.512.tiff", 2)
img = np.mat(img)/1.0
AAT = img.dot((img.transpose()))
values1, vectors1 = np.linalg.eig(AAT)
values2 = values1
vectors2 = (img.transpose()).dot(vectors1)

vectors1 = np.array(vectors1)
newvalues1 = np.sort(values1)[::-1]
vectors2 = np.array(vectors2)
newvalues2 = np.sort(values2)[::-1]


# calculate low rank approximation matrix
def approximation(rank):
    # newvectors1 represents the matrix U
    newvectors1 = np.zeros([len(vectors1), rank], dtype=complex)
    # newvectors2 represents transpose of the matrix V
    newvectors2 = np.zeros([rank, len(vectors2)], dtype=complex)
    for i in range(rank):
        sum = np.linalg.norm(vectors2[:, (np.argwhere(values1 == (newvalues2[i])))[0][0]])
        if sum == 0:
            newvectors2[i, :] = vectors2[:, (np.argwhere(values2 == (newvalues2[i])))[0][0]]
        else:
            # standardized vectors and store in newvectors2
            newvectors2[i, :] = vectors2[:, (np.argwhere(values2 == (newvalues2[i])))[0][0]]/sum
        # store in newvectors1
        newvectors1[:, i] = vectors1[:, (np.argwhere(values1 == (newvalues1[i])))[0][0]]
    diagsort = np.diag(np.sqrt(newvalues1))
    # diag represents the diagonal matrix of singular values
    diag = np.zeros([rank, rank], dtype=complex)
    for c in range(rank):
        diag[c][c] = diagsort[c][c]
    # calculate low rank approximation matrix
    resmatrix = newvectors1.dot(diag).dot(newvectors2)
    return resmatrix


cv2.imwrite("rank5.tiff", np.real(approximation(5)))
img1 = cv2.imread("rank5.tiff", 2)
cv2.imwrite("rank10.tiff", np.real(approximation(10)))
img2 = cv2.imread("rank10.tiff", 2)
cv2.imwrite("rank20.tiff", np.real(approximation(20)))
img3 = cv2.imread("rank20.tiff", 2)
cv2.imwrite("rank50.tiff", np.real(approximation(50)))
img4 = cv2.imread("rank50.tiff", 2)
cv2.imwrite("rank100.tiff", np.real(approximation(100)))
img5 = cv2.imread("rank100.tiff", 2)
cv2.imwrite("rank200.tiff", np.real(approximation(200)))
img6 = cv2.imread("rank200.tiff", 2)

cv2.imshow('Rank5', img1)
cv2.imshow('Rank10', img2)
cv2.imshow('Rank20', img3)
cv2.imshow('Rank50', img4)
cv2.imshow('Rank100', img5)
cv2.imshow('Rank200', img6)
cv2.waitKey(0)
cv2.destroyAllWindows()
