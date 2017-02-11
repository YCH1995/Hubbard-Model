# Hubbard-Model
import numpy as np
from math import *
from numpy.linalg import eig
from scipy.sparse.linalg import eigsh
from scipy.sparse import *
from itertools import combinations
from numba import jit
 
#The function below calculate (2n)!/(n!)(n!).
def combination(n):                  
    temp=int(factorial(2*n)/int(factorial(n))**2)
    return temp                            
    
def make_base(n,state): 
#This function return the base ket of n,n+1,n-1 electrons on 2n sites.
#Coressponding to state=0,1,-1.The essence of the function is "combinations"
#Combinations(a,b)means randomly choose b items from list a.
    base=np.arange(int(2*n))               
    index=0
    if(state==0):
        elec_num=n
        length=combination(n)
    elif(state==1):
        elec_num=n+1
        length=int(combination(n)*n/(n+1))
    elif(state==-1):
        elec_num=n-1
        length=int(combination(n)*n/(n+1))
    else:
        print("Error")
    trans=[0]*length
    for i in combinations(base,elec_num):
        result=0
        for elem in i:
            result += 1<<elem            
        trans[index]=result/1
        index=index+1   
    trans.sort()
    return trans
    
def locat(i,array):  
#Using dichotomy to find the location of i in list array.If i is not in array,
#return Error.
    high = len(array) - 1
    low = 0
    if i > array[high] or i < array[0]:
        raise InputError('The item does not in the collection!')
    elif i == array[0]:
        return 0
    elif i == array[high]:
        return high
    else:
        while (high - low) > 1:
            mid = int((low + high) / 2)
            if i == array[mid]:
                return mid
            elif i < array[mid]:
                high = mid
            elif i > array[mid]:
                low =mid
        raise InputError('The item does not in the collection!')
        
def num(n,i,j):
#Return the number of "1"s between the ith and jth items in list n.
    count=0
    if(i>=j):
        raise InputError('i is bigger than j')
    if((j-i)==1):
        return 0
    for pos in range(i+1,j):
        if(np.bitwise_and(n,1<<pos)==1<<pos):
            count=count+1            
    return count

def hopping_matrix(i,j,array,n):
#i,j Means two different sites.This function returns the hopping term.
    a=[]
    for k in range(n):
        a.append(int(array[k]))
    buff0=np.bitwise_and(a,1<<i)==0
    buff1=np.bitwise_and(a,1<<j)==1<<j
    buff=np.logical_and(buff0,buff1)  
    column=np.where(buff==True)[0]
    len1=len(column)
    init=[0]*len1
    final=[0]*len1
    diff=(1<<i)-(1<<j)
    for count in range(len1):
        init[count]=a[column[count]]
        final[count]=init[count]+diff
    count=0
    row=[0]*len1
    eff=[0]*len1
    for sta in final:
        row[count]=locat(sta,a)
        eff[count]=(-1)**(num(sta,i,j))
        count=count+1
    mtx = csr_matrix((eff, (row, column)), shape=(n,n))
    mtx=mtx+csr_matrix((eff, (column, row)), shape=(n,n))
    return mtx
 
def interpart(array,dim,U,n):
#This function returns the interaction term.
    a=[]
    for k in range(dim):
        a.append(int(array[k]))
    mtx=0
    for j in range(n):
        i=2*j
        k=2*j+1
        buff0=np.bitwise_and(a,1<<i)==1<<i
        buff1=np.bitwise_and(a,1<<k)==1<<k
        buff=np.logical_and(buff0,buff1)
        column=np.where(buff==True)[0]
        len1=len(column)
        eff=[U]*len1
        mtx=mtx+csr_matrix((eff,(column,column)),shape=(dim,dim))
    return mtx 
    
def make_matrix(a,U,dim,num):
#Return the matrix representation of Hubbard Hamilton.
#Notice,state =0,1,-1 apply to the same algorithm.You only need to choose the dim.
    mtx=interpart(a,dim,U,num)
    for i in range(2*num-2):
        for j in range(i+1,2*num):
            if(i%2==j%2):
                mtx=mtx+hopping_matrix(i,j,a,dim)
    return mtx

def normal(array):                           #归一化
    mold=sqrt(np.vdot(array,array))
    return array/mold    
    
def lanczos(mtx,iter_time,dim):              #Lanczos法求基态能
    a=[]
    b=[]
    temp=np.ones(dim)
    sta=normal(temp)
    c0=np.vdot(sta,mtx.dot(sta))
    sta_old=sta
    sta=mtx.dot(sta_old)-c0*sta_old
    sta=normal(sta)
    a.append(c0)
    
    for i in range(1,iter_time):
        c0=np.vdot(sta,mtx.dot(sta))
        c1=np.vdot(sta_old,mtx.dot(sta))
        sta_new=mtx.dot(sta)-c0*sta-c1*sta_old
        a.append(c0)
        b.append(c1)
        if(np.vdot(sta_new,sta_new)<1E-8):
            break
        sta_new=normal(sta_new)
        sta_old=sta
        sta=sta_new
        
    row=np.arange(i+1)
    column=row
    H_tri=csr_matrix((a, (row, column)), shape=(i+1,i+1))
    row=np.arange(i)
    column=row+1
    H_tri=H_tri+csr_matrix((b, (row, column)), shape=(i+1,i+1))
    H_tri=H_tri+csr_matrix((b, (column, row)), shape=(i+1,i+1)) 
    a,b=eigsh(H_tri,2,which='SA',tol=1E-4)
    return a[0]

def H_tra(a1,a2,n,state,l):
#a1 is the origin one,a2 is added or subtracted one electron from a1.
#This function is used to transfer two bases.
#Which can also consider as the matrix representation of annhilation OP or
#produce OP.Necessary when calculating the green function with Lanczos.
    if(state==1):
        buff=np.bitwise_and(a1,1<<l)==0
        diff=1<<l
    elif(state==-1):
        buff=np.bitwise_and(a1,1<<l)==1<<l
        diff=(-1)*1<<l
    column=np.where(buff==True)[0]
    lenl=len(column)
    init=[0]*lenl
    final=[0]*lenl
    for count in range(lenl):
        init[count]=a1[column[count]]
        final[count]=init[count]+diff
    count=0
    row=[0]*lenl
    eff=[0]*lenl
    for sta in final:
        row[count]=locat(sta,a2)
        eff[count]=(-1)**(num(sta,l,2*n))
        count=count+1
    dim=combination(n)
    dim2=int(dim*n/(n+1))
    mtx = csr_matrix((eff, (row, column)), shape=(dim2,dim))
    return mtx        
#First step,count the ket of the base state and the corresponding Hamilton.
n=2
#num means the number of electrons in the model.
U=0
array=make_base(n,0)
array2=make_base(n,1)
array3=make_base(n,-1)
#corresponding ket
dim=combination(n)
dim2=int(dim*n/(n+1))
H=make_matrix(array,U,dim,n)
#corresponding Hamilton matrix
H_inc=make_matrix(array2,U,dim2,n)
H_dec=make_matrix(array3,U,dim2,n)
#These two matrixes are to be used when calculating the Green function.H_inc means
#that an electron was added to this model.
level=Test_Lanczos(H,dim)
ground_energy=Lanczos(H,level,dim)
ground_ket=Lanczos_make_ket(H,level,dim)
#Use Lanczos algorithm to calculate the ground energy and the ground state ket.
#This method is also very uesful in calculating Green function.
#Next,the Green function
pointnum=100
x=np.linspace(mu,mu+U,pointnum)    
y=np.zeros(pointnum,dtype=complex) 
k=3.141592653589793/2              
for i in range(pointnum):
    z=x[i]+0.03j
    ket_store=Lanczos_In_Process(H,H_inc,H_dec,ground_ket,num)
    G=GreenMtx(H,ket_store,num)
    V=np.zeros((num,num),dtype=complex)
    t = num * k
    V[0][num-1]=complex(cos(t),(-1)*sin(t))
    V[num-1][0]=complex(cos(t),sin(t))
    GreenRPA=G+G.dot(VMtx.dot(G)) #CPT
    GreenCPT=0
    for a in range(number):
        for b in range(number): 
            t=(b-a)*k
            GreenCPT=GreenCPT+complex(cos(t),sin(t))*GreenRPA[a][b]
    GreenCPT=GreenCPT/number
    y[i]=GreenCPT.imag*(-2)

plt.figure()
plt.contourf(k,x,y,200)
plt.colorbar()
plt.show()
