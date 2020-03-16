import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['axes.formatter.useoffset']=False
import matplotlib.pyplot as plt
import scipy as sc
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh,expm

import KagomeFunctions as kgf

def print_update(msg):
    sys.stdout.write("\r"+msg)
    sys.stdout.flush()

class MatrixProvider:

    def __init__(self,L, use_dense=False):
        self.L=L
        self.nspins=9*L**2
        self.__zmat=None
        self.__xmat=None
        self.__smmat=None
        self.__spmat=None
        self.__id=None
        self.use_dense=use_dense
        if use_dense:
            self.kron=np.kron
            self.eye=np.eye
            self.zero_matrix=np.zeros
        else:
            self.kron=sp.kron
            self.eye=sp.eye
            self.zero_matrix=sp.csr_matrix

    @property
    def xmat(self):
        if self.__xmat is None:
            self.__zmat,self.__xmat, self.__smmat, self.__spmat,self.__id=self.generate_matrices()
        return self.__xmat

    @property
    def zmat(self):
        if self.__zmat is None:
            self.__zmat,self.__xmat, self.__smmat, self.__spmat,self.__id=self.generate_matrices()
        return self.__zmat

    @property
    def smmat(self):
        if self.__smmat is None:
            self.__zmat,self.__xmat, self.__smmat, self.__spmat,self.__id=self.generate_matrices()
        return self.__smmat

    @property
    def spmat(self):
        if self.__spmat is None:
            self.__zmat,self.__xmat, self.__smmat, self.__spmat,self.__id=self.generate_matrices()
        return self.__spmat

    @property
    def id(self):
        if self.__id is None:
            self.__zmat, self.__smmat, self.__spmat,self.__id=self.generate_matrices()
        return self.__id

    def get_ndof(self):
        #We know that we are using Ising spins
        return 2

    def get_nspin(self):
        #We know that we are using Ising spins
        return 9*self.L**2 

    def generate_matrices(self):
        #We know that we are dealing with Ising spins
        ndof=self.get_ndof()
        smmat=np.diag([1.]*(ndof-1),k=1)
        id=np.eye(ndof)
        zmat=np.array([[1,0],[0,-1]])
        xmat=np.array([[0,1],[1,0]])
        if self.use_dense:
            spmat=np.transpose(smmat)
        else:
            zmat=sp.csr_matrix(zmat)
            xmat=sp.csr_matrix(xmat)
            smmat=sp.csr_matrix(smmat)
            spmat=smmat.transpose(copy=True)
            id=sp.csr_matrix(id)

        return zmat,xmat,smmat,spmat,id

    def create_ising_term(self,J1,J2,J3,J4):
        ndof=self.get_ndof()
        nspin=self.get_nspin()
        dest=self.zero_matrix((ndof**nspin,ndof**nspin))
        s_ijl,ijl_s=kgf.createspinsitetable(self.L)
        if J1!=0:
            nn1vec=kgf.NNpairs(ijl_s, s_ijl, self.L)
            for i,j in nn1vec:
                dest+=J1*self.gen_sys(self.zmat,i)@self.gen_sys(self.zmat,j)
        if J2!=0:
            nn2vec=kgf.NN2pairs(ijl_s, s_ijl, self.L)
            for i,j in nn2vec:
                dest+=J2*self.gen_sys(self.zmat,i)@self.gen_sys(self.zmat,j)
        if J3!=0:
            nn3vec=kgf.NN3pairs(ijl_s, s_ijl, self.L)
            for i,j in nn3vec:
                dest+=J3*self.gen_sys(self.zmat,i)@self.gen_sys(self.zmat,j)
        if J4!=0:
            nn3vec=kgf.NN4pairs(ijl_s, s_ijl, self.L)
            for i,j in nn4vec:
                dest+=J4*self.gen_sys(self.zmat,i)@self.gen_sys(self.zmat,j)
        return dest

    def create_transverse_field_term(self,h):
        nspin=self.get_nspin()
        ndof=self.get_ndof()
        dest=self.zero_matrix((ndof**nspin,ndof**nspin))
        for i in range(nspin):
            dest+=self.gen_sys(self.xmat,i)
        return dest

    def gen_sys(self,mat,i):
        ndof=self.get_ndof()
        nspin=self.get_nspin()
        return self.kron(self.kron(self.eye(ndof**i),mat),self.eye(ndof**(nspin-i-1)))

    def create_hamilton(self, J1, J2, J3, J4, h, verbose=False, check_herm=False):
        ndof=self.get_ndof()
        nspin=self.get_nspin()
        dest=self.zero_matrix((ndof**nspin,ndof**nspin))
        dest += self.create_ising_term(J1,J2,J3,J4)
        if h!=0.:
            dest -= h*self.create_transverse_field_term(h)
        if check_herm:
            if is_hermitian(dest):
                print("Hamiltonian is hermitian")
            else:
                print("Hamiltonian is not hermitian",file=sys.stderr)
        return dest

def is_hermitian(mat):
    if sp.issparse(mat):
        return (mat.getH()-mat).nnz==0
    else:
        return np.all(mat.T.conj()==mat)

def is_diagonal(mat):
    if sp.issparse(mat):
        mat=mat.toarray()
    return np.count_nonzero(mat - np.diag(np.diagonal(mat))) == 0

def print_info_groundstate(args):
    print("===================== INFO ====================")
    print("J1: {:.2f}".format(args.J1))
    print("J2: {:.2f}".format(args.J2))
    print("J3: {:.2f}".format(args.J3))
    print("J4: {:.2f}".format(args.J4))
    print("h : {:.2f}".format(args.h))
    print("===============================================")

def print_info_thermal(args):
    print("===================== INFO ====================")
    print("J1: {:.2f}".format(args.J1))
    print("J2: {:.2f}".format(args.J2))
    print("J3: {:.2f}".format(args.J3))
    print("J4: {:.2f}".format(args.J4))
    print("h : {:.2f}".format(args.h))
    print("Tmin : {:.2f}".format(args.Tmin))
    print("Tmax : {:.2f}".format(args.Tmax))
    print("===============================================")

def groundstate_scan(args):
    print_info_groundstate(args)
    evals=[]
    evecs=[]
    mp=MatrixProvider(args.L,use_dense=args.dense)
    ham=mp.create_hamilton(args.J1,args.J2,args.J3,args.J4,args.h)
    eval,evec=eigsh(ham,k=1,which='SA',return_eigenvectors=True)
    evals.append(eval)
    evecs.append(evec)
    evals=np.squeeze(np.asarray(evals))
    evecs=np.squeeze(np.asarray(evecs))
    print("")
    print("GS: {}".format(eval[0]))

def thermal_expectation(args):
    print_info_thermal(args)
    mp=MatrixProvider(args.L,use_dense=args.dense)
    ham=mp.create_hamilton(args.J1,args.J2,args.J3,args.J4,args.h)
    if args.Tmin==args.Tmax:
        temp_vec=[args.Tmin]
    elif args.Tmin<args.Tmax:
        if args.steps is not None:
            if not args.logsteps:
                temp_vec=np.linspace(args.Tmin,args.Tmax,args.steps)
            else:
                temp_vec=np.geomspace(args.Tmin,args.Tmax,args.steps)
        elif args.inc is not None:
            temp_vec=np.arange(args.Tmin,args.Tmax+args.inc,args.inc)
        else:
            print("No method for interpolation given: Specify --inc or --steps.",file=sys.stderr)
            sys.exit(1)
    else:
        print("Tmin has to be smaller or equal to Tmax.",file=sys.stderr)
        sys.exit(1) 
    if not args.dense:
        ham=ham.tocsc()
    energyvec=[]
    for temp in temp_vec:
        print_update("T: {:.02f}".format(temp))
        density_matrix=expm(-ham/temp)
        if args.dense:
            energy=sum(np.diagonal(ham@density_matrix))/sum(np.diagonal(density_matrix))
        else:
            energy=(ham@density_matrix).diagonal().sum()/(density_matrix).diagonal().sum()
        energyvec.append(energy)
    print("")
    print("   T     Energy")
    for t,e in zip(temp_vec,energyvec):
        print("{:.03f} {:.06f}".format(t,e))
    if args.plot:
        f,ax=plt.subplots()
        ax.set_xlabel("T")
        ax.set_ylabel("Energy")
        ax.plot(temp_vec,energyvec)
        plt.show()

if __name__=="__main__":
    import argparse
    parser= argparse.ArgumentParser(prog="Python ED for the TFIM on the kagome lattice", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--L",type=np.int,default=1,help="Extent of the system (length of lower left edge")
    parser.add_argument("--J1",type=np.float,default=1.,help="J1 coupling (Nearest Neighbours)")
    parser.add_argument("--J2",type=np.float,default=0.,help="J2 coupling (2nd NN)")
    parser.add_argument("--J3",type=np.float,default=0.,help="J3 coupling (3rd NN)")
    parser.add_argument("--J4",type=np.float,default=0.,help="J4 coupling (4th NN)")
    parser.add_argument("--h",type=np.float,default=0.,help="transverse field in x direction")
    parser.add_argument("--dense",default=False, action="store_true", help="Activate calculation dense matrices")
    parser.add_argument("--plot",default=False, action="store_true", help="Activate plotting")

    subparsers=parser.add_subparsers(dest='mode')
    subparsers.required=True
    parser_gs=subparsers.add_parser('groundstate', help="Calculate the ground state", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_gs.set_defaults(func=groundstate_scan)

    parser_th=subparsers.add_parser('thermal', help="Calculate the ground state", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_th.add_argument("--Tmin", type=float, help="Minimal temperature")
    parser_th.add_argument("--Tmax", type=float, help="Maximal temperature")
    parser_th.set_defaults(func=thermal_expectation)
    parser_th.add_argument("--steps", type=int, help="Number of steps")
    parser_th.add_argument("--inc", type=float, help="Increment with each step")
    parser_th.add_argument("--logsteps", default=False, action="store_true", help="Number of logarithmic steps")
    
    args=parser.parse_args()

    args.func(args)
