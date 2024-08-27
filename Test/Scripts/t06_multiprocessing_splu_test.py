import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool


def func_solve(params):

    mat = params[0]
    x = np.array([1., 2., 3.], dtype=float)

    mat_lu = splu(mat)
    z = mat_lu.solve(x)

    x1 = mat.dot(z)

    print("Norm of difference = ", np.linalg.norm(x - x1))


if __name__ == "__main__":

    mat_ = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
    mat_lu_ = splu(mat_)

    param_tuple_list = [(mat_,)]

    with Pool(min(len(param_tuple_list), mp.cpu_count())) as pool:
        max_ = len(param_tuple_list)

        with tqdm(total=max_) as pbar:
            for _ in pool.imap_unordered(func_solve, param_tuple_list):
                pbar.update()
