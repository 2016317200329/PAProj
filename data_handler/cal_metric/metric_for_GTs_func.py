import numpy as np

from GT_model.GT_2.SA_for_PT_funcs_delta_eq1 import *
from GT_model.GT_3.SA_for_PT_funcs_delta_eq1 import get_U_GT3
# For GT-2
alpha = -0.013581112
delta = 1
labda = 3.312402533

def GT_1(setting, target,q=1):
    b = np.array(setting.bidfee)               # bid fee (dollar)
    v = np.array(setting.retail)               # valuation
    d = np.array(setting.bidincrement)         # bid inc

    LEN ,T =  get_LEN_T(v ,b ,d ,max(target))

    U = get_U_GT1(LEN ,v, d, b ,eps = 0)

    nll_metric = get_nll_meric(target, U, LEN ,TARGET = 1,q=q)

    return nll_metric

def GT_2(setting, target,q=1, KL = False):
    b = np.array(setting.bidfee)  # bid fee (dollar)
    v = np.array(setting.retail)  # valuation
    d = np.array(setting.bidincrement)

    LEN, T = get_LEN_T(v, b, d, max(target))

    # Solve for U
    U = get_U_GT2(LEN, v, d, b, alpha, labda, eps=0.)

    nll_metric = get_nll_meric(target, U, LEN, TARGET=1, q=1)

    return nll_metric

def GT1_KL(setting, target_df_uniq):
    b = np.array(setting.bidfee)               # bid fee (dollar)
    v = np.array(setting.retail)               # valuation
    d = np.array(setting.bidincrement)         # bid inc

    target = list(target_df_uniq.iloc[:, 0])
    target_p = list(target_df_uniq.iloc[:, 1])

    LEN ,T = get_LEN_T(v ,b ,d ,max(target))

    U = get_U_GT1(LEN ,v, d, b ,eps = 0)

    KL_metric = get_KL_meric(target, target_p, U, LEN ,TARGET = 1)

    return KL_metric

def GT2_KL(setting, target_df_uniq, alpha_i, labda_i):
    b = np.array(setting.bidfee)               # bid fee (dollar)
    v = np.array(setting.retail)               # valuation
    d = np.array(setting.bidincrement)         # bid inc

    target = list(target_df_uniq.iloc[:, 0])
    target_p = list(target_df_uniq.iloc[:, 1])

    LEN ,T =  get_LEN_T(v ,b ,d ,max(target))

    U = get_U_GT2(LEN, v, d, b, alpha_i, labda_i, eps=0.)

    KL_metric = get_KL_meric(target, target_p, U, LEN ,TARGET = 1)

    return KL_metric

def GT3_KL(setting, target_df_uniq, alpha_i):
    b = np.array(setting.bidfee)               # bid fee (dollar)
    v = np.array(setting.retail)               # valuation
    d = np.array(setting.bidincrement)         # bid inc

    target = list(target_df_uniq.iloc[:, 0])
    target_p = list(target_df_uniq.iloc[:, 1])

    LEN ,T =  get_LEN_T(v ,b ,d ,max(target))

    U = get_U_GT3(LEN, v, d, b, alpha_i, eps=0.)

    KL_metric = get_KL_meric(target, target_p, U, LEN ,TARGET = 1)

    return KL_metric


# 为不同的auction使用uniq params
def GT_2_uniq(setting, target, alpha_i, labda_i,q=1):
        """

        Args:
            setting:
            target:
            alpha_i:
            labda_i:
            q:

        Returns: nll

        """
        b = np.array(setting.bidfee)               # bid fee (dollar)
        v = np.array(setting.retail)               # valuation
        d = np.array(setting.bidincrement)

        LEN,T =  get_LEN_T(v,b,d,max(target))

        # Solve for U
        U = get_U_GT2(LEN,v,d,b,alpha_i,labda_i,eps=0.)

        nll_metric = get_nll_meric(target, U, LEN,TARGET = 1,q=q)

        return nll_metric


def GT_3_uniq(setting, target, alpha_i, q=1):
    """

    Args:
        setting:
        target:
        alpha_i:
        q:

    Returns: nll

    """
    b = np.array(setting.bidfee)  # bid fee (dollar)
    v = np.array(setting.retail)  # valuation
    d = np.array(setting.bidincrement)

    LEN, T = get_LEN_T(v, b, d, max(target))

    # Solve for U
    U = get_U_GT3(LEN, v, d, b, alpha_i, eps=0.)

    nll_metric = get_nll_meric(target, U, LEN, TARGET=1, q=q)

    return nll_metric