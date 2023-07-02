from GT_model.GT_2.SA_for_PT_funcs_delta_eq1 import *
# For GT-2
alpha = -0.013581112
delta = 1
labda = 3.312402533

def GT_1(setting, target):
    b = np.array(setting.bidfee)               # bid fee (dollar)
    v = np.array(setting.retail)               # valuation
    d = np.array(setting.bidincrement)         # bid inc

    LEN ,T =  get_LEN_T(v ,b ,d ,max(target))

    U = get_U_GT1(LEN ,v, d, b ,eps = 0)

    nll_metric = get_nll_meric(target, U, LEN ,TARGET = 1)

    return nll_metric

def GT_2(setting, target):
    b = np.array(setting.bidfee)  # bid fee (dollar)
    v = np.array(setting.retail)  # valuation
    d = np.array(setting.bidincrement)

    LEN, T = get_LEN_T(v, b, d, max(target))

    # Solve for U
    U = get_U_GT2(LEN, v, d, b, alpha, labda, eps=0.)

    nll_metric = get_nll_meric(target, U, LEN, TARGET=1)

    return nll_metric


# 为不同的auction使用uniq params
def GT_2_uniq(setting, target, alpha_i, labda_i):
        b = np.array(setting.bidfee)               # bid fee (dollar)
        v = np.array(setting.retail)               # valuation
        d = np.array(setting.bidincrement)

        LEN,T =  get_LEN_T(v,b,d,max(target))

        # Solve for U
        U = get_U_GT2(LEN,v,d,b,alpha_i,labda_i,eps=0.)

        nll_metric = get_nll_meric(target, U, LEN,TARGET = 1)

        return nll_metric