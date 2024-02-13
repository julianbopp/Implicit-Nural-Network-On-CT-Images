import torch
from torch import lgamma

def nck(n,k,device):
    n = torch.tensor([n],device=device)
    k = torch.tensor([k],device=device)
    return ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()).exp()

def compute_alpha_beta(a,b,c,d,x,y):
    device = x.device
    alpha = torch.zeros([4,4],device=device)
    beta = torch.zeros([4,4],device=device)

    for i in range(4):
        for k in range(4):
            alpha[i,k] = a**i * (b-x) ** (k-i) * nck(k,i,device)
            beta[i,k] = c**i * (d-y) ** (k-i) * nck(k,i,device)

    return alpha, beta

def get_coefficients(x_sign, x_dist, y_sign, y_dist):
    if x_dist == 1:
        ak = torch.tensor([1.0, 0.0, -5/2, 3/2])
    else:
        ak = torch.tensor([2.0, -4.0, 5/2, -1/2])


    if y_dist == 1:
        bk = torch.tensor([1.0, -5 / 2, 0.0, 3 / 2])
    else:
        bk = torch.tensor([2.0, -4.0, 5 / 2, -1 / 2])

    if x_sign == "neg":
        ak = ak * torch.tensor([1.0, -1.0, 1.0, -1.0])
    if y_sign == "neg":
        bk = bk * torch.tensor([1.0, -1.0, 1.0, -1.0])

    return ak, bk

def integrate_exact(x_sign, x_dist, y_sign, y_dist, a, b, c, d, x, y, h, start, end):
    alpha, beta = compute_alpha_beta(a, b, c, d, x, y)
    ak, bk = get_coefficients(x_sign, x_dist, y_sign, y_dist)

    result = 0
    for k in range(4):

        tmp1 = 0
        for kp in range(4):

            tmp2 = 0
            for i in range(k+1):

                tmp3 = 0
                for ip in range(kp+1):
                    tmp3 = tmp3 + beta[ip,kp] * (end ** (ip+i+1) - start ** (ip+i+1)) / (ip+i+1)

                tmp2 = tmp2 + alpha[i,k] * tmp3
            tmp1 = tmp1 + bk[kp] * 1/h**kp * tmp2
        result = result + ak[k] * 1/h**k * tmp1

    return result




x_sign = "neg"
y_sign = "neg"
x_dist = 1
y_dist = 1

ak,bk = get_coefficients(x_sign, x_dist, y_sign, y_dist)
print(ak)
print(bk)
