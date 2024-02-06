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


def int01(a,b,c,d,x,y,start,end):
    # Xneg2, Yneg2
    ak = torch.tensor([2,4,5/2,1/2])
    bk = torch.tensor([2,4,5/2,1/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k+1):
            for kp in range(M):
                for ip in range(kp+1):
                    result = result + ak[k] * alpha[i,k] * bk[kp]* beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result


def int02(a,b,c,d,x,y,start,end):
    # Xneg2, Yneg1
    ak = torch.tensor([2,4,5/2,1/2])
    bk = torch.tensor([1,0,-5/2,-3/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int03(a,b,c,d,x,y,start,end):
    # Xneg2, Ypos1
    ak = torch.tensor([2,4,5/2,1/2])
    bk = torch.tensor([1,0,-5/2,3/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int04(a,b,c,d,x,y,start,end):
    # Xneg2, Ypos2
    ak = torch.tensor([2,4,5/2,1/2])
    bk = torch.tensor([2,-4,5/2,-1/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int05(a,b,c,d,x,y,start,end):
    # Xneg1, Yneg2
    ak = torch.tensor([1,0,-5/2,-3/2])
    bk = torch.tensor([2,4,5/2,1/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int06(a,b,c,d,x,y,start,end):
    # Xneg1, Yneg1
    ak = torch.tensor([1,0,-5/2,-3/2])
    bk = torch.tensor([1,0,-5/2,-3/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int07(a,b,c,d,x,y,start,end):
    # Xneg1, Ypos1
    ak = torch.tensor([1,0,-5/2,-3/2])
    bk = torch.tensor([1,0,-5/2,3/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int08(a,b,c,d,x,y,start,end):
    # Xneg1, Ypos2
    ak = torch.tensor([1,0,-5/2,-3/2])
    bk = torch.tensor([2,-4,5/2,-1/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int09(a,b,c,d,x,y,start,end):
    # Xpos1, Yneg2
    ak = torch.tensor([1,0,-5/2,3/2])
    bk = torch.tensor([2,4,5/2,1/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int10(a,b,c,d,x,y,start,end):
    # Xpos1, Yneg1
    ak = torch.tensor([1,0,-5/2,3/2])
    bk = torch.tensor([1,0,-5/2,-3/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int11(a,b,c,d,x,y,start,end):
    # Xpos1, Ypos1
    ak = torch.tensor([1,0,-5/2,3/2])
    bk = torch.tensor([1,0,-5/2,3/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int12(a,b,c,d,x,y,start,end):
    # Xpos1, Ypos2
    ak = torch.tensor([1,0,-5/2,3/2])
    bk = torch.tensor([2,-4,5/2,-1/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int13(a,b,c,d,x,y,start,end):
    # Xpos2, Yneg2
    ak = torch.tensor([1,0,-5/2,3/2])
    bk = torch.tensor([2,4,5/2,1/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int14(a,b,c,d,x,y,start,end):
    # Xpos2, Yneg1
    ak = torch.tensor([1,0,-5/2,3/2])
    bk = torch.tensor([1,0,-5/2,-3/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int15(a,b,c,d,x,y,start,end):
    # Xpos2, Ypos1
    ak = torch.tensor([1,0,-5/2,3/2])
    bk = torch.tensor([1,0,-5/2,-3/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result

def int16(a,b,c,d,x,y,start,end):
    # Xpos2, Ypos2
    ak = torch.tensor([1,0,-5/2,3/2])
    bk = torch.tensor([2,-4,5/2,-1/2])
    M = 4
    alpha, beta = compute_alpha_beta(a,b,c,d,x,y)
    result = 0
    for k in range(M):
        for i in range(k + 1):
            for kp in range(M):
                for ip in range(kp + 1):
                    result = result + ak[k] * alpha[i,k] * bk[kp] * beta[ip,kp] * (end**(i+ip+1) - start**(i+ip+1)) / (i+ip+1)

    return result
