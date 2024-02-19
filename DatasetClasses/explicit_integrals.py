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

    for k in range(4):
        for i in range(k+1):
            alpha[i,k] = (a**i) * ((b-x) ** (k-i)) * nck(k,i,device)
            beta[i,k] = (c**i) * ((d-y) ** (k-i)) * nck(k,i,device)

    return alpha, beta

def get_coefficients(x_sign, x_dist, y_sign, y_dist):
    if x_dist == 1:
        ak = torch.tensor([1.0, 0.0, -5/2, 3/2])
    else:
        ak = torch.tensor([2.0, -4.0, 5/2, -1/2])


    if y_dist == 1:
        bk = torch.tensor([1.0, 0.0, -5/2, 3/2])
    else:
        bk = torch.tensor([2.0, -4.0, 5/2, -1/2])

    if x_sign == "neg":
        ak = ak * torch.tensor([1.0, -1.0, 1.0, -1.0])
    if y_sign == "neg":
        bk = bk * torch.tensor([1.0, -1.0, 1.0, -1.0])

    return ak, bk

def get_spline_coefficients(sign, dist):
    if dist == 1:
        coefficients = torch.tensor([1.0, 0.0, -5 / 2, 3 / 2])
    elif dist == 2:
        coefficients = torch.tensor([2.0, -4.0, 5 / 2, -1 / 2])
    else:
        coefficients = torch.tensor([0.0, 0.0, 0.0, 0.0])

    if sign == "neg":
        coefficients = coefficients * torch.tensor([1.0, -1.0, 1.0, -1.0])

    return coefficients


def integrate_exact(x_sign, x_dist, y_sign, y_dist, a, b, c, d, x, y, h, start, end):
    ak, bk = get_coefficients(x_sign, x_dist, y_sign, y_dist)
    Ax, Bx, Cx, Dx = get_t_coeff(ak, a, b, x, h)
    Ay, By, Cy, Dy = get_t_coeff(bk, c, d, y, h)

    result = integrate(Ax,Bx,Cx,Dx,Ay,By,Cy,Dy,start,end)
    return result

def integrate_1d(sign, dist, slope, bias, cp, h, start, end):
    spline_coefficients = get_spline_coefficients(sign, dist)
    A, B, C, D = get_t_coeff(spline_coefficients, slope, bias, cp, h)

    result = A*(end**4-start**4)/4 + B*(end**3-start**3)/3 + C*(end**2-start**2)/2 + D*(end-start)

    return result


def get_t_coeff(spline_coeff, n, bias, cp, h):
    alpha = (bias-cp)
    a,b,c,d = spline_coeff[3],spline_coeff[2],spline_coeff[1],spline_coeff[0]

    A = a * n ** 3 / (h ** 3)
    B = (3 * a * n ** 2 * alpha + h* b * n ** 2)/(h**3)
    C = (a * 3 * n * alpha ** 2 + h * b * 2 * n * alpha + h**2*c * n)/(h**3)
    D = (a * alpha ** 3 + b * alpha ** 2 *h + c * alpha *h**2 + d*h**3)/(h**3)
    return A, B, C, D

def integrate(Ax,Bx,Cx,Dx,Ay,By,Cy,Dy,t0,t1):
    int_value = ((t1**7-t0**7)*Ax*Ay/7).view(1)
    int_value += (t1**6-t0**6)*(Ax*By + Bx*Ay)/6
    int_value += (t1**5-t0**5)*(Ax*Cy + Bx*By + Cx*Ay)/5
    int_value += (t1**4-t0**4)*(Ax*Dy + Bx*Cy + Cx*By + Dx*Ay)/4

    int_value += (t1**3-t0**3)*(Bx*Dy +Cx*Cy + Dx*By)/3
    int_value += (t1**2-t0**2)*(Cx*Dy + Dx*Cy)/2
    int_value += (t1-t0)*(Dx*Dy)


    int_value = t0*(-Dx*Dy + t0*(-Cx*Dy/2 - Cy*Dx/2 + t0*(-Bx*Dy/3 - By*Dx/3 - Cx*Cy/3 + t0*(-Ax*Dy/4 - Ay*Dx/4 - Bx*Cy/4 - By*Cx/4 + t0*(-Ax*Cy/5 - Ay*Cx/5 - Bx*By/5 + t0*(-Ax*Ay*t0/7 - Ax*By/6 - Ay*Bx/6)))))) + t1*(Dx*Dy + t1*(Cx*Dy/2 + Cy*Dx/2 + t1*(Bx*Dy/3 + By*Dx/3 + Cx*Cy/3 + t1*(Ax*Dy/4 + Ay*Dx/4 + Bx*Cy/4 + By*Cx/4 + t1*(Ax*Cy/5 + Ay*Cx/5 + Bx*By/5 + t1*(Ax*Ay*t1/7 + Ax*By/6 + Ay*Bx/6))))))



    return int_value






