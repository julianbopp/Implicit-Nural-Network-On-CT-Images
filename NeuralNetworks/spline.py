import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
from skimage.transform import radon
from torch import nn

from DatasetClasses.funcInterval import FuncInterval
from DatasetClasses.explicit_integrals import *



# Given alpha and h, return the solution of 
#    |t*n - alpha| = h
# where the two t solutions are given in increasing order
def find_interval(alpha, n, h):
    A_tilde = n**2
    B_tilde = 2*n*alpha 
    C_tilde1 = alpha**2 - h**2
    delta = B_tilde**2 - 4*A_tilde*C_tilde1
    if torch.abs(delta)<1e-4:
        delta = delta*0.
    t_p = (-B_tilde + torch.sqrt(delta))/(2*A_tilde)
    t_m = (-B_tilde - torch.sqrt(delta))/(2*A_tilde)
    return t_m, t_p

# Given the 1d-position of a control point and the 1d-position of another point, 
# return the coefficients of the spline to get the value at position X
def get_spline_coeffs(X_ctrl,X,h):
    dist = torch.abs(X_ctrl-X)/h
    if dist < 1.:
        a = 3./2.
        b = -5./2.
        c = 0.
        d = 1.
    elif dist < 2.:
        a = -1./2.
        b = 5./2.
        c = -4.
        d = 2.
    else:
        a = 0
        b = 0
        c = 0
        d = 0
    return a, b, c, d


def integrate(Ax,Bx,Cx,Dx,Ay,By,Cy,Dy,t0,t1):
    int_value =  (t1**7-t0**7)*Ax*Ay/7
    int_value += (t1**6-t0**6)*(Ax*By + Bx*Ay)/6
    int_value += (t1**5-t0**5)*(Ax*Cy + Bx*By + Cx*Ay)/5
    int_value += (t1**4-t0**4)*(Ax*Dy + Bx*Cy + Cx*By + Dx*Ay)/4
    int_value += (t1**3-t0**3)*(Bx*Dy +Cx*Cy + Dx*By)/3
    int_value += (t1**2-t0**2)*(Cx*Dy + Dx*Cy)/2
    int_value += (t1-t0)*(Dx*Dy)
    return int_value


# Return coefficients A, B, C and D for integration
# n is the normal direction in the wanted axis
# h is the distance betwee ctrl point
# alpha is (x0 - C), where x0 is the staring point of the line and Cx is the position of the ctrl point in the wanted axis
def coeff_integration(a,b,c,d,n,h,alpha):
    A = a*n**3/(h**3)
    B = 3*a*n**2*alpha/(h**3) + b*n**2/(h**2)
    C = a*3*n*alpha**2/(h**3) + b*2*n*alpha/(h**2) + c*n/(h)
    D = a*alpha**3/(h**3)+b*alpha**2/(h**2)+c*alpha/(h)+d
    return A, B, C, D

def get_mgrid(sidelen, dim=2):
    """
    Generates a flattened grid of (x,y) coordinates in a range of -1 to 1, Shape: (sidelen, dim)
    :param sidelen: sidelength of grid. Grid will have sidelen ** 2 points
    :param dim: dimension of coordiantes
    :return: Grid with shape: (sidelen, dim)
    """
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="xy"), dim=-1)
    return mgrid


class SplineNetwork(nn.Module):
    def __init__(self, N, circle=False):
        """
        :param N: size of sidelength. Grid size will be N*N
        """
        super().__init__()

        self.N = N
        self.weights = torch.zeros(N, N, requires_grad=True)
        # self.weights = LodopabImage(N, pad=False).image
        # self.weights = self.weights.cpu()
        self.weights = self.weights.reshape(-1, 1)
        self.weights = nn.Parameter(self.weights)
        # self.weights.data.uniform_(-1 / N, 1 / N)

        self.control_points = get_mgrid(sidelen=N, dim=2).view(-1, 2)
        #self.control_points[:,1] = - self.control_points[:,1]

        if circle:
            self.control_points = self.control_points / math.sqrt(2)
        self.control_points.requires_grad = False

    def forward(self, x):
        """
        :param x: input with shape (batch, 2)
        :return: cubic interpolation function: sum w_i conv(||x-X||/h), shape (batch)
        """
        K = 16

        device = x.device

        self.weights = self.weights.to(device)
        self.control_points = self.control_points.to(device)

        old_shape = x.shape
        is3d = False
        if x.dim() == 3:
            is3d = True
            x = x.reshape(-1, 2)

        X_i = LazyTensor(x, axis=0)
        X_j = LazyTensor(self.control_points, axis=1)

        # Calculate symbolic (euclidean) distance matrix
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        # Find indices of K nearest neighbors
        indices = D_ij.argKmin(K, dim=1)  # Shape: (batch, K)
        indices = indices.to(device)

        # access the K nearest neighbors
        neighbors = self.control_points[indices]  # Shape: (batch, K, 2)

        # Prepare input for convolutional kernel function
        h_x = (self.control_points[0][0] - self.control_points[1][0]).norm()
        h_y = (self.control_points[0][1] - self.control_points[self.N][1]).norm()
        pairwise_norm = x.unsqueeze(2) - neighbors.permute(0, 2, 1)

        input_x = pairwise_norm[:, 0, :] / h_x  # Shape: (batch, K)
        input_y = pairwise_norm[:, 1, :] / h_y  # Shape: (batch, K)

        conv_out_x = self.cubic_conv(input_x)  # Shape: (batch, K)
        conv_out_y = self.cubic_conv(input_y)  # Shape: (batch, K)

        # Perform row-wise dot product between weight and conv_out vectors
        conv_out_prod = conv_out_x * conv_out_y
        weights = self.weights[indices].squeeze()

        output = torch.sum(weights * conv_out_prod, dim=1)

        if is3d:
            output = output.reshape(old_shape[0], old_shape[1], 1)

        return output, x


    def cubic_conv(self, s):
        """
        Calculate convolutional kernel.

        :param s: ||x-X||/h, shape: (batch, K)
        :return: kernel operation on s, shape: (batch, K)
        """
        device = s.device
        result = torch.zeros(s.shape, device=device)

        cond1 = (0 <= torch.abs(s)) & (torch.abs(s) < 1)
        cond2 = (1 <= torch.abs(s)) & (torch.abs(s) < 2)
        cond3 = 2 < torch.abs(s)

        result[cond1] = (
            3 / 2 * torch.abs(s[cond1]) ** 3 - 5 / 2 * torch.abs(s[cond1]) ** 2 + 1
        )
        result[cond2] = (
            -1 / 2 * torch.abs(s[cond2]) ** 3
            + 5 / 2 * torch.abs(s[cond2]) ** 2
            - 4 * torch.abs(s[cond2])
            + 2
        )
        # result[cond3] = 0

        return result

    def integrate_line(self, z, theta):
        """
        Integrate the spline network along a line
        :param z: Position on the detection plane
        :param theta: Angle of rotation in degrees
        :return: Integral along line specified by z and theta
        """
        device = z.device
        # 0. Find line slope and normal of slope
        # 1. Find all control points that are close to the line
        #   - project all control points to normal of slope to find distance
        h_x = (self.control_points[0][0] - self.control_points[1][0]).norm()
        h_y = (self.control_points[0][1] - self.control_points[self.N][1]).norm()
        threshold = torch.max(2* h_x, 2* h_y)
        control_points = self.control_points.to(device)

        # Transform degree into radians and compute rotation matrix
        theta_rad = (theta * math.pi / 180.).to(device)
        s = torch.sin(theta_rad)
        c = torch.cos(theta_rad)

        rot = torch.stack(
            [torch.stack([c, s]), torch.stack([-s, c])]
        )
        control_points_rot = torch.matmul(control_points, rot)
        

        # Calculate the normal and its orthogonal to the line
        normal = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=device)
        # normal = torch.matmul(rot, normal)
        normal_orth = torch.tensor([[1.0], [0.0]], dtype=torch.float32, device=device)
        # normal_orth = torch.matmul(rot, normal_orth)
        # project onto the detector space (so need the orthogonal of the normal!)
        projections_orth = torch.matmul(control_points_rot, normal_orth)
        distances = torch.abs(projections_orth - z)
        indices = (distances.squeeze(1) < threshold).nonzero()

        # For each control points
        int_value_tot = torch.zeros(1,device=device,dtype=torch.float32)
        for k in indices:
            if self.weights[k,0]!=0:
                # Find values of t for which the line cross the support of the inner and outer splines
                # In the x direction
                x0 = z*1.
                alphax = (x0 - control_points_rot[k,0])        
                tx1_m,tx1_p = find_interval(alphax, normal[0], h_x)
                tx2_m,tx2_p = find_interval(alphax, normal[0], 2*h_x)
                txz_m,txz_p = find_interval(alphax, normal[0], 0)
                # In the y direction
                y0 = 0.
                alphay = (y0 - control_points_rot[k,1])        
                ty1_m,ty1_p = find_interval(alphay, normal[1], h_y)
                ty2_m,ty2_p = find_interval(alphay, normal[1], 2*h_y)
                tyz_m,tyz_p = find_interval(alphay, normal[1], 0)

                if torch.abs(txz_m-txz_p)>1e-6 or torch.abs(tyz_m-tyz_p)>1e-6:
                    print("##################")
                    print("ERROR: spline crossing 0 has more than one value, shouldn't happen!")
                    print("##################")

                # Order all the t values
                if normal[0].item() == 0:
                    t_list = torch.sort(torch.concat([ty1_m,ty1_p,ty2_m,ty2_p,tyz_m]))[0]
                elif normal[1].item() == 0:
                    t_list = torch.sort(torch.concat([tx1_m,tx1_p,tx2_m,tx2_p,txz_m]))[0]
                else:
                    t_list = torch.sort(torch.concat([tx1_m,tx1_p,tx2_m,tx2_p,txz_m,ty1_m,ty1_p,ty2_m,ty2_p,tyz_m]))[0]

                # Integrate bewteen each interval
                for ind_interv in range(len(t_list)-1):
                    # Integrate between t0 abd t1
                    t0 = t_list[ind_interv]
                    t1 = t_list[ind_interv+1]
                    # For that we need to know which spline coefficient to use
                    # define one point in the interval to know its distance to the center and the sign
                    x_interv = x0 + 0.5*(t0+t1)*normal[0]
                    y_interv = y0 + 0.5*(t0+t1)*normal[1]
                    ax, bx, cx, dx = get_spline_coeffs(control_points_rot[k,0],x_interv, h_x)
                    ay, by, cy, dy = get_spline_coeffs(control_points_rot[k,1],y_interv, h_y)
                    # Need that to take into account the abosulte values
                    sign_x = torch.sign(x_interv - control_points_rot[k,0])
                    sign_y = torch.sign(y_interv - control_points_rot[k,1])
                    if sign_x==0:
                        sign_x = torch.ones_like(sign_x)
                    if sign_y==0:
                        sign_y = torch.ones_like(sign_y)
                    if ax*ay != 0:# if 0 this means that both splines are 0 valued
                        Ax, Bx, Cx, Dx = coeff_integration(sign_x*ax,bx,sign_x*cx,dx,normal[0],h_x,alphax)
                        Ay, By, Cy, Dy = coeff_integration(sign_y*ay,by,sign_y*cy,dy,normal[1],h_y,alphay)
                        # integrate over this interval
                        int_value = integrate(Ax,Bx,Cx,Dx,Ay,By,Cy,Dy,t0,t1)
                        int_value_tot += self.weights[k,0]*int_value
                        # print(int_value)
                        # TODO: 
                        # here maybe I'm missing the normalization due to the speed of the 
                        # line not being constant from one tilt to the other. Anyway, it should
                        # at most change the value by a smal scaling factor
        return int_value_tot

                # # Approximation of the integral
                # h = h_x
                # N = 1000
                # lin = np.linspace(t0,t1,N)
                # int_discr = 0.
                # int_discr_r1 = 0.
                # int_discr_r2 = 0.
                # for tt in range(N):
                #     sx = (x0 + lin[tt]*normal[0] - control_points[k,0])/h
                #     sy = (y0 + lin[tt]*normal[1] - control_points[k,1])/h
                #     int_discr += self.cubic_conv(sx)*self.cubic_conv(sy)
                # int_discr /= N
                # int_discr *= t1-t0
                # print(int_discr)


            # TODO: forget that the spline is define with absolute value
            # Include that include two addition value of t, and pass the signa as argument of integrate function
            # Integrate function needs to be adapted then


            # # Check that the t values are really point where spline goes to 0
            # h = h_x
            # for kk in range(len(t_list)):
            #     tt = t_list[kk]
            #     xx = x0 + tt*normal[0]
            #     ssx = (xx - control_points[k,0])/h
            #     tt = t_list[kk]
            #     xx = y0 + tt*normal[1]
            #     ssy = (xx - control_points[k,1])/h
            #     ss = torch.min(torch.concat([torch.abs(self.cubic_conv(ssx)),torch.abs(self.cubic_conv(ssy))]))
            #     print(ss)

