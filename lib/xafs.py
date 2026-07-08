import numpy as np
from math import log, sin, pi

def read_9809_xafs(file, factor=1):
    energy = np.array([])
    ut = np.array([])
    with open(file, 'r') as fi:
        while True:
            lines = fi.readline()
            if not lines.find('Mono') == -1:
                d = float(lines.split('D=')[1].split('A')[0].strip())
                break
        brag = 2.9979 * 4.1357 * 1e3 / 2 / d
        while True:
            lines = fi.readline()
            if not lines.find('Offset') == -1:
                break
        while True:
            lines = fi.readline()
            if not lines or lines == '':
                break
            temp = lines.split()
            energy = np.append(energy, brag / sin(float(temp[1]) / 180 * pi))
            ut = np.append(ut, log(float(temp[3])*factor / float(temp[4])))
    return energy, ut

def read_dat(file):
    with open(file, 'r') as f:
        e, mu = [], []
        while True:
            lines = f.readline()
            if not lines or lines.strip() == '':
                break
            if (not lines.find('#') == -1) or (not lines.find('x') == -1):
                continue
            temp = lines.split()
            e.append(float(temp[0]))
            mu.append(float(temp[1]))
    return np.asarray(e), np.asarray(mu)

def xanes_analysis(energy, ut, eref=-1.0, en=20, ep=30, return_fit=False, return_norm=True):
    pre_i = np.where(energy < (eref - 30))[0]
    post_i = np.where(energy > (eref + 100))[0]
    edge_i = np.where((energy > (eref - en)) & (energy < (eref + ep)))[0]
    e0_i = np.where((energy - eref) > 0)[0][0]
    poly_deg = 1

    plm_pre = victoreen_svd(energy[pre_i], ut[pre_i])
    # plt.plot(energy, ut)
    # plt.plot(energy, energy ** (-3) * plm_pre[0] + energy ** (-4) * plm_pre[1])

    norm = ut - energy ** (-3) * plm_pre[0] - energy ** (-4) * plm_pre[1]
    plm_post = polyfit_svd(energy[post_i]-eref, norm[post_i], poly_deg)
    # plt.plot(energy, norm)
    # plt.plot(energy, plm_post[0] + (energy-eref) * plm_post[1] + (0 if poly_deg == 1 else (energy-eref) ** 2 * plm_post[2]))
    # plt.show()

    for i in range(2):
        half_height = (plm_post[0] + (energy[e0_i]-eref) * plm_post[1] + (0 if poly_deg == 1 else (energy[e0_i]-eref) ** 2 * plm_post[2])) * 0.6
        e0_i = np.argmin(np.abs(norm[edge_i] - half_height)) + int(edge_i[0])
    amp = plm_post[0] + (energy[e0_i]-eref) * plm_post[1] + (0 if poly_deg == 1 else (energy[e0_i]-eref) ** 2 * plm_post[2])

    if return_fit:
        norm[e0_i:] -= ((energy[e0_i:]-eref) * plm_post[1] + (0 if poly_deg == 1 else (energy[e0_i:]-eref) ** 2 * plm_post[2]))
        #plt.plot(energy, norm)
        #plt.show()
        return norm / (amp if return_norm else 1)
    else:
        return norm / (amp if return_norm else 1)
    
def xanes_analysis_2d(energy, ut, eref=-1.0, en=20, ep=30, return_norm=False):
    post_deg = 2 if (energy[-1] - min(eref + 150,
                                           (energy[-1] - eref) / 3 + eref)) > 350 else 1
    epre = np.where(energy < (eref - 30))[0]
    epost = np.where(energy > (eref + 150))[0]
    edge = np.where((energy < (eref + ep)) & (energy > (eref - en)))[0]
    poly_pre = victoreen_svd(energy[epre], ut[:, epre].T)

    norm = ut - (energy ** (-3))[None, :] * poly_pre[0][:, None] - (
            energy ** (-4))[None, :] * poly_pre[1][:, None]
    poly_post = polyfit_svd(energy[epost] - eref, norm[:, epost].T, post_deg)
    e0_i = np.zeros(norm.shape[0], dtype=int) + np.where(energy > eref)[0][0]
    half_height = (poly_post[0] + (energy[e0_i] - eref) * poly_post[1] +
                   + (0 if poly_post.shape[0] == 2 else (energy[e0_i] - eref) ** 2 * poly_post[2])) * 0.6
    e0_i = np.argmin(np.abs(norm[:, edge] - half_height[:, None]), axis=1) + int(edge[0])
    amp = (poly_post[0] + (energy[e0_i] - eref) * poly_post[1]
                + (0 if poly_post.shape[0] == 2 else (energy[e0_i] - eref) ** 2 * poly_post[2]))
    e0_ref_i = np.where(energy-eref>0)[0][0]
    norm[:, e0_ref_i:] -= (
        ((energy[e0_ref_i:] - eref)[None, :] * poly_post[1][:, None]
         + (0 if poly_post.shape[0] == 2 else (energy[e0_ref_i:] - eref)[None, :] ** 2 *
                                              poly_post[2][:, None])))
    return norm / (amp[:, None] if return_norm else 1)

def victoreen(e_in, mu_in):
    matr = np.vstack([e_in ** (-3), e_in ** (-4)]).T
    mtm = matr.T @ matr
    mty = matr.T @ mu_in
    return np.linalg.solve(mtm, mty)

def polyfit(e_in, mu_in, degree):
    matr = np.vstack([e_in ** _ for _ in range(degree + 1)]).T
    mtm = matr.T @ matr
    mty = matr.T @ mu_in
    return np.linalg.solve(mtm, mty)

def victoreen_svd(e_in, mu_in):
    matr = np.vstack([e_in ** (-3), e_in ** (-4)]).T
    u, s, vt = np.linalg.svd(matr, full_matrices=False)
    s_inv = np.diag(1 / s)
    a_pinv = vt.T @ s_inv @ u.T
    return a_pinv @ mu_in

def polyfit_svd(e_in, mu_in, degree):
    matr = np.vstack([e_in ** _ for _ in range(degree + 1)]).T
    u, s, vt = np.linalg.svd(matr, full_matrices=False)
    s_inv = np.diag(1 / s)
    a_pinv = vt.T @ s_inv @ u.T
    return a_pinv @ mu_in

def add_gaussian_noise(signal, snr_db=20):
    signal_power = np.mean(signal ** 2, axis=-1, keepdims=True)
    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)
    return signal + noise