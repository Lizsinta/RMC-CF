import os
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from math import exp as mexp
from time import perf_counter as timer, strftime, localtime
from time import time

from scipy.optimize import nnls
from random import randrange

from xafs import xanes_analysis, read_9809_xafs, read_dat, add_gaussian_noise

def metropolis(r_0, r, tau=0.0):
    if r_0 > r:
        return True
    else:
        if tau == 0:
            return False
        met = mexp(-(r - r_0) / tau)
        judge = randrange(0, 100)
        if (judge / 100) < met:
            return True
        else:
            return False

class Worker(QThread):
    sig_warning = pyqtSignal(str)
    sig_statusbar = pyqtSignal(str, int)

    sig_change_tau = pyqtSignal(float)
    sig_flush = pyqtSignal(int)#, int, float)
    def __init__(self, parent=None, file='', element='', ratio_axis=np.array([]), ratio_plane=np.array([]), snr=50):
        super(Worker, self).__init__(parent)
        self.folder = os.path.dirname(file)
        self.folder_save = self.folder + (r'\RMC' if element == '' else '')
        print(self.folder, self.folder_save)
        file = file.replace('\\', '/')
        self.fname = file.split('/')[-1].split('.')[0]
        self.flag = False
        self.step_count = 0
        self.file = ''
        self.suffix = ''
        self.element = element
        self.species = ['Cu', 'Cu2O', 'CuO']#np.array(['Fe', 'FeO', 'Fe2O3a'])
        self.eref = 8979
        self.e0_ref_i = -1
        self.plot_flag = False

        # energy related
        self.energy = np.array([])
        self.edge = np.array([])
        self.emin = -1
        self.emax = -1
        self.edge_temp = np.array([])
        self.edge_change_flag = False

        # spectrum related
        self.exper0 = np.array([])
        self.exper = np.array([])
        self.exper_bottom = np.array([])
        self.ref0 = np.array([])
        self.ref = np.array([])
        self.i1_group = np.array([])
        self.fit = np.array([])

        # model related
        if ratio_axis.size > 0 and ratio_plane.size > 0:
            self.ratio_gauss = np.random.randn(ratio_axis.shape[0], 10, ratio_axis.shape[1]) * 0.1
            self.ratio_axis = ratio_axis.clip(1e-6, None)
            self.ratio_plane = ratio_plane.clip(1e-6, None)
        else:
            self.ratio_gauss = np.array([])
            self.ratio_axis = np.array([])
            self.ratio_plane = np.array([])
        self.noise = snr

        # fitting relate
        self.nblock = 100
        self.std = np.arange(3, dtype=int)
        self.step_size = int(self.nblock / 10)
        self.ratio = np.zeros((self.nblock, self.std.size))#(1 / self.std_amount) # np.random.rand(self.nblock, self.std.size).clip(1e-6, 1)/self.std.size#
        self.ratio_best = self.ratio.copy()
        self.i1_limit = mexp(-6)
        mvRange, mvReso = 0.01, 0.01
        self.mvRange = int(mvRange / mvReso)
        self.mvReso = int(1 / mvReso)

        # metropolis related
        self.tau = 1e-12 * 10 ** np.log10(self.nblock)
        self.r_factor = -1
        self.r_factor_best = -1
        self.r_factor_seq = []
        self.tau_seq = []
        self.auto_tau_flag = False
        self.ac_rate = 0.0

        # timer related
        self.timestamp = 0.0
        self.start_stamp = time()
        self.accum_time = 0.0


    def read(self):
        # read target and reference spectrum
        try:
            e, mu = read_dat(self.folder + '/' + self.fname + '.dat')
        except Exception as e:
            self.sig_warning.emit(f'{e}')
            return False

        # target background removal
        self.energy, mu = np.array(e), np.array(mu)
        e_diff = np.diff(self.energy, prepend=self.energy[-1:])
        ref_i = np.argmax(np.diff(mu, prepend=mu[-1]) / e_diff)
        table_e0 = np.loadtxt(os.getcwd() + r'\energy table.dat', usecols=2, dtype=float)
        table_ele = np.loadtxt(os.getcwd() + r'\energy table.dat', usecols=1, dtype=str)
        self.element = table_ele[np.argmin(np.abs(table_e0 - self.energy[ref_i]))]
        self.eref = table_e0[np.argmin(np.abs(table_e0 - self.energy[ref_i]))]
        self.e0_ref_i = np.argmin(np.abs(self.energy - self.eref))
        print(self.eref, self.element)

        ref_folder = os.getcwd() + f'\\ref\\{self.element}'
        self.species = [_.split('.')[0] for _ in os.listdir(ref_folder)]
        self.std = np.arange(len(self.species))

        self.edge = np.where((self.energy < (self.eref + 50)) & (self.energy > (self.eref - 30)))[0]
        self.emin, self.emax = self.energy[self.edge[0]], self.energy[self.edge[-1]]

        mu = xanes_analysis(self.energy, mu, self.eref, en=30, ep=50, return_fit=True, return_norm=False)
        self.exper0 = np.exp(-mu)
        self.exper = self.exper0[self.edge]

        # std. background removal
        ref = []
        for i in self.std:
            file = ref_folder + r'/%s.dat' % self.species[i]
            with open(file, 'r') as f:
                if f.readline().find('9809') == -1:
                    e, mu = read_dat(file)
                else:
                    e, mu = read_9809_xafs(file)
            ut = xanes_analysis(e, mu, self.eref, en=30, ep=50, return_fit=True)
            ref.append(np.interp(self.energy, e, ut))
        self.ref0 = np.array(ref)
        self.ref = self.ref0[:, self.edge]
        return True

    def read_ref(self):
        table_e0 = np.loadtxt(os.getcwd() + r'\energy table.dat', usecols=2, dtype=float)
        table_ele = np.loadtxt(os.getcwd() + r'\energy table.dat', usecols=1, dtype=str)
        self.eref = table_e0[np.where(table_ele == self.element)[0][0]]

        ref_folder = os.getcwd() + f'\\ref\\{self.element}'
        self.species = [_.split('.')[0] for _ in os.listdir(ref_folder)]
        self.std = np.arange(len(self.species))

        # std. background removal
        ref = []
        first_file = True
        for i in self.std:
            file = ref_folder + r'/%s.dat' % self.species[i]
            with open(file, 'r') as f:
                if f.readline().find('9809') == -1:
                    e, mu = read_dat(file)
                else:
                    e, mu = read_9809_xafs(file)
            if first_file:
                self.energy = e
                self.edge = np.where((self.energy < (self.eref + 50)) & (self.energy > (self.eref - 30)))[0]
                self.emin, self.emax = self.energy[self.edge[0]], self.energy[self.edge[-1]]
            ut = xanes_analysis(self.energy, mu, self.eref, en=30, ep=50, return_fit=True)
            ref.append(np.interp(self.energy, e, ut))
        self.ref0 = np.array(ref)
        self.ref = self.ref0[:, self.edge]

    def build_model(self):
        self.read_ref()
        # ratio_axis = self.ratio_axis[:, None, :] + np.random.randn(self.ratio_axis.shape[0], 10, self.ratio_axis.shape[1]) * 0.1
        i1fit = np.exp(-(self.ratio_axis[:, self.std] @ self.ref0[self.std]))
        exper0 = self.ratio_plane @ i1fit.clip(self.i1_limit, 1)
        if self.noise > 0:
            self.exper0 = add_gaussian_noise(exper0, self.noise)
        else:
            self.exper0 = exper0
        self.exper = self.exper0[self.edge]
        print('rf', np.sum((exper0[self.edge] - self.exper) ** 2) / np.sum(exper0[self.edge] ** 2))
        print('lcf', nnls(self.ref.T, -np.log(self.exper))[0])

    def init(self, ratio_reset=True):
        if self.energy.size == 0:
            self.read_ref()
        # manually run 1 time for updating, for parameters regulation before simulation
        # parameters regulation: std. selection (self.std), fitting energy range (self.edge)
        self.ref = self.ref0[self.std][:, self.edge]
        if ratio_reset:
            self.ratio = np.random.rand(self.nblock, self.std.size).clip(1e-6, 1)  / self.std.size
        #self.ratio[:, 1] += 0.6
        self.i1_group = np.exp(-(self.ratio @ self.ref)).clip(self.i1_limit, 1)  # calculated ut group
        self.fit = self.i1_group.mean(0)# calculate i1 and sum

        self.exper = self.exper0[self.edge]
        self.exper_bottom = np.sum(self.exper ** 2)
        self.r_factor = np.sum((self.fit - self.exper) ** 2) / self.exper_bottom
        self.r_factor_best = self.r_factor
        self.ratio_best = self.ratio.copy()
        self.r_factor_seq = []

        #self.sig_current.emit(self.r_factor)
        #self.sig_best.emit(self.r_factor_best)
        self.plot_flag = True
        self.sig_flush.emit(0)

    def load(self):
        # reload the previous results
        file = self.folder_save + '/' + self.fname + f'_ratio{self.suffix}.dat'
        with open(file, 'r') as f:
            lines = []
            while True:
                temp = f.readline()
                if temp.find('#') == -1:
                    break
                lines.append(temp)

        lines = np.asarray(lines)
        head_dat = np.char.split(lines, ':')
        head = np.array([_[0][2:] for _ in head_dat])
        dat = np.array([_[1].strip() if len(_) > 1 else '' for _ in head_dat])
        print(head)
        print(dat)
        edge = dat[np.char.find(head, 'emi')>=0][0][:-2]
        self.emin = float(edge.split('-')[0])
        self.emax = float(edge.split('-')[1])

        fitting = np.loadtxt(self.folder_save + '/' + self.fname + f'_fitting{self.suffix}.dat', dtype=float).T
        self.energy = fitting[0]


        self.species = dat[np.char.find(head, 'std')>=0][0].split('-')
        self.std = np.arange(len(self.species))

        self.accum_time = float(dat[np.char.find(head, 'time')>=0][0].split('s')[0].strip())
        self.step_size = int(dat[np.char.find(head, 'step')>=0][0])
        self.step_count = int(dat[np.char.find(head, 'count')>=0][0])
        self.tau = float(dat[np.char.find(head, 'tau')>=0][0])
        if (np.char.find(head, 'r2')>=0).any():
            self.r_factor_best = float(dat[np.char.find(head, 'r2') >= 0][0])
        if (np.char.find(head, 'walk')>=0).any():
            walk = dat[np.char.find(head, 'walk') >= 0][0].split()
            self.mvRange = int(float(walk[0])/float(walk[1]))
            self.mvReso = int(1/float(walk[1]))
        self.edge = np.where((self.energy > self.emin) & (self.energy < self.emax))[0]
        self.ratio = np.loadtxt(file, dtype=float)
        self.nblock = self.ratio.shape[0]

        self.exper0 = np.exp(-fitting[1])
        self.exper = self.exper0[self.edge]
        self.exper_bottom = np.sum(self.exper ** 2)
        self.ref0 = np.zeros((len(self.species), self.energy.size))
        self.ref0[self.std] += fitting[3:]
        self.ref = self.ref0[:, self.edge]

        lcf = nnls(self.ref.T, -np.log(self.exper))[0]
        i1_lcf = np.exp(-lcf @ self.ref)
        rf_lcf = np.sum((i1_lcf - self.exper) ** 2) / np.sum(i1_lcf ** 2)
        i1_rmc = np.exp(-(self.ratio @ self.ref[self.std])).clip(self.i1_limit, 1).mean(0)

        print('lcf', lcf)
        print('rf lcf', rf_lcf)
        print('rmc', self.ratio.mean(0))
        print('rf rmc', np.sum((i1_rmc - self.exper) ** 2) / self.exper_bottom)


    def save(self):
        if not os.path.exists(self.folder_save):
            os.makedirs(self.folder_save)
        fname = self.folder_save + f'/{self.fname}'
        used_std = 'std: '
        for i in self.std:
            used_std += '%s-' % self.species[i]
        used_std = used_std[:-1] + '\n'
        edge_info = f'emin-emax: {self.emin:.1f}-{self.emax:.1f}eV\n'
        time_info = f'time consume: {self.accum_time:.2f}s\n'
        nblock_info = f'nblock: {self.nblock}\n'
        count_info = f'count: {self.step_count}\n'
        step_info = f'step size: {self.step_size}\n'
        walk_info = f'walk param (range, step): {self.mvRange / self.mvReso:.3f} {1 / self.mvReso:.3f}\n'
        rf_info = f'best r2: {self.r_factor_best:.8e}\n'
        tau_info = f'tau: {self.tau:.1e}\n'

        np.savetxt(fname + r'_ratio.dat', self.ratio,
                   fmt='%.3f', header=f'{edge_info}{nblock_info}{time_info}'
                                      f'{count_info}{step_info}{walk_info}{rf_info}{tau_info}{used_std}')
        # np.save(fname + r'_rfactor%s' % self.thread.suffix, np.asarray(self.thread.r_factor_seq))
        # np.save(fname + r'_tau%s' % self.thread.suffix, np.asarray(self.thread.tau_seq))

        fit = -np.log(np.exp(-(self.ratio @ self.ref0[self.std])).clip(self.i1_limit, 1).mean(0))
        wdata = np.vstack((self.energy, -np.log(self.exper0), fit))
        for i in self.std:
            wdata = np.vstack((wdata, self.ref0[i]))
        np.savetxt(fname + r'_fitting.dat', wdata.T, fmt='%.8f',
                   header=f'#fitting: e-mu-fit-{used_std[:-1]} [R-factor:{self.r_factor:.3e}]')

    def run(self):
        self.sig_statusbar.emit('Running', 0)
        r_diff = 0
        gamma = 1e-4 ** (1 / 10000)
        start_t = timer()  # - self.timestamp
        self.start_stamp = time()
        accum_time = 0.0
        acc = 0
        tau_step = 0.1
        while True:
            self.step_count += 1
            '''if self.step_count == 100000:
                self.sig_autoend.emit(0)
                break'''
            # asynchronous update of fitting range
            if self.edge_change_flag:
                self.edge = self.edge_temp.copy()
                self.ref = self.ref0[self.std][:, self.edge]
                self.i1_group = np.exp(-(self.ratio @ self.ref)).clip(self.i1_limit, 1)
                self.exper = self.exper0[self.edge]
                self.exper_bottom = np.sum(self.exper ** 2)
                self.edge_change_flag = False
                self.fit = self.i1_group.mean(0)
                self.r_factor = np.sum((self.fit - self.exper) ** 2) / self.exper_bottom
                self.r_factor_best = self.r_factor
                # self.sig_current.emit(self.r_factor)
                # self.sig_best.emit(self.r_factor_best)
                self.plot_flag = True
                self.sig_flush.emit(0)
                self.step_count -= 1
                continue

            # tau tuning by acceptant rate
            if self.step_count % 10000 == 0:
                self.ac_rate = acc / 10000 * self.nblock
                if self.auto_tau_flag:
                    if self.ac_rate < 0.2:
                        self.tau *= (1 + tau_step)
                        self.sig_change_tau.emit(self.tau)
                    elif self.ac_rate > 0.4:
                        self.tau *= (1 - tau_step)
                        self.sig_change_tau.emit(self.tau)
                acc = 0

            # randomly select self.step_size groups, randomly change [self.step_size, self.std_i.size] ratio
            target = np.random.permutation(self.nblock)[:self.step_size]
            ratio_temp = self.ratio[target].copy()
            # moving = (1 + (2*np.random.randint(0, 2, (target.size, self.std_i.size))-1)/10)
            # ratio_temp *= moving
            moving = np.random.randint(-self.mvRange, self.mvRange + 1, (target.size, self.std.size)) / self.mvReso
            ratio_temp = (ratio_temp + moving).clip(1e-6, None)  # cut from 0 after change
            ratio_temp[ratio_temp > 10] = 1e-6

            # recalculate the changed group->new R2
            self.i1_group[target] = np.exp(-(ratio_temp @ self.ref)).clip(self.i1_limit, 1)
            fit = self.i1_group.mean(0)
            r_factor_new = np.sum((fit - self.exper) ** 2) / self.exper_bottom

            # compared new R2 with old R2
            # r_diff += r_factor_new - self.r_factor
            result = metropolis(self.r_factor, r_factor_new, self.tau)
            accum_time = timer() - start_t
            self.timestamp = self.accum_time + accum_time
            if result:
                # self.tau_seq.append(self.tau)
                # if accepted, update ratio, r2, plotting
                acc += 1
                self.r_factor = r_factor_new
                # self.r_factor_seq.append(r_factor_new)
                self.ratio[target] = ratio_temp.copy()
                if self.r_factor < self.r_factor_best:
                    # when new R2 is the lowest R2
                    self.ratio_best = self.ratio.copy()
                    self.r_factor_best = self.r_factor
                self.fit = fit.copy()
                # replot the fitting spectrum
                self.plot_flag = True
            else:
                # if reject, recover the selected group
                self.i1_group[target] = np.exp(-(self.ratio[target] @ self.ref)).clip(self.i1_limit, 1)
            self.sig_flush.emit(0)
            if not self.flag:
                break
        self.accum_time += accum_time