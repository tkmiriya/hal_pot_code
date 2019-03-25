import matplotlib.pyplot as plt
import numpy as np
import os, glob
import scipy.optimize

import scipy.constants

hbarc = (scipy.constants.hbar * scipy.constants.speed_of_light
         / scipy.constants.mega / scipy.constants.electron_volt / scipy.constants.femto)

class Corr_2pt_Baryon(object):
    """ baryon 2pt correlator 
    corr_type : ex.  proton_CG05_CG05
    bin_size : size of jackknife bin
    result_dir : directory of "correlator.PS.dir"
    confMax : default None (use all available files in result_dir dir)
    lat_unit : default None (use a = 1)  or lattice spacing in fm
    """
    def __init__(self, corr_type, bin_size, result_dir='results', confMax = None, lat_unit = None):
        self.corr_type = corr_type
        self.bin_size = bin_size
        self.result_dir = result_dir
        self.confMax = confMax
        self.lat_unit = lat_unit 

        if not lat_unit:
            self.ainv = 1.0 # a  = 1
        else:
            self.ainv = hbarc/self.lat_unit
        
        self.eval_corr_meff()
        
    def eval_corr_meff(self):
        dir_list = os.listdir(self.result_dir + '/correlator.PS.dir/')
        dir_list.sort()
        corrs = []
        corrs_bin = {}
        print('# Read {}'.format(self.corr_type))
        for fw_bw in ['', 'anti']:
            corrs_bin[fw_bw] = []
            for dir_name in dir_list[:self.confMax]:
                files = glob.glob(self.result_dir + '/correlator.PS.dir/' 
                                  + dir_name + '/' + fw_bw + self.corr_type + '_*')
                for file in files:
                    corrs.append(np.loadtxt(file))
                
                if len(corrs) == self.bin_size:
                    corrs_bin[fw_bw].append(np.mean(np.array(corrs)[:,:,1], axis=0))
                    corrs = []
            corrs = [] # reset

        corrs_fw_bin = np.array(corrs_bin[''])
        corrs_bw_bin = np.array(corrs_bin['anti'])

        self.bin_num, self.nt = corrs_fw_bin.shape
        self.nconf = self.bin_num * self.bin_size
        print(f'# total {self.nconf} conf., bin size = {self.bin_size}, number of samples =  {self.bin_num}')

        corrs_2pt_fb = self.proj(corrs_fw_bin, corrs_bw_bin)
        ip = 0
        self.corr_jk = (np.sum(corrs_2pt_fb[ip][:,:], axis=0) 
                    - corrs_2pt_fb[ip])/float(self.bin_num - 1)
        self.nt  = self.corr_jk.shape[1]
        self.meff_jk = np.log(self.corr_jk[:,:self.nt//2]/self.corr_jk[:,1:self.nt//2+1])

    def proj(self, corrs_fw_bin, corrs_bw_bin):
        corrs_2pt_baryon = [corrs_fw_bin, corrs_fw_bin, corrs_bw_bin, corrs_bw_bin]
        
        corrs_2pt_pm_fw = self.calc_proj_pm_2pt(corrs_2pt_baryon, (0,1))
        corrs_2pt_pm_bw = self.calc_proj_pm_2pt(corrs_2pt_baryon, (1,0))
        corrs_2pt_fb = self.folding(corrs_2pt_pm_fw, corrs_2pt_pm_bw)
        return corrs_2pt_fb
        
    def folding(self, corrs_2pt_pm_fw, corrs_2pt_pm_bw):
        SrcT = 1 # 
        Iflg_bc = 'p'
        corrs_2pt_fb = {}
        its = np.arange(self.nt)
        itrevs = (self.nt - its + 2*(SrcT - 1)) % self.nt

        for ip, sign in enumerate([+1.0, -1.0]):
            if Iflg_bc == 'd':
                corrs_2pt_fb[ip] = corrs_2pt_pm_fw[ip][:,its]
            elif Iflg_bc == 'p':
                corrs_2pt_fb[ip] = sign * 0.5 * (corrs_2pt_pm_fw[ip][:,its] 
                          + corrs_2pt_pm_bw[ip][:,itrevs])
            elif Iflg_bc == 'a':
                corrs_2pt_fb[ip] = sign * 0.5 * (corrs_2pt_pm_fw[ip][:,its]
                          - corrs_2pt_pm_bw[ip][:,itrevs])
                    
        return corrs_2pt_fb
        
    def calc_proj_pm_2pt(self, corr_2pt, mode):
# projection  ---      particle (1 + gamma_4)/2, (1 - gamma_4)/2
#             --- anti-partycle (1 - gamma_4)/2, (1 + gamma_4)/2
        corrs_2pt_parity = {}
        corrs_2pt_parity[mode[0]] = corr_2pt[0] + corr_2pt[1]
        corrs_2pt_parity[mode[1]] = corr_2pt[2] + corr_2pt[3]
        return corrs_2pt_parity

    def plot_meff(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        if not self.lat_unit:
            ax.set_ylabel(r'$m_\mathrm{eff}(t)$ [$a^{-1}$]')
        else:
            ax.set_ylabel(r'$m_\mathrm{eff}(t)$ [MeV]')

        ax.errorbar(np.arange(self.nt//2),
                    self.meff_jk.mean(axis=0)*self.ainv,
                    self.meff_jk.std(axis=0)*np.sqrt(self.bin_num - 1)*self.ainv,
                    fmt='o', color='red', mfc='none', mew=2.0, capthick=2.0, capsize=5)
        ax.set_xlabel(r'$t$ [$a$]')

    def fit_meff(self, fit_min=10, fit_max=15, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        xs = np.arange(self.nt//2)
        yerrs = self.meff_jk.std(axis=0) * np.sqrt(self.bin_num - 1)
        ys_jk = self.meff_jk
        mask = (xs >= fit_min) & (xs <= fit_max)
        ffit = lambda p, x: p[0]
        errf = lambda p, y, x, err: (ffit(p,x)-y)/err
        fit_vals = []
        p0 = [0.5]
        for ibin in range(self.bin_num):
            pfit = scipy.optimize.leastsq(errf, p0, args=(
                ys_jk[ibin,mask], xs[mask], yerrs[mask]), full_output=True)
            fit_vals.append(pfit[0])
        fit_vals = np.array(fit_vals)
        fit_av = fit_vals.mean()
        fit_err = fit_vals.std()*np.sqrt(self.bin_num - 1)

        self.plot_meff(ax=ax)
        its = np.linspace(fit_min-0.2, fit_max+0.2)
        ax.plot(its, np.zeros_like(its) + fit_av*self.ainv, lw=5, color='blue',
                label=r'${:3.4f} \pm {:3.4f}$'.format(fit_av*self.ainv, fit_err*self.ainv))
        ax.plot(its, np.zeros_like(its) + (fit_av + fit_err)*self.ainv, ls='--', lw=1, color='blue')
        ax.plot(its, np.zeros_like(its) + (fit_av - fit_err)*self.ainv, ls='--', lw=1, color='blue')
        ax.fill_between(its, 
                        np.zeros_like(its) + (fit_av + fit_err)*self.ainv, 
                        np.zeros_like(its) + (fit_av - fit_err)*self.ainv, 
                        color='blue', alpha=0.2)
        return fit_av, fit_err
        


class Corr_2pt_2Baryons(object):
    """ 2-baryon 2pt correlator 
    channel : ex.  nn  or xixi
    bin_size : size of jackknife bin
    result_dir : directory of "correlator.multi.dir"
    confMax : default None (use all available files in result_dir dir)
    lat_unit : default None (use a = 1)  or lattice spacing in fm
    """
    def __init__(self, channel, bin_size, result_dir, confMax = None,  lat_unit = None):
        self.corr_label = channel

        self.lat_unit = lat_unit 

        if not lat_unit:
            self.ainv = 1.0 # a  = 1
        else:
            self.ainv = hbarc/self.lat_unit

        if channel == 'nn':
            self.corr_type = 'O01O02_O01O02_CG05_CG05'
            _opt = '*'
        elif channel == 'xixi':
            self.corr_type = 'O06O07_O06O07_CG05_CG05'
            _opt = '*'
            
        self.bin_size = bin_size

        dir_list = os.listdir(result_dir + '/correlator.multi.dir/')
        dir_list.sort()
        corrs = []
        corrs_1s0_bin = []
        corrs_3s1_bin = []
        print('Read {}'.format(channel))
        for dir_name in dir_list[:confMax]:
            files = glob.glob(result_dir + '/correlator.multi.dir/' + dir_name + '/' + self.corr_type + _opt)
            for file in files:
        #        print('read >> ...', file[-70:])
                corrs.append(np.loadtxt(file))
                
            if len(corrs) == bin_size:
        #        print('--binning--')
                corrs_1s0_bin.append(np.mean(np.array(corrs)[:,:,1], axis=0))
                corrs_3s1_bin.append(np.mean(np.array(corrs)[:,:,3], axis=0))
                corrs = []

        self.corrs_bin = {}
        self.corrs_bin['1s0'] = np.array(corrs_1s0_bin)
        self.corrs_bin['3s1'] = np.array(corrs_3s1_bin)

        self.bin_num, self.nt2 = self.corrs_bin['1s0'].shape

        self.nconf = self.bin_num * bin_size

        self.corr_jk = {}
        self.Eeff_jk = {}
        for spin in ['1s0', '3s1']:
            self.corr_jk[spin] = (np.sum(self.corrs_bin[spin][:,:], axis=0) 
                        - self.corrs_bin[spin])/float(self.bin_num - 1)
            self.Eeff_jk[spin] = np.log(self.corr_jk[spin][:,:self.nt2-1]/self.corr_jk[spin][:,1:self.nt2])

        print(f'# total {self.nconf} conf., bin size = {self.bin_size}, number of samples =  {self.bin_num}')

    def plot_Eeff(self, spin='1s0', ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.errorbar(np.arange(self.nt2-1),
                    self.Eeff_jk[spin].mean(axis=0)*self.ainv,
                    self.Eeff_jk[spin].std(axis=0)*np.sqrt(self.bin_num - 1)*self.ainv,
                    fmt='o', color='red', mfc='none', mew=2.0, capthick=2.0, capsize=5)
        ax.set_xlabel(r'$t$ [$a$]')
        if not self.lat_unit:
            ax.set_ylabel(r'$E_\mathrm{{eff}}^\mathrm{{BB}}(t)$ [$a^{{-1}}$] {}'.format(spin))
        else:
            ax.set_ylabel(r'$E_\mathrm{{eff}}^\mathrm{{BB}}(t)$ [MeV] {}'.format(spin))

    def fit_Eeff(self, spin='1s0', fit_min=10, fit_max=15, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        xs = np.arange(self.nt2 - 1)
        yerrs = self.Eeff_jk[spin].std(axis=0) * np.sqrt(self.bin_num - 1)
        ys_jk = self.Eeff_jk[spin]
        mask = (xs >= fit_min) & (xs <= fit_max)
        ffit = lambda p, x: p[0]
        errf = lambda p, y, x, err: (ffit(p,x)-y)/err
        fit_vals = []
        p0 = [0.5]
        for ibin in range(self.bin_num):
            pfit = scipy.optimize.leastsq(errf, p0, args=(
                ys_jk[ibin,mask], xs[mask], yerrs[mask]), full_output=True)
            fit_vals.append(pfit[0])
        fit_vals = np.array(fit_vals)
        fit_av = fit_vals.mean()
        fit_err = fit_vals.std()*np.sqrt(self.bin_num - 1)

        self.plot_Eeff(ax=ax)
        its = np.linspace(fit_min-0.2, fit_max+0.2)
        ax.plot(its, np.zeros_like(its) + fit_av*self.ainv, lw=5, color='blue',
                label=r'${:3.4f} \pm {:3.4f}$'.format(fit_av*self.ainv, fit_err*self.ainv))
        ax.plot(its, np.zeros_like(its) + (fit_av + fit_err)*self.ainv, ls='--', lw=1, color='blue')
        ax.plot(its, np.zeros_like(its) + (fit_av - fit_err)*self.ainv, ls='--', lw=1, color='blue')
        ax.fill_between(its, 
                        np.zeros_like(its) + (fit_av + fit_err)*self.ainv, 
                        np.zeros_like(its) + (fit_av - fit_err)*self.ainv, 
                        color='blue', alpha=0.2)
        return fit_av, fit_err
 

class Delta_Eeff(object):
    """ calculate effective energy shift 
    channel : ex.  nn  or xixi
    bin_size : size of jackknife bin
    result_dir : directory of "correlator.multi.dir"
    confMax : default None (use all available files in result_dir dir)
    lat_unit : default None (use a = 1)  or lattice spacing in fm
    """
    def __init__(self, channel, bin_size, result_dir, confMax = None, lat_unit = None):
        self.lat_unit = lat_unit
        if not lat_unit:
            self.ainv = 1.0 # a  = 1
        else:
            self.ainv = hbarc/self.lat_unit
 
        cbb = Corr_2pt_2Baryons(channel, bin_size, result_dir, confMax=confMax, lat_unit=lat_unit)

        corr_type = {'xixi': 'Xi_CG05_CG05', 'nn': 'proton_CG05_CG05'}[channel]
        cb = Corr_2pt_Baryon(corr_type, bin_size, result_dir=result_dir, confMax=confMax, lat_unit=lat_unit)

        if cb.bin_num == cbb.bin_num:
            self.bin_num = cb.bin_num
        else:
            print('Error: mismatch of baryon corr and two-baryon corr.')


        self.nts = np.arange(cbb.nt2 - 1)
        self.deltaEeff_jk = {}
        for spin in ['1s0', '3s1']:
            self.deltaEeff_jk[spin] = np.array([
                np.log(cbb.corr_jk[spin][:,it]/(cb.corr_jk[:,it]**2) \
                /(cbb.corr_jk[spin][:,it+1]/(cb.corr_jk[:,it+1]**2)))
                for it in self.nts ]).T


    def plot_dEeff(self, spin='1s0', ax=None):
        if not ax:
            fig, ax = plt.subplots()


        ax.errorbar(self.nts,
                    self.deltaEeff_jk[spin].mean(axis=0)*self.ainv,
                    self.deltaEeff_jk[spin].std(axis=0)*np.sqrt(self.bin_num - 1)*self.ainv,
                    fmt='o', color='red', mfc='none', mew=2.0, capthick=2.0, capsize=5)
        ax.set_xlabel(r'$t$ [$a$]')
        if not self.lat_unit:
            ax.set_ylabel(r'$\Delta E_\mathrm{{eff}}^\mathrm{{BB}}(t)$ [$a^{{-1}}$] {}'.format(spin))
        else:
            ax.set_ylabel(r'$\Delta E_\mathrm{{eff}}^\mathrm{{BB}}(t)$ [MeV] {}'.format(spin))

        ax.axhline(0, color='black')

    def fit_dEeff(self, spin='1s0', fit_min=10, fit_max=15, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        xs = self.nts
        yerrs = self.deltaEeff_jk[spin].std(axis=0) * np.sqrt(self.bin_num - 1)
        ys_jk = self.deltaEeff_jk[spin]
        mask = (xs >= fit_min) & (xs <= fit_max)
        ffit = lambda p, x: p[0]
        errf = lambda p, y, x, err: (ffit(p,x)-y)/err
        fit_vals = []
        p0 = [0.5]
        for ibin in range(self.bin_num):
            pfit = scipy.optimize.leastsq(errf, p0, args=(
                ys_jk[ibin,mask], xs[mask], yerrs[mask]), full_output=True)
            fit_vals.append(pfit[0])
        fit_vals = np.array(fit_vals)
        fit_av = fit_vals.mean()
        fit_err = fit_vals.std()*np.sqrt(self.bin_num - 1)

        self.plot_dEeff(spin=spin, ax=ax)
        its = np.linspace(fit_min-0.2, fit_max+0.2)
        ax.plot(its, np.zeros_like(its) + fit_av*self.ainv, lw=5, color='blue',
                label=r'${:3.4f} \pm {:3.4f}$'.format(fit_av*self.ainv, fit_err*self.ainv))
        ax.plot(its, np.zeros_like(its) + (fit_av + fit_err)*self.ainv, ls='--', lw=1, color='blue')
        ax.plot(its, np.zeros_like(its) + (fit_av - fit_err)*self.ainv, ls='--', lw=1, color='blue')
        ax.fill_between(its, 
                        np.zeros_like(its) + (fit_av + fit_err)*self.ainv, 
                        np.zeros_like(its) + (fit_av - fit_err)*self.ainv, 
                        color='blue', alpha=0.2)
        return fit_av, fit_err
 
