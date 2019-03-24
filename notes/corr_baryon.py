import matplotlib.pyplot as plt
import numpy as np
import os, glob
import scipy.optimize

class Corr_2pt_Baryon(object):
    """ baryon 2pt correlator 
    corr_type : ex.  proton_CG05_CG05
    bin_size : size of jackknife bin
    result_dir : directory of "correlator.PS.dir"
    confMax : default None (use all available files in result_dir dir)
    """
    def __init__(self, corr_type, bin_size, result_dir='results', confMax = None):
        self.corr_type = corr_type
        self.bin_size = bin_size
        self.result_dir = result_dir
        self.confMax = confMax
        
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
        ax.errorbar(np.arange(self.nt//2),
                    self.meff_jk.mean(axis=0),
                    self.meff_jk.std(axis=0)*np.sqrt(self.bin_num - 1),
                    fmt='o', color='red', mfc='none', mew=2.0, capthick=2.0, capsize=5)
        ax.set_xlabel(r'$t$ [$a$]')
        ax.set_ylabel(r'$m_\mathrm{eff}(t)$ [$a^{-1}$]')

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
        ax.plot(its, np.zeros_like(its) + fit_av, lw=5, color='blue',
                label=r'${:3.4f} \pm {:3.4f}$'.format(fit_av, fit_err))
        ax.plot(its, np.zeros_like(its) + fit_av + fit_err, ls='--', lw=1, color='blue')
        ax.plot(its, np.zeros_like(its) + fit_av - fit_err, ls='--', lw=1, color='blue')
        ax.fill_between(its, 
                        np.zeros_like(its) + fit_av + fit_err, 
                        np.zeros_like(its) + fit_av - fit_err, 
                        color='blue', alpha=0.2)
        return fit_av, fit_err
        



