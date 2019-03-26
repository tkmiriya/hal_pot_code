import numpy as np
import os, glob
import struct
from corr_baryon import Corr_2pt_Baryon

def main():
    # test code
    mxi = 0.665 # in lattice unit
    m_red = 0.5 * mxi # reduced mass
    hal = HAL_pot(m_red=m_red, result_dir='../data/sample_data', 
                  binned_nbs_dir='sample.nbs.binned', decompressed_nbs_dir='sample.nbs.decomp',
                  pot_output_dir='sample.results.pot', binned_rcorr_dir='sample.rcorr.binned',
                  it0=10, channel='xixi',
                  Ns=48, bin_size=2)
    # calculate Vc(r) for XiXi(1S0) 
    hal.calc_pot(pot_type='cen', spin='1s0')


class HAL_pot(object):
    """
    the HAL QCD potential code for V_C in 1S0 and V_C, V_T, V_eff_cen in 3S1

    Parameters:
    m_red : reduced mass [required]
    result_dir : directory including BBwave.dir.Sx.xx and correlator.PS.dir
    it0 : euclidean time (integer)
    channel  : only "nn" or "xixi" are supported
    Ns : size of lattice 
    bin_size  : size of jack-knife samples

    pot_output_dir : directory for potential (default "results.pot")
    binned_rcorr_dir : directory for binned Rcorr (default "results.rcorr.binned")
    binned_nbs_dir : directory for binned NBS (default "results.nbs.binned")
    decompressed_nbs_dir : directory for decompressed binned NBS (default "results.nbs.decomp")
    confMax : default None (use all available files in result_dir dir)
    reload_nbs : default True (if False recalculate R-correlator)
    """
    def __init__(self, m_red, result_dir='results', 
                 it0=10, channel='xixi', Ns=48, bin_size=1, 
                 binned_nbs_dir='results.nbs.binned', decompressed_nbs_dir='results.nbs.decomp', 
                 pot_output_dir='results.pot', binned_rcorr_dir='results.rcorr.binned',
                 confMax=None, reload_nbs=True):
        self.m_red = m_red # reduced mass
        self.it0 = it0
        self.channel = channel # xixi or nn
        self.Ns = Ns
        self.bin_size = bin_size
        self.result_dir = result_dir

        self.binned_nbs_dir = binned_nbs_dir
        self.decompressed_nbs_dir = decompressed_nbs_dir

        self.pot_output_dir = pot_output_dir
        self.binned_rcorr_dir = binned_rcorr_dir

        self.confMax = confMax
        self.reload_nbs = reload_nbs
        
        single_lbl = {'xixi': 'Xi_CG05_CG05', 'nn': 'proton_CG05_CG05'}[channel]
        self.C_N = Corr_2pt_Baryon(single_lbl, bin_size, result_dir, confMax=confMax)
        self.bin_num = self.C_N.bin_num

        for _dir in [binned_nbs_dir, decompressed_nbs_dir, pot_output_dir, binned_rcorr_dir]:
            if not os.path.isdir(_dir): os.mkdir(_dir)
        
    def calc_pot(self, pot_type='cen', spin='1s0'):
        """
        calculate the HAL QCD potential

        Parameters
        pot_type : cen (default) or ten
        spin : 1s0 (default) or 3s1_+1, 3s1_+0, 3s1_-1

        Returns
        pot_jk (and pot_ten_jk) : dictionary 
            jack-knife samples of (effective)central (tensor) potential

            keys 
                lap  : H0-term
                dt : d/dt-term
                dt2 : d2/dt2-term
                tot : sum of these terms

                rs : r (distance in lattice unit) 
        """
        self.spin = spin
        Rcorr_tm_jk = self.load_Rcorr(self.it0-1, spin)
        Rcorr_t_jk  = self.load_Rcorr(self.it0,   spin)
        Rcorr_tp_jk = self.load_Rcorr(self.it0+1, spin)
        
        if pot_type == 'cen':
            self.pot_jk = self.calc_t_dep_HAL(np.real(Rcorr_tm_jk[spin]),
                                              np.real(Rcorr_t_jk[spin]),
                                              np.real(Rcorr_tp_jk[spin]))
            print('>> return jackknife samples of (effective) central')
            if '3s1' in spin:
                self.save_pot(self.pot_jk, spin + '_eff')
            else:
                self.save_pot(self.pot_jk, spin + '_cen')
            return self.pot_jk
        elif pot_type == 'ten':
            if spin == '1s0': print('for tensor force use spin = 3s1')
            self.pot_jk, self.pot_ten_jk = self.calc_vc_vt(Rcorr_tm_jk, Rcorr_t_jk, Rcorr_tp_jk)
            print('>> return jackknife samples of central and tensor')
            self.save_pot(self.pot_jk, spin + '_cen')
            self.save_pot(self.pot_ten_jk, spin + '_ten')
            return self.pot_jk, self.pot_ten_jk

        else:
            print('>> select pot_type = cen or ten')
        
    def calc_t_dep_HAL(self, R_tm_jk, R_t_jk, R_tp_jk):
        Ns = self.Ns
        lap = lambda vec: - 6.0*vec + (  np.roll(vec,+1,0) + np.roll(vec,-1,0) 
                                       + np.roll(vec,+1,1) + np.roll(vec,-1,1)
                                       + np.roll(vec,+1,2) + np.roll(vec,-1,2))
        pot_jk = {}

        pot_jk['lap'] = np.array([ A1_proj(lap(R_t_jk[ibin,:,:,:])/R_t_jk[ibin,:,:,:])/(2.0*self.m_red)
            for ibin in range(self.bin_num)]).reshape(self.bin_num,Ns**3)
        pot_jk['dt'] = np.array([ - A1_proj(R_tp_jk[ibin,:,:,:] - R_tm_jk[ibin,:,:,:])/(2.0*R_t_jk[ibin,:,:,:])
            for ibin in range(self.bin_num)]).reshape(self.bin_num,Ns**3)

        pot_jk['dt2'] = np.array([
            A1_proj((R_tp_jk[ibin,:,:,:] - 2.0*R_t_jk[ibin,:,:,:] + R_tm_jk[ibin,:,:,:]) /(R_t_jk[ibin,:,:,:]))
                        /(8.0*self.m_red)
            for ibin in range(self.bin_num)]).reshape(self.bin_num,Ns**3)

        rs = np.array([np.sqrt(x**2 + y**2 + z**2) for z in range(-Ns//2+1,Ns//2+1)
                                                   for y in range(-Ns//2+1,Ns//2+1)
                                                   for x in range(-Ns//2+1,Ns//2+1)]).reshape(Ns,Ns,Ns)
        pot_jk['rs'] = np.roll(rs,(Ns//2+1,Ns//2+1,Ns//2+1),(0,1,2)).flatten()

        pot_jk['tot'] = pot_jk['lap'] + pot_jk['dt'] + pot_jk['dt2']

        return pot_jk

    def calc_vc_vt(self, R_tm_jk,  R_t_jk,  R_tp_jk):
        Ns = self.Ns
        wave_p_jk = self.get_p_phi(R_tm_jk, R_t_jk, R_tp_jk)

        wave_q_jk = self.get_q_phi(R_tm_jk, R_t_jk, R_tp_jk)

        det = wave_p_jk['wave'] * wave_q_jk['ten'] - wave_q_jk['wave'] * wave_p_jk['ten']
        pot_vc_jk = {}
        pot_vt_jk = {}
        with np.errstate(divide='ignore',invalid='ignore'):
            for t in ['lap', 'dt', 'dt2']:
                pot_vc_jk[t] = np.real(1.0/det * (wave_q_jk['ten'] * wave_p_jk[t] - wave_p_jk['ten'] * wave_q_jk[t]))
                pot_vc_jk[t][:,0] = np.real(wave_p_jk[t][:,0]/wave_p_jk['wave'][:,0])

                pot_vt_jk[t] = np.real(-1.0/det * (wave_q_jk['wave'] * wave_p_jk[t] - wave_p_jk['wave'] * wave_q_jk[t]))
                pot_vt_jk[t][:,0] = 0.0


        pot_vc_jk['tot'] = pot_vc_jk['lap'] + pot_vc_jk['dt'] + pot_vc_jk['dt2']
        pot_vt_jk['tot'] =  pot_vt_jk['lap'] + pot_vt_jk['dt'] + pot_vt_jk['dt2']

        rs = np.array([np.sqrt(x**2 + y**2 + z**2) for z in range(-Ns//2+1,Ns//2+1)
                                                   for y in range(-Ns//2+1,Ns//2+1)
                                                   for x in range(-Ns//2+1,Ns//2+1)]).reshape(Ns,Ns,Ns)
        pot_vc_jk['rs'] = np.roll(rs,(Ns//2+1,Ns//2+1,Ns//2+1),(0,1,2)).flatten()
        pot_vt_jk['rs'] = np.roll(rs,(Ns//2+1,Ns//2+1,Ns//2+1),(0,1,2)).flatten()

        return pot_vc_jk, pot_vt_jk
    
    def save_pot(self, pot_jk, spin):
        Ns = self.Ns

        with open('{}/pot_{}_{}_t{:03d}_{:03d}conf_{:03d}bin.dat'.format(self.pot_output_dir,
            spin, self.channel, self.it0, self.bin_num * self.bin_size, self.bin_size), 'w') as fout:

            print('# output potential >> ', fout.name)

            uniq_a1 = [ix + Ns*(iy + Ns*iz) for iz in range(0,Ns//2+1)
                            for iy in range(iz,Ns//2+1) for ix in range(iy,Ns//2+1)]

            fout.write('# M_red = {:4.3f}\n'.format(self.m_red))
            fout.write('# r   H0R/R  dR/R  d2R/R tot.\n')

            for r in uniq_a1:
                vtot_av, vtot_err = pot_jk['tot'][:,r].mean(axis=0), pot_jk['tot'][:,r].std(axis=0)*np.sqrt(self.bin_num - 1)
                vlap_av, vlap_err = pot_jk['lap'][:,r].mean(axis=0), pot_jk['lap'][:,r].std(axis=0)*np.sqrt(self.bin_num - 1)
                vdt_av, vdt_err = pot_jk['dt'][:,r].mean(axis=0), pot_jk['dt'][:,r].std(axis=0)*np.sqrt(self.bin_num - 1)
                vdt2_av, vdt2_err = pot_jk['dt2'][:,r].mean(axis=0), pot_jk['dt2'][:,r].std(axis=0)*np.sqrt(self.bin_num - 1)

                fout.write('{:e}   {:e} {:e}   {:e} {:e}   {:e} {:e} {:e} {:e}\n'.format(pot_jk['rs'][r], 
                    vlap_av, vlap_err, vdt_av, vdt_err, vdt2_av, vdt2_err, vtot_av, vtot_err))


    def get_p_phi(self, R_tm_jk, R_t_jk, R_tp_jk):
        Ns = self.Ns
        lap = lambda vec: - 6.0*vec + (  np.roll(vec,+1,0) + np.roll(vec,-1,0) 
                                       + np.roll(vec,+1,1) + np.roll(vec,-1,1)
                                       + np.roll(vec,+1,2) + np.roll(vec,-1,2))
        wave_p_jk = {}
        wave_p_jk['wave'] = np.array([A1_proj(R_t_jk[self.spin][ibin,:,:,:]) 
            for ibin in range(self.bin_num)]).reshape(self.bin_num, Ns**3)

        wave_p_jk['lap'] = np.array([ A1_proj(lap(R_t_jk[self.spin][ibin,:,:,:]))/(2.0*self.m_red) 
            for ibin in range(self.bin_num)]).reshape(self.bin_num, Ns**3)

        wave_p_jk['dt']  = np.array([ - A1_proj(0.5*(R_tp_jk[self.spin][ibin,:,:,:] - R_tm_jk[self.spin][ibin,:,:,:])) 
            for ibin in range(self.bin_num)]).reshape(self.bin_num,Ns**3)

        wave_p_jk['dt2'] = np.array([ A1_proj((R_tp_jk[self.spin][ibin,:,:,:] - 2.0*R_t_jk[self.spin][ibin,:,:,:] + R_tm_jk[self.spin][ibin,:,:,:])) /(8.0*self.m_red)
            for ibin in range(self.bin_num)]).reshape(self.bin_num,Ns**3)

        wave_p_jk['ten'] = np.array([A1_proj(R_t_jk['ten_' + self.spin][ibin,:,:,:]) for ibin in range(self.bin_num)]).reshape(self.bin_num, Ns**3)

        return wave_p_jk

    def get_q_phi(self, R_tm_jk, R_t_jk, R_tp_jk):
        Ns = self.Ns
        unit = lambda vec: vec
        lap = lambda vec: - 6.0*vec + (  np.roll(vec,+1,0) + np.roll(vec,-1,0) 
                                       + np.roll(vec,+1,1) + np.roll(vec,-1,1)
                                       + np.roll(vec,+1,2) + np.roll(vec,-1,2))
        wave_q_jk = {}

        for kind, R_jk in zip(['wave', 'wave_p', 'wave_m'], [R_t_jk, R_tp_jk, R_tm_jk]):
            wave_q_jk[kind] = self.sum_d_waves(R_jk, unit)

        wave_q_jk['ten'] = self.sum_d_waves(R_t_jk, unit, ten='ten_').reshape(self.bin_num, Ns**3)

        wave_q_jk['lap'] = self.sum_d_waves(R_t_jk, lap).reshape(self.bin_num, Ns**3)/(2.0*self.m_red)
        
        wave_q_jk['dt'] = np.array([ -(0.5*(wave_q_jk['wave_p'][ibin,:,:,:] - wave_q_jk['wave_m'][ibin,:,:,:])) 
            for ibin in range(self.bin_num)]).reshape(self.bin_num,Ns**3)

        wave_q_jk['dt2'] = np.array([ ((wave_q_jk['wave_p'][ibin,:,:,:] - 2.0*wave_q_jk['wave'][ibin,:,:,:] + wave_q_jk['wave_m'][ibin,:,:,:])) /(8.0*self.m_red)
            for ibin in range(self.bin_num)]).reshape(self.bin_num,Ns**3)

        wave_q_jk['wave'] = wave_q_jk['wave'].reshape(self.bin_num, Ns**3)

        return wave_q_jk



    def sum_d_waves(self, Rcorr, op, ten=''):
        Ns = self.Ns
        Ylms = spherical_harmonics(Ns)
        jz0 = int(self.spin.split('_')[1]) # +1, +0, -1
        label = lambda jz: '3d{:+d}_y2{:+d}'.format(jz0, jz0 + jz)


        tmp_q = [np.array(
                [A1_subt(op(Rcorr[ten + label(m)][ibin,:,:,:]))
                * np.conjugate(Ylms[jz0+m]) for ibin in range(self.bin_num)]).reshape(self.bin_num, Ns, Ns, Ns)
                for m in [+1, 0, -1]]

        coeff = {'3s1_+1': [np.sqrt(6), - np.sqrt(3), 1.0],
                 '3s1_+0': [np.sqrt(3), -2.0, np.sqrt(3)],
                 '3s1_-1': [1.0, - np.sqrt(3), np.sqrt(6)]}[self.spin]
        return (coeff[0] * tmp_q[0] + coeff[1] * tmp_q[1]  + coeff[2] * tmp_q[2]) / np.sqrt(10)



        
    def load_Rcorr(self, it, spin):
        ch_index = {'nn': '0.00', 'xixi': '4.00'}[self.channel]
        mode_list = {'1s0': ['1s0'],
             '3s1_+1': ['3s1_+1', 'ten_3s1_+1', '3d+1_y2+2', 'ten_3d+1_y2+2', 
                        '3d+1_y2+1', 'ten_3d+1_y2+1', '3d+1_y2+0', 'ten_3d+1_y2+0'],
             '3s1_+0': ['3s1_+0', 'ten_3s1_+0', '3d+0_y2+1', 'ten_3d+0_y2+1', 
                        '3d+0_y2+0', 'ten_3d+0_y2+0', '3d+0_y2-1', 'ten_3d+0_y2-1'],
             '3s1_-1': ['3s1_-1', 'ten_3s1_-1', '3d-1_y2+0', 'ten_3d-1_y2+0', 
                        '3d-1_y2-1', 'ten_3d-1_y2-1', '3d-1_y2-2', 'ten_3d-1_y2-2'] }[spin]

        f_name = lambda ch: '{}/Rcorr_{}_{}_t{:03d}_{:d}bin_{:d}conf.dat'.format(self.binned_rcorr_dir,
                        ch, self.channel, it, self.bin_num, self.bin_num * self.bin_size)

        fsize = self.Ns**3*self.bin_num * 2
        Rcorr_jk = {}

        if os.path.isfile(f_name(mode_list[0])) and self.reload_nbs:
            for ch in mode_list:
                print('# load Rcorr ', f_name(ch))
                with open(f_name(ch), 'rb') as infile: 
                    tmpr = np.array(struct.unpack('{:d}d'.format(fsize), infile.read(8*fsize))).reshape(self.bin_num,self.Ns,self.Ns,self.Ns,2)
                    Rcorr_jk[ch] = tmpr[:,:,:,:,0] + tmpr[:,:,:,:,1]*1j
        else:
            print('# calc. Rcorr ', f_name(mode_list[0]))

            flist = glob.glob(f'{self.decompressed_nbs_dir}/NBSwave.S{ch_index}.t{it:03d}.binned.*.decomp.dat')
            if len(flist) == 0: # call binning NBS
                print('# binning and decompress NBS')
                nbs_bin = NBS_binning(self.channel, it, result_dir=self.result_dir, 
                                      binned_nbs_dir=self.binned_nbs_dir, decompressed_nbs_dir=self.decompressed_nbs_dir,
                                      bin_size=self.bin_size, confMax=self.confMax)

            nbs = NBSwave(Ns=self.Ns, channel=self.channel, spin=spin, it=it, decompressed_nbs_dir=self.decompressed_nbs_dir)
            
            Rcorr_jk = {}
            for ch in mode_list:
                if 'ten' in ch:
                    Rcorr_jk[ch] = np.array([nbs.wave_wf_ten_jk[ch][ibin,:,:,:]/
                                 self.C_N.corr_jk[ibin,it]**2 for ibin in range(self.bin_num)])
                else:
                    Rcorr_jk[ch] = np.array([nbs.wave_wf_jk[ch][ibin,:,:,:]/
                                 self.C_N.corr_jk[ibin,it]**2 for ibin in range(self.bin_num)])
                
                with open(f_name(ch), 'wb') as fout: 
                    print(f'# save {f_name(ch)}')
                    fout.write(bytearray(Rcorr_jk[ch].flatten()))
        return Rcorr_jk
        

class NBSwave(object):
    def __init__(self, Ns, channel, spin, it, decompressed_nbs_dir):
        self.Ns = Ns
        self.spin = spin
        ch_index = {'nn': '0.00', 'xixi': '4.00'}[channel]
        
        flist = glob.glob(f'{decompressed_nbs_dir}/NBSwave.S{ch_index}.t{it:03d}.binned.*.decomp.dat')
        flist.sort()
        
        waves_bin = []
        for fname in flist:
            print('# load ', fname)
            waves_bin.append(self.load_wavefunc(fname))
        waves_bin = np.array(waves_bin)
        self.bin_num = waves_bin.shape[0]
        wave_jk = (np.sum(waves_bin[:,:,:,:,:,:], axis=0)
                - waves_bin)/float(self.bin_num - 1)
        wave_ten_jk = np.array([self.mult_tensor(wave_jk[ibin,:,:,:,:,:]) 
                                        for ibin in range(self.bin_num)])
        self.calc_wave_func(wave_jk, wave_ten_jk)

    def load_wavefunc(self, fname):
        ntotal = self.Ns**3 * 2 * 2*2 * 2*2
        with open(fname, 'rb') as infile:
            tmpw = np.array(struct.unpack('>{:d}d'.format(ntotal), infile.read(8*ntotal)))
            wave_spin_proj = self.spin_projection(tmpw)
        # parity proj.
        tmpw = wave_spin_proj.reshape(self.Ns,self.Ns,self.Ns,2,2)
        wave_proj = 0.5 * (tmpw[:,:,:,:,:] + np.roll(tmpw[::-1,::-1,::-1,:,:],(1,1,1),(0,1,2)))

        return wave_proj.reshape(self.Ns, self.Ns, self.Ns,2,2)

    def spin_projection(self, wave_in):
        wave_fw = wave_in.reshape(2,2,self.Ns**3,2,2,2)
        wave_fw = wave_fw[:,:,:,:,:,0] + wave_fw[:,:,:,:,:,1]*1j

        if self.spin == '1s0':
            return 1.0/np.sqrt(2.0) * (wave_fw[1,0,:,:,:] - wave_fw[0,1,:,:,:])
        elif self.spin == '3s1_+1':
            return wave_fw[0,0,:,:,:]
        elif self.spin == '3s1_+0':
            return 1.0/np.sqrt(2.0) * (wave_fw[1,0,:,:,:] + wave_fw[0,1,:,:,:])
        elif self.spin == '3s1_-1':
            return wave_fw[1,1,:,:,:]
        
    def mult_tensor(self, wave_in):
        """
        ! S12 = 3/r^2 (sigma1.r)(sigma2.r) - (sigma1).(sigma2)

        ! S12 * ( A, B  ) = \sqrt(pi/5) * ( A', B' )
        !       ( B, C  )                 ( B', D' )

        ! S12 * ( 0, B  ) = 0
        !       (-B, 0  )
        """

        A = wave_in[:,:,:,0,0]
        B = 0.5 * (wave_in[:,:,:,0,1] + wave_in[:,:,:,1,0])
        C = wave_in[:,:,:,1,1]

        Ylms = spherical_harmonics(self.Ns)

        wave_ten = np.zeros((self.Ns,self.Ns,self.Ns,2,2), dtype=complex)

        wave_ten[:,:,:,0,0] = np.sqrt(np.pi/5.0) * (
                  4.0*Ylms[0]*A 
                + 4.0*np.sqrt(6.0)*Ylms[-1]*B
                + 4.0*np.sqrt(6.0)*Ylms[-2]*C)

        wave_ten[:,:,:,0,1] = np.sqrt(np.pi/5.0) * (
                - 2.0*np.sqrt(6.0)*Ylms[1]*A 
                - 8.0*Ylms[0]*B
                - 2.0*np.sqrt(6.0)*Ylms[-1]*C)

        wave_ten[:,:,:,1,0] = wave_ten[:,:,:,0,1]

        wave_ten[:,:,:,1,1] = np.sqrt(np.pi/5.0) * (
                  4.0*np.sqrt(6.0)*Ylms[2]*A 
                + 4.0*np.sqrt(6.0)*Ylms[1]*B
                + 4.0*Ylms[0]*C)
        return wave_ten
    
    def calc_wave_func(self, wave_jk, wave_ten_jk):
        self.wave_wf_jk = {}
        self.wave_wf_ten_jk = {}

        if self.spin == '1s0':
            self.wave_wf_jk[self.spin] = 1.0/np.sqrt(2.0) * np.array(
                    [A1_proj(wave_jk[ibin,:,:,:,1,0] - wave_jk[ibin,:,:,:,0,1]) 
                        for ibin in range(self.bin_num)])
            self.wave_wf_ten_jk[self.spin] = 1.0/np.sqrt(2.0) * np.array(
                    [A1_proj(wave_ten_jk[ibin,:,:,:,1,0] - wave_ten_jk[ibin,:,:,:,0,1])
                        for ibin in range(self.bin_num)])

        elif self.spin == '3s1_+1':
            self.wave_wf_jk[self.spin] = np.array(
                    [A1_proj(wave_jk[ibin,:,:,:,0,0]) for ibin in range(self.bin_num)])
            self.wave_wf_ten_jk['ten_' + self.spin] = np.array(
                    [A1_proj(wave_ten_jk[ibin,:,:,:,0,0]) for ibin in range(self.bin_num)])

        elif self.spin == '3s1_+0':
            self.wave_wf_jk[self.spin] = 1.0/np.sqrt(2.0) * np.array(
                    [A1_proj(wave_jk[ibin,:,:,:,0,1] + wave_jk[ibin,:,:,:,1,0]) 
                        for ibin in range(self.bin_num)])
            self.wave_wf_ten_jk['ten_' + self.spin] = 1.0/np.sqrt(2.0) * np.array(
                    [A1_proj(wave_ten_jk[ibin,:,:,:,0,1] + wave_ten_jk[ibin,:,:,:,1,0]) 
                        for ibin in range(self.bin_num)])

        elif self.spin == '3s1_-1':
            self.wave_wf_jk[self.spin] = np.array(
                [A1_proj(wave_jk[ibin,:,:,:,1,1]) for ibin in range(self.bin_num)])
            self.wave_wf_ten_jk['ten_' + self.spin] = np.array(
                [A1_proj(wave_ten_jk[ibin,:,:,:,1,1]) for ibin in range(self.bin_num)])

        if '3s1' in self.spin:
            jz0 = int(self.spin.split('_')[1])
            label = lambda jz: '3d{:+d}_y2{:+d}'.format(jz0, jz0 + jz)

            self.wave_wf_jk[label(1)] = np.array([A1_subt(wave_jk[ibin,:,:,:,1,1]) for ibin in range(self.bin_num)])
            self.wave_wf_ten_jk['ten_' + label(1)] = np.array([A1_subt(wave_ten_jk[ibin,:,:,:,1,1]) for ibin in range(self.bin_num)])

            self.wave_wf_jk[label(0)] = 1.0/np.sqrt(2.0) * np.array(
                    [A1_subt(wave_jk[ibin,:,:,:,1,0] + wave_jk[ibin,:,:,:,0,1]) for ibin in range(self.bin_num)])
            self.wave_wf_ten_jk['ten_' + label(0)] = 1.0/np.sqrt(2.0) * np.array(
                    [A1_subt(wave_ten_jk[ibin,:,:,:,1,0] + wave_ten_jk[ibin,:,:,:,0,1]) for ibin in range(self.bin_num)])

            self.wave_wf_jk[label(-1)] = np.array([A1_subt(wave_jk[ibin,:,:,:,0,0]) for ibin in range(self.bin_num)])
            self.wave_wf_ten_jk['ten_' + label(-1)] = np.array([A1_subt(wave_ten_jk[ibin,:,:,:,0,0]) for ibin in range(self.bin_num)])
            

def spherical_harmonics(Ns):

    tmp_ind = np.array([[ix, iy, iz]
        for iz in np.arange(-Ns//2+1,Ns//2+1)
        for iy in np.arange(-Ns//2+1,Ns//2+1)
        for ix in np.arange(-Ns//2+1,Ns//2+1)]).reshape(Ns,Ns,Ns,3)
    xyz = np.roll(np.roll(np.roll(tmp_ind,Ns//2+1,0),
        Ns//2+1,1),Ns//2+1,2).reshape(Ns**3,3)
    # x = 0, 1, ..., N/2, -N/2+1, ...

    def spherical_harmonics_Dwave(m):

        Ylm = np.zeros(Ns**3,dtype=complex)

        rsq = xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2
        
        if m == -2:
            Ylm[1:] = 1.0/4.0 * ( np.sqrt(7.5/np.pi)
                              * (xyz[1:,0]**2 
                                 - 2.0j*xyz[1:,0]*xyz[1:,1]
                                 - xyz[1:,1]**2) / rsq[1:])
        elif m == -1:
            Ylm[1:] = 1.0/2.0 * ( np.sqrt(7.5/np.pi)
                              * (xyz[1:,0]*xyz[1:,2] 
                                 - 1.0j*xyz[1:,1]*xyz[1:,2])
                                 / rsq[1:])
        elif m == 0:
            Ylm[1:] = 1.0/4.0 * ( np.sqrt(5.0/np.pi)
                                * (-xyz[1:,0]**2 -xyz[1:,1]**2
                                  +2.0*xyz[1:,2]**2) / rsq[1:])
        elif m == 1:
            Ylm[1:] = -1.0/2.0 * ( np.sqrt(7.5/np.pi)
                                 * (xyz[1:,0]*xyz[1:,2]
                                   + 1.0j*xyz[1:,1]*xyz[1:,2])
                                 / rsq[1:])
        elif m == 2:
            Ylm[1:] = 1.0/4.0 * ( np.sqrt(7.5/np.pi)
                                * (xyz[1:,0]**2
                                  + 2.0j*xyz[1:,0]*xyz[1:,1]
                                  - xyz[1:,1]**2) / rsq[1:])

        return Ylm.reshape(Ns,Ns,Ns)

    Ylms = {m: spherical_harmonics_Dwave(m)
           for m in [-2,-1,0,1,2]}
    return Ylms

def A1_subt(wave_in):
    return wave_in -  A1_proj(wave_in)

def A1_proj(wave_in):
    wave_tmp0 = wave_in
    wave_tmp1 = (wave_tmp0[:,:,:] + np.roll(wave_tmp0,-1,0)[::-1,:,:]
                + np.roll(wave_tmp0,-1,1)[:,::-1,:]
                + np.roll(wave_tmp0,-1,2)[:,:,::-1]
                + np.roll(np.roll(wave_tmp0,-1,0),-1,1)[::-1,::-1,:]
                + np.roll(np.roll(wave_tmp0,-1,1),-1,2)[:,::-1,::-1]
                + np.roll(np.roll(wave_tmp0,-1,2),-1,0)[::-1,:,::-1]
                + np.roll(np.roll(np.roll(wave_tmp0,-1,0),-1,1),-1,2)[::-1,::-1,::-1])/8.0
    wave_tmp2 = (wave_tmp1 
                + np.swapaxes(wave_tmp1,0,1)
                + np.swapaxes(wave_tmp1,1,2)
                + np.swapaxes(wave_tmp1,2,0)
                + np.swapaxes(np.swapaxes(wave_tmp1,0,1),1,2)
                + np.swapaxes(np.swapaxes(wave_tmp1,0,2),2,1))/6.0e0

    return wave_tmp2


class NBS_binning(object):
    def __init__(self, channel, it, result_dir='results', 
                 binned_nbs_dir='results.nbs.binned', decompressed_nbs_dir='results.nbs.decomp',
                 bin_size=1, confMax = None, decompress = True):
        ch_index = {'nn': '0.00', 'xixi': '4.00'}[channel]
        print(f'# binning NBS S{ch_index} t = {it}')
        dir_list = os.listdir(result_dir + f'/BBwave.dir.S{ch_index}/')
        dir_list.sort()

        flist = []
        ibin = 0
        for dir_name in dir_list[:confMax]:
            files =  glob.glob(result_dir + f'/BBwave.dir.S{ch_index}/'
                              + dir_name + f'/NBSwave.+{it:03d}*.NUC_CG05')
            for fname in files:
                flist.append(fname)
                if len(flist) == bin_size:
                    print(f'--- binning NBS wavefunc. --- {ibin:03d}')
                    binned_file = f'{binned_nbs_dir}/NBSwave.S{ch_index}.t{it:03d}.binned.{ibin:03d}.dat'
                    binned_decomp_file = f'{decompressed_nbs_dir}/NBSwave.S{ch_index}.t{it:03d}.binned.{ibin:03d}.decomp.dat'
                    self.binning_wavefunc(flist, binned_file)
                    if decompress == True and os.path.exists('./PH1.compress48'):
                        print(f'>> decompress {binned_file}')
                        os.system(f'./PH1.compress48 -d {binned_file} {binned_decomp_file}')
                    if decompress == True and not os.path.exists('./PH1.compress48'):
                        print('>> missing PH1.compress48')

                    ibin += 1
                    flist = [] # reset
        print(f'>> total {ibin} binned NBS files')
        
    def binning_wavefunc(self, flist, fname):
        with open(fname, 'wb') as out_binned:
            infiles = [open(fname, 'rb') for fname in flist]

            for iter in [1,2,3]:
                hc = struct.unpack('<i', infiles[0].read(4))[0]//4
                struct.unpack('<{}i'.format(hc+1), infiles[0].read(4*(hc+1)))
                tmp_data = [struct.unpack('<{}i'.format(hc+2), infile.read(4*(hc+2))) for infile in infiles[1:]]
                data = np.array(tmp_data[0])
                out_binned.write(struct.pack('<{:d}i'.format(hc+2), *data))

            hc = struct.unpack('<i', infiles[0].read(4))[0]//8
            tmp = [struct.unpack('<i', infile.read(4)) for infile in infiles[1:]]

            out_binned.write(struct.pack('<i', (hc*8)))

            data = np.array([struct.unpack('<{:d}d'.format(hc), infile.read(8*hc))
                            for infile in infiles]).mean(axis=0)

            out_binned.write(struct.pack('<{:d}d'.format(hc), *data))
            out_binned.write(struct.pack('<i', (hc*8)))

if __name__ == '__main__':
    main()
