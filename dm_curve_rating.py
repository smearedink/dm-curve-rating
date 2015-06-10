import numpy as np
import pylab as plt
from matplotlib.cm import Greys
from prepfold import pfd
import psr_utils
from scipy.optimize import leastsq
from scipy import stats
from glob import glob

make_plots = True

#pfd_files = glob("/homes/borg/madsense/PALFA/dm_curve_rating/larger_set/pulsars/*.pfd")
pfd_files = glob("/homes/borg/madsense/PALFA/dm_curve_rating/larger_set/rfi/*.pfd")

cands = []
for pfd_file in pfd_files:
    cands.append(pfd(pfd_file))

"""
cands.append(pfd("0540+32_known.pfd"))
cands.append(pfd("1903+0327_known.pfd"))
cands.append(pfd("1904+0412_known.pfd"))
cands.append(pfd("1913+10_known.pfd"))
cands.append(pfd("1937+21_known.pfd"))
cands.append(pfd("rfi1.pfd"))
cands.append(pfd("rfi2.pfd"))
cands.append(pfd("rfi3.pfd"))
cands.append(pfd("rfi4.pfd"))
cands.append(pfd("rfi5.pfd"))
cands.append(pfd("rfi6.pfd"))
cands.append(pfd("1851+0232_discovery.pfd"))
cands.append(pfd("1851+0242_discovery.pfd"))
cands.append(pfd("1906+0725_discovery.pfd"))
cands.append(pfd("1913+0617_discovery.pfd"))
cands.append(pfd("0631+1036_harmonic_61.pfd"))
cands.append(pfd("1916+14_harmonic_10.3.pfd"))
cands.append(pfd("1957+2831_harmonic_7.2.pfd"))
cands.append(pfd("noise1.pfd"))
cands.append(pfd("noise2.pfd"))
cands.append(pfd("noise3.pfd"))
cands.append(pfd("noise4.pfd"))
cands.append(pfd("noise5.pfd"))
"""

for cand in cands:
    cand.dedisperse(interp=1)

# based on the plot_chi2_vs_DM code in the prepfold.pfd class
def dm_curve_check(cand, spec_index=0.):
    # Sum the profiles in time
    profs = cand.profs.sum(0)

    ### Generate simulated profiles ###

    # prof_avg: median profile value per subint per subband
    #  Sum over subint axis to get median per subband
    prof_avg = cand.stats[:,:,4].sum(0)
    # prof_var: variance of profile per subint per subband
    #  Sum over subint axis to get variance per subband
    prof_var = cand.stats[:,:,5].sum(0)
    # The standard deviation in each subband is proportional to the median
    # value of that subband.  Here we scale all subbands to equal levels.
    scaled_vars = prof_var / prof_avg**2
    scaled_profs = (profs.T / prof_avg).T - 1.
    # The mean profile (after scaling) will be our "clean" profile--hardly
    # true in most cases, but we pretend it's noiseless for what follows
    scaled_mean_prof = scaled_profs.mean(0)
    # Extend this "clean" profile across the number of subbands
    sim_profs_clean = np.tile(scaled_mean_prof,\
        scaled_profs.shape[0]).reshape(scaled_profs.shape)
    # Scale these subbands according to the input spectral index
    spec_mult = (cand.subfreqs/cand.subfreqs[0])**spec_index
    spec_mult /= spec_mult.mean()
    sim_profs_spec = (sim_profs_clean.T*spec_mult).T
    # For consistency, set a seed for generating noise
    np.random.seed(1967)
    # Add white noise that matches the variance of the real subbands
    # on a subband by subband basis
    noise = np.random.normal(scale=np.sqrt(scaled_vars),\
        size=scaled_profs.T.shape).T
    # sim_profs_noisy is the simulated equivalent of scaled_profs
    sim_profs_noisy = sim_profs_spec + noise
    # sim_profs_final is the simulated equivalent of profs
    sim_profs_final = ((sim_profs_noisy + 1.).T * prof_avg).T
    
    # Make a copy that can be manipulated since we want to return
    # sim_profs_final at the end
    sim_profs = sim_profs_final.copy()

    # The rest of this is essentially code from the prepfold.pfd class
    # in which we loop over DM values and see how strong a signal we
    # get by dedispersing at each of these values
    DMs = np.linspace(cand.dms[0], cand.dms[-1], len(cand.dms)/10)
    chis = np.zeros_like(DMs)
    sim_chis = np.zeros_like(DMs)
    subdelays_bins = cand.subdelays_bins.copy()
    for ii, DM in enumerate(DMs):
        subdelays = psr_utils.delay_from_DM(DM, cand.barysubfreqs)
        hifreqdelay = subdelays[-1]
        subdelays = subdelays - hifreqdelay
        delaybins = subdelays*cand.binspersec - subdelays_bins
        new_subdelays_bins = np.floor(delaybins+0.5)
        for jj in range(cand.nsub):
            profs[jj] = psr_utils.rotate(profs[jj], int(new_subdelays_bins[jj]))
            sim_profs[jj] = psr_utils.rotate(sim_profs[jj],\
                int(new_subdelays_bins[jj]))
        subdelays_bins += new_subdelays_bins
        # The set of reduced chi2s like those in the prepfold plot
        # (should be the same if the same DMs are used)
        chis[ii] = cand.calc_redchi2(prof=profs.sum(0), avg=cand.avgprof)
        # The same thing but for our "simulated" data
        sim_chis[ii] = cand.calc_redchi2(prof=sim_profs.sum(0), avg=cand.avgprof)
    return DMs, chis, sim_chis, sim_profs_final

def dm_curve_diff(spec_index, cand):
    dms, dmcurve, fdmcurve, sim_profs = dm_curve_check(cand, spec_index)
    # I'm not quite sure whether to use the variance of both curves or just
    # one of them...

    # This variance comes from basic error propagation through the
    # pfd.calc_redchi2 function
    var1 = 4./cand.DOFcor * dmcurve

    #var2 = 4./cand.DOFcor * fdmcurve

    return (dmcurve - fdmcurve) / np.sqrt(var1)

def find_spec_index(cand):
    """
    Run a least squares fit to find the "spectral index" that gives the
    best match between real and simulated DM curves
    """
    fit = leastsq(dm_curve_diff, 0, cand, full_output=True)
    return fit[0][0]

def calc_prepfold_sigma(cand):
    """
    This won't work unless a .bestprof file is present (they can be
    regenerated using show_pfd)
    """
    try:
        red_chi2 = cand.bestprof.chi_sqr
        dof = cand.proflen - 1
        return -stats.norm.ppf(stats.chi2.sf(red_chi2*dof, dof))
    except:
        print "Error: bestprof file not available"
        return -1.

dmvals = []
dmcurves = []
fdmcurves = []
spec_inds = []
pf_sigmas = []
final_values = []

ncands = len(cands)

for candnum, cand in enumerate(cands):
    spec_index = find_spec_index(cand)
    #spec_index = 0.

    x = np.linspace(0, 2, len(cand.profs.sum(0).sum(0))*2, endpoint=False)
    dms, dmcurve, fdmcurve, sim_profs = dm_curve_check(cand, spec_index)
    prof_scaled = (cand.profs.sum(0).T / np.median(cand.profs.sum(0).T, axis=0)).T
    sim_prof_scaled = (sim_profs.T / np.median(cand.profs.sum(0).T, axis=0)).T
    vmin = np.min((prof_scaled, sim_prof_scaled))
    vmax = np.max((prof_scaled, sim_prof_scaled))

    # subtract 1 from len(dms) due to spectral index... not sure if this is
    # exactly right
    final_value = np.sum(dm_curve_diff(spec_index, cand)**2) / (len(dms)-1)

    dmvals.append(dms)
    dmcurves.append(dmcurve)
    fdmcurves.append(fdmcurve)
    spec_inds.append(spec_index)
    pf_sigmas.append(calc_prepfold_sigma(cand))
    final_values.append(final_value)

    print "%d of %d:" % (candnum+1, ncands)
    print "%60s %.3f" % (cand.pfd_filename.split("/")[-1], final_value)

    if make_plots:
        fig = plt.figure(figsize=(10, 10))
        fig.text(0.5, 0.975, "%s" % (cand.pfd_filename.split("/")[-1]),\
            horizontalalignment='center', verticalalignment='top')

        # real phase-vs-freq plot
        ax1 = fig.add_subplot(221)
        ax1.pcolormesh(x, cand.subfreqs, np.tile(prof_scaled, 2),\
            cmap=Greys, vmin=vmin, vmax=vmax)
        ax1.set_ylim(cand.subfreqs[0], cand.subfreqs[-1])
        ax1.set_title("Real (%.1f ms)" % (cand.bary_p1*1000.))
        ax1.set_xlabel("phase")
        ax1.set_ylabel("frequency (MHz)")
        
        # simulated phase-vs-freq plot
        ax2 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
        ax2.pcolormesh(x, cand.subfreqs, np.tile(sim_prof_scaled, 2),\
            cmap=Greys, vmin=vmin, vmax=vmax)
        ax1.set_ylim(cand.subfreqs[0], cand.subfreqs[-1])
        ax2.set_title("Simulated (\"spectral index\" %.2f)" % spec_index)
        ax2.set_xlabel("phase")
        ax2.set_ylabel("frequency (MHz)")

        # both DM curves
        ax3 = fig.add_subplot(222)
        ax3.plot(dms, dmcurve, lw=2, c='0.75', label='obs')
        ax3.plot(dms, fdmcurve, c='black', label='sim')
        ax3.set_xlim(dms[0], dms[-1])
        ax3.set_title("Simulated and observed DM curves")
        ax3.set_xlabel("DM")
        ax3.set_ylabel("reduced chi-squared")
        ax3.legend()

        # difference of DM curves
        ax4 = fig.add_subplot(224, sharex=ax3)
        #err = np.sqrt(4./cand.DOFcor * (dmcurve + fdmcurve))
        err = np.sqrt(4./cand.DOFcor * dmcurve)
        ax4.errorbar(dms, dmcurve-fdmcurve, err, fmt='o', capsize=0,\
            ecolor='black', color='black')
        ax4.set_xlim(dms[0], dms[-1])
        #ax4.set_title("Observed minus simulated")
        ax4.set_title("Final reduced chi-squared: %.2f" % final_value)
        ax4.set_xlabel("DM")
        ax4.set_ylabel("difference")
        

        #fig.tight_layout()

        plt.savefig(cand.pfd_filename + ".dmplot.png")

        plt.close('all')
