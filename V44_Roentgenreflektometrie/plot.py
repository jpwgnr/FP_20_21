import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy import stats
from uncertainties import ufloat
from scipy.optimize import curve_fit

detector_phi, detector_intensity = np.genfromtxt("data/detector_scan.UXD", unpack=True)

def gauss(x, amp, mu, sigma):
    return amp/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x- mu)**2/(2*sigma**2))

detector_params, detector_pcov = curve_fit(gauss, detector_phi, detector_intensity)
detector_err = np.sqrt(np.diag(detector_pcov))

detector_phi_new = np.linspace(detector_phi[0]-0.05, detector_phi[-1]+0.05, 10000)
plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.ylabel(r"Anzahl $\cdot 10^{6}$")
plt.plot(detector_phi, detector_intensity*1e-6, ".k", label="Datenpunkte")
plt.plot(detector_phi_new, gauss(detector_phi_new, *detector_params)*1e-6, "r", label="Ausgleichskurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/detector_scan.pdf")

messung_phi, messung_intensity = np.genfromtxt("data/messung.UXD", unpack=True)

diffus_phi, diffus_intensity = np.genfromtxt("data/messung2.UXD", unpack=True)

rel_intensity = messung_intensity - diffus_intensity

plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.yscale("log")
plt.ylabel(r"Anzahl")
plt.plot(messung_phi, messung_intensity, label="Messwerte")
plt.plot(diffus_phi, diffus_intensity, label="Diffuser Scan")
plt.plot(diffus_phi, rel_intensity, label="Korrigierte Messwerte")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/messwerte_relativ.pdf")


f=  open("content/results.txt", "w")
f.write(f"Gauss amp = {detector_params[0]:.2e}+/-{detector_err[0]:.2e}, mu = {detector_params[1]:.2e}+/-{detector_err[1]:.2e}, sigma = {detector_params[2]:.2e}+/-{detector_err[2]:.2e}\n")
f.close()