import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy import stats
from uncertainties import ufloat
from scipy.optimize import curve_fit

# z_scan

z, z_intensity = np.genfromtxt("data/z_scan.UXD", unpack=True)

plt.figure()
plt.xlabel(r"$z$ / \si{\milli\meter}")
plt.ylabel(r"Anzahl Events $\cdot 10^{6}$")
plt.axvline(x = -0.36, linestyle="--", color="r", label=r"Strahlbreite $d$")
plt.axvline(x = -0.12, linestyle="--", color="r")
plt.plot(z, z_intensity*1e-6, label="Datenpunkte")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/z_scan.pdf")


#Detektor Scan

# read data
detector_phi, detector_intensity = np.genfromtxt("data/detector_scan.UXD", unpack=True)

# fitting function x = theta in degree
def gauss(x, amp, mu, sigma):
    return amp/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x- mu)**2/(2*sigma**2))

# fit and errors
detector_params, detector_pcov = curve_fit(gauss, detector_phi, detector_intensity)
detector_err = np.sqrt(np.diag(detector_pcov))

# Plot
detector_phi_new = np.linspace(detector_phi[0]-0.05, detector_phi[-1]+0.05, 10000)
plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.ylabel(r"Anzahl Events $\cdot 10^{6}$")
plt.plot(detector_phi, detector_intensity*1e-6, ".", label="Datenpunkte")
plt.plot(detector_phi_new, gauss(detector_phi_new, *detector_params)*1e-6, label="Ausgleichskurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/detector_scan.pdf")

# Messung und Diffusionskorrektur
#read data
messung_phi, messung_intensity = np.genfromtxt("data/messung.UXD", unpack=True)

diffus_phi, diffus_intensity = np.genfromtxt("data/messung2.UXD", unpack=True)

#relative data
rel_intensity = messung_intensity - diffus_intensity
rel_phi = diffus_phi

#Plot
plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.yscale("log")
plt.ylabel(r"Anzahl Events")
plt.plot(messung_phi, messung_intensity, label="Messwerte")
plt.plot(diffus_phi, diffus_intensity, label="Diffuser Scan")
plt.plot(rel_phi, rel_intensity, label="Korrigierte Messwerte")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/messwerte_relativ.pdf")

# rel_test = rel_intensity[7::]
# rel = rel_phi[7::]
# print(rel[rel_test == max(rel_test)])

# Normierung und Silizium Reinkurve

alpha_crit = 0.195
alpha_si = 0.223
r_lambda = 1.54e-10
k = 2*np.pi / r_lambda
n = 1 - 7.6e-6 + 1.54e-8j*141/(4*np.pi)

# Ideale Kurve
def ideal(alpha):
    return (np.abs((k * np.sin(alpha)- k*np.sqrt(n**2-np.cos(alpha)**2))/(k * np.sin(alpha)+ k*np.sqrt(n**2-np.cos(alpha)**2))))**2

# Normierung
rel_intensity = rel_intensity/rel_intensity[rel_phi == alpha_crit]

# Plot
plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.yscale("log")
plt.ylabel(r"Normierte Anzahl Events (a.u.)")
plt.plot(rel_phi, rel_intensity, label="Normierte Werte")
plt.plot(rel_phi, ideal(np.deg2rad(rel_phi)), label="Ideale Siliziumoberfläche")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/messwerte_norm.pdf")

# Korrektur durch Geometriefaktor
dreieck_phi, dreieck_intensity = np.genfromtxt("data/rocking_scan.UXD", unpack=True)

# Plot
plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
#plt.yscale("log")
plt.ylabel(r"Anzahl Events $\cdot 10^{6}")
plt.plot(dreieck_phi, dreieck_intensity*1e-6, ".", label="Messwerte")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/dreieck.pdf")

# Bestimmung G-Faktor
D = 0.0215
a_g = 0.70
d_0 = np.sin(np.deg2rad(a_g))*D
G_faktor = D/d_0

# Liste der jeweiligen G-Faktoren
faktor = []
for i in rel_phi:
    if i < a_g:
        faktor = np.append(faktor, G_faktor*np.sin(np.deg2rad(i)))
    else:
        faktor = np.append(faktor, 1)

# Anwendung G-Faktor und erneute Normalisierung
faktor[0]= 0.001
rel_intensity = rel_intensity/faktor
rel_intensity = rel_intensity/rel_intensity[rel_phi == alpha_crit]

# Bestimmung der Dicke 
# Minima einlesen
np.savetxt("data/normalised_and_corrected.txt",np.array([rel_phi, rel_intensity]).T, fmt="%.3e")
num, min_phi, min_intensity = np.genfromtxt("data/minima.txt", unpack=True)

# Wellenvektorüberträge bestimmen
q_z = 2*k*np.sin(np.deg2rad(min_phi))
del_q = []

for i in range(len(q_z)-1):
    del_q = np.append(del_q, q_z[i+1]-q_z[i])

dicken = 2*np.pi/del_q
dicke_mean = dicken.mean()
dicke_err = dicken.std(ddof=1)/np.sqrt(len(dicken))
dicke = ufloat(dicke_mean, dicke_err)

np.savetxt("data/neuewinkel.txt",np.array([min_phi, q_z*1e-8]).T, fmt="%.3f")

np.savetxt("data/neuedicken.txt",np.array([del_q*1e-8, dicken*1e10]).T, fmt="%.3f")
# Parratt 

z= 8.7e-8 
delta1=5.1e-6
delta2=5.7e-6
beta1=400 
beta2=500
sigma1=5.5e-10
sigma2=8e-10
n1=1


const_p = 1.54e-8j/(4*np.pi)
normal_p = 1.54e-8/(4*np.pi)
b1 = beta1*normal_p 
b2 = beta2*normal_p

def parat(a,delta1,delta2,beta1,beta2,sigma1,sigma2):
    return np.abs(
    (((k*np.sqrt(n1**2-np.cos(a)**2))-(k*np.sqrt((1- delta1 -beta1 * const_p)**2-np.cos(a)**2)))/ # r_{0,1} Zähler
    ((k*np.sqrt(n1**2-np.cos(a)**2))+(k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2)))       # r_{0,1} Nenner
    *np.exp(-2*(k*np.sqrt(n1**2-np.cos(a)**2))*(k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2))*sigma1**2) # Rauheit Korrektur e^{-2*k_{0}*k{1} *sigma1**2}
    +np.exp(-2j*(k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2))*z) # + X_2 : e^{2i k_{1} z}
    *((k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2)) -(k*np.sqrt((1-delta2-beta2*const_p)**2-np.cos(a)**2))) #r_{1,2} Zähler
    /((k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2)) +(k*np.sqrt((1-delta2-beta2*const_p)**2-np.cos(a)**2))) # r_{1,2} Nenner
    *np.exp(-2*(k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2))*(k*np.sqrt((1-delta2-beta2*const_p)**2-np.cos(a)**2))*sigma2**2))/ # Rauheit Korrektur e^{-2*k_{0}*k{1} *sigma2**2}
    (1+((k*np.sqrt(n1**2-np.cos(a)**2))-(k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2))) # Nenner startet hier 1+ r_{0,1} Zähler
    /((k*np.sqrt(n1**2-np.cos(a)**2))+(k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2)))    # r_{0,1} Nenner
    *np.exp(-2*(k*np.sqrt(1**2-np.cos(a)**2))*(k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2))*sigma1**2) # Rauheit Korrektur e^{-2*k_{0}*k{1} *sigma1**2}
    *np.exp(-2j*(k*np.sqrt((1-delta1-beta1*const_p)**-np.cos(a)**2))*z) # *X_2 : e^{2i k_{1} z}
    *((k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2)) -(k*np.sqrt((1-delta2-beta2*const_p)**2-np.cos(a)**2)))/ #r_{1,2} Zähler
    ((k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2))+(k*np.sqrt((1-delta2-beta2*const_p)**2-np.cos(a)**2))) # r_{1,2} Nenner
    *np.exp(-2*(k*np.sqrt((1-delta1-beta1*const_p)**2-np.cos(a)**2))*(k*np.sqrt((1-delta2-beta2*const_p)**2-np.cos(a)**2))*sigma2**2)) # Rauheit Korrektur e^{-2*k_{0}*k{1} *sigma1**2}
    )**2

parat_params, parat_pcov = curve_fit(parat, rel_phi, rel_intensity, p0=[10e-6, 9.8e-6, 4, 100,0.8e-10, 0.8e-10])
parat_err = np.sqrt(np.diag(parat_pcov))
print(parat_params, parat_err)

plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.yscale("log")
plt.axvline(x = 0.195, linestyle="--", color="k", label=r"$\theta_\text{c, exp}")
plt.ylabel(r"Anzahl")
plt.plot(rel_phi, rel_intensity, label="Normierte Werte")
plt.plot(rel_phi, parat(np.deg2rad(rel_phi),delta1, delta2, beta1, beta2, sigma1, sigma2), label="Parratt Theoriekurve")
#plt.plot(rel_phi, parat(np.deg2rad(rel_phi), *parat_params), label="Parat Fit")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/parat.pdf")

#print(parat(np.deg2rad(rel_phi),delta1, delta2, beta1, beta2, sigma1, sigma2))

amp = ufloat(detector_params[0], detector_err[0])
mu = ufloat(detector_params[1], detector_err[1])
sigma = ufloat(detector_params[2], detector_err[2])
amp1 = amp/(sigma*np.sqrt(2*np.pi))
halb = sigma*2*np.sqrt(2*np.log(2))
a_c_1 = np.rad2deg(np.sqrt(2*delta1))
a_c_2 = np.rad2deg(np.sqrt(2*delta2))
f=  open("content/results.txt", "w")
f.write(f"Gauss amp = {amp}, \nmu = {mu}, \nsigma = {sigma}\n")
f.write(f"Amplitude={amp1}\n")
f.write(f"Halbwertsbreite={halb}\n\n")
f.write(f"alpha_si = {alpha_si}\nlambda_Cu = {r_lambda:.2e}\nn={n}\n")
f.write(f"Strahlhöhe d_0 = {d_0}\n")
f.write(f"G-Faktor = {G_faktor}\n")
f.write(f"Dicke = {dicke}\n\n")
f.write(f"delta1 =  {delta1}, \ndelta2 = {delta2}, \nbeta1 = {b1}, \nbeta2 = {b2} \nsigma1 = {sigma1}, \nsigma2 = {sigma2}\n")
f.write(f"Winkel a_cr_1 = {a_c_1},\n Winkel a_cr_2 = {a_c_2},\n ")
f.close()