import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy import stats
from uncertainties import ufloat
from scipy.optimize import curve_fit

#Detektor Scan
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

# Messung und Diffusionskorrektur
messung_phi, messung_intensity = np.genfromtxt("data/messung.UXD", unpack=True)

diffus_phi, diffus_intensity = np.genfromtxt("data/messung2.UXD", unpack=True)

rel_intensity = messung_intensity - diffus_intensity
rel_phi = diffus_phi

plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.yscale("log")
plt.ylabel(r"Anzahl")
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

def ideal(alpha):
    return (np.abs((k * np.sin(alpha)- k*np.sqrt(n**2-np.cos(alpha)**2))/(k * np.sin(alpha)+ k*np.sqrt(n**2-np.cos(alpha)**2))))**2

rel_intensity = rel_intensity/rel_intensity[rel_phi == alpha_crit]

plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.yscale("log")
plt.ylabel(r"Anzahl")
plt.plot(rel_phi, rel_intensity, label="Normierte Werte")
plt.plot(rel_phi, ideal(np.deg2rad(rel_phi)), label="Ideale Siliziumoberfläche")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/messwerte_norm.pdf")

# Korrektur durch Geometriefaktor
dreieck_phi, dreieck_intensity = np.genfromtxt("data/rocking_scan.UXD", unpack=True)

plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
#plt.yscale("log")
plt.ylabel(r"Anzahl")
plt.plot(dreieck_phi, dreieck_intensity, label="Messwerte")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/dreieck.pdf")

D = 0.02
a_g = 0.72
d = np.sin(np.deg2rad(a_g))*D
G_faktor = D/d

faktor = []
for i in rel_phi:
    if i < a_g:
        faktor = np.append(faktor, G_faktor*np.sin(np.deg2rad(i)))
    else:
        faktor = np.append(faktor, 1)

faktor[0]= 0.00001
rel_intensity = rel_intensity/faktor
rel_intensity = rel_intensity/rel_intensity[rel_phi == alpha_crit]

# Bestimmung der Dicke 

np.savetxt("data/normalised_and_corrected.txt",np.array([rel_phi, rel_intensity]).T, fmt="%.3e")
num, min_phi, min_intensity = np.genfromtxt("data/minima.txt", unpack=True)

q_z = 2*k*np.sin(np.deg2rad(min_phi))
del_q = []

for i in range(len(q_z)-1):
    del_q = np.append(del_q, q_z[i+1]-q_z[i])

dicke = 2*np.pi/del_q
dicke_mean = dicke.mean()
dicke_err = dicke.std(ddof=1)/np.sqrt(len(dicke))
dicke = ufloat(dicke_mean, dicke_err)

# Parratt 

z= 8.2e-8 #dicke_mean
delta1=5.1e-6
delta2=5.7e-6
beta1=400 
beta2=500
sigma1=5.5e-10
sigma2=8e-10

# Brechungsindizes
n1=1
n2=1-delta1-1.54j*10**(-8)*400/(4*np.pi)
n3=1-delta2-1.54j*10**(-8)*40/(4*np.pi)

qz=2*k*np.sin(np.deg2rad(rel_phi))

kz1=k*np.sqrt(n1**2-np.cos(np.deg2rad(rel_phi))**2)
kz2=k*np.sqrt(n2**2-np.cos(np.deg2rad(rel_phi))**2)
kz3=k*np.sqrt(n3**2-np.cos(np.deg2rad(rel_phi))**2)

#modifizierte Fresnelkoeffizienten
r12=(kz1-kz2)/(kz1+kz2)*np.exp(-2*kz1*kz2*sigma1**2)
r23=(kz2-kz3)/(kz2+kz3)*np.exp(-2*kz2*kz3*sigma2**2)

w2=np.exp(-2j*kz2*z)*r23
w1=(r12+w2)/(1+r12*w2)
w1_1=np.abs(w1)**2

const_p = 1.54e-8j/(4*np.pi)
normal_p = 1.54e-8/(4*np.pi)
beta1 = beta1*normal_p 
beta2 = beta2*normal_p

#  * e^{-2*k_{j}*k{j+1} * sigma1 }
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

#parat_params, parat_pcov = curve_fit(parat, rel_phi, rel_intensity, p0=[10e-6, 9.8e-6, 2, 2,0.8e-10, 0.8e-10])
#parat_err = np.sqrt(np.diag(parat_pcov))
#print(parat_params, parat_err)

plt.figure()
plt.xlabel(r"$\theta$ / \si{\degree}")
plt.yscale("log")
plt.ylabel(r"Anzahl")
plt.plot(rel_phi, rel_intensity, label="Normierte Werte")
plt.plot(rel_phi, parat(np.deg2rad(rel_phi), delta1,delta2,beta1,beta2,sigma1,sigma2), label="Parratt Theoriekurve")
#plt.plot(rel_phi, parat(np.deg2rad(rel_phi), *parat_params), label="Parat Fit")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/parat.pdf")

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
f.write(f"Halbwertsbreite={halb}\n")
f.write(f"alpha_si = {alpha_si}\nlambda_Cu = {r_lambda:.2e}\nn={n}\n")
f.write(f"G-Faktor = {G_faktor}\n")
f.write(f"Dicke = {dicke}\n")
f.write(f"delta1 =  {delta1}, \ndelta2 = {delta2}, \nbeta1 = {beta1}, \nbeta2 = {beta2} \nsigma1 = {sigma1}, \nsigma2 = {sigma2}\n")
f.write(f"Winkel a_cr_1 = {a_c_1},\n Winkel a_cr_2 = {a_c_2},\n ")
f.close()