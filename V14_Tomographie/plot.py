import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy import stats
from uncertainties import ufloat
from scipy.optimize import curve_fit

#Spektrum der 137 Cs Quelle
N = np.genfromtxt("data/leere_messung.txt", unpack=True) # C-Channel, N-Counts pro Sekunde
N = N/300
C = np.arange(0, len(N), 1, int)

plt.xlabel("Kanal")
plt.ylabel("Zählrate / $s^{-1}$")
#plt.yscale('log')
plt.xlim(20, 129)
plt.plot(C[20:130], N[20:130], label="Spektrum")
plt.axvline(x=104, ymin=0, ymax=10**4, linewidth=1, linestyle="-", color = 'grey', label = r"$662\,$keV")
plt.legend(loc="best") 
plt.grid()
plt.savefig("figures/Spektrum.pdf")


#Nullmessung
Nullmsg = ufloat(49118, 264)/300 #erstmal unnötig


#Defintion der Matrix A
A = np.matrix([[0, 0, 0, 0, 0, np.sqrt(2), 0, np.sqrt(2), 0],
               [0, 0, np.sqrt(2), 0, np.sqrt(2), 0, np.sqrt(2), 0, 0],
               [0, np.sqrt(2), 0, np.sqrt(2), 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1, 0, 0, 0],
               [1, 1, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, np.sqrt(2), 0, 0, 0, np.sqrt(2), 0],
               [np.sqrt(2), 0, 0, 0, np.sqrt(2), 0, 0, 0, np.sqrt(2)],
               [0, np.sqrt(2), 0, 0, 0, np.sqrt(2), 0, 0, 0],
               [1, 0, 0, 1, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 1, 0, 0, 1]])

A_t = np.transpose(A)
A_total = np.linalg.inv(A_t*A)*A_t


#Würfel 1 - leer
N_0, err, t = np.genfromtxt("data/wuerfel1.txt", unpack=True)
N_0 = unp.uarray(N_0, err)/t 
print(f"\n Die Zählraten des leeren Würfels sind (I_2 I_3 I_5):\n {N_0}")
N_0_12 = ([N_0[1], N_0[0], N_0[1], 
           N_0[2], N_0[2], N_0[2], 
           N_0[1], N_0[0], N_0[1], 
           N_0[2], N_0[2], N_0[2], ]) #I_0 Vektor für alle Projektionsebenen
#I_0 = unp.log(N_0/N)
#d = 0.1 #cm
#d = ([2*np.sqrt(2)*d, 2*np.sqrt(2)*d, 2*d])
#mu = I_0/d
#rho = 2.71 # gr/cm**3
#print(f" \n Absorptionskoeffizienten leerer Aluminiumwürfel: \n  mu = {mu}cm^-1 \n = {mu/rho}cm^2/g")


#Würfel 2 - homogen
N, err, t = np.genfromtxt("data/wuerfel2.txt", unpack=True)
N = unp.uarray(N, err)/t 
print(f"\n Die Zählraten des Würfels 2 sind (I_2 I_3 I_5):\n {N}")
I = unp.log(N_0/N)
d = 1 #cm
d = ([3*d*np.sqrt(2), 2*d*np.sqrt(2), 3*d])
mu = I/d
print(f" \n Absorptionskoeffizienten homogener Würfel 2: \n  mu = {mu}cm^-1 \n m_mittel = {np.mean(mu)}")


#Würfel 3 - homogen
N, err, t = np.genfromtxt("data/wuerfel3.txt", unpack=True)
N = unp.uarray(N, err)/t 
print(f"\n Die Zählraten des Würfels 3 sind (I_2 I_3 I_5):\n {N}")
I = unp.log(N_0/N)
d = 1 #cm
d = ([3*d*np.sqrt(2), 2*d*np.sqrt(2), 3*d])
mu = I/d
print(f" \n Absorptionskoeffizienten homogener Würfel 3: \n  mu = {mu}cm^-1 \n m_mittel = {np.mean(mu)}")

#Würfel 5 -unbekannte Zusammensetzung
#N, err = np.genfromtxt("data/wuerfel5.txt", unpack=True)
#t = 300
#N = unp.uarray(N, err)/t 
#print(f"\n Die Zählraten des Würfels 5 sind (I_1 - I_12):\n {N}")
#I = unp.log(N_0_12/N)
#d = 1 #cm
#mu = A_total*I.transpose()
#print(f" \n Absorptionskoeffizienten Würfel 5: \n  mu = {mu}cm^-1 \n m_mittel = {np.mean(mu)}")

#Würfel 5 -unbekannte Zusammensetzung
N, err = np.genfromtxt("data/wuerfel5.txt", unpack=True)
t = 300
N = N/t
err = err/t
print(f"\n Die Zählraten des Würfels 5 sind (I_1 - I_12):\n {N}+/- {err}")
I = np.log(unp.nominal_values(N_0_12)/N)
I_err = (unp.std_devs(N_0_12)/unp.nominal_values(N_0_12))**2+ (err/N)**2 # Varianzen I \sigma**2
I_t = np.transpose(np.array([I]))
d = 1 #cm
mu = d*A_total*I_t
print(f" \n Absorptionskoeffizienten Würfel 5: \n  mu = {mu}cm^-1")

#Varianzen berechnen
V_I = np.diag(I_err)
V_mu = np.linalg.inv(A_t*np.linalg.inv(V_I)*A)
mu_err = np.sqrt(np.diagonal(V_mu))
print(mu_err)

#Abweichung zu anderen mu
mu_Fe_lit = 0.607
mu_Fe_exp = 0.613
mu_De_lit = 0.12
mu_De_exp = 0.089

print(f"\n absolute Abweichungen zu Eisen Literatur: \n {mu-mu_Fe_lit} \n")
print(f"\n absolute Abweichungen zu Eisen Experiment: \n {mu-mu_Fe_exp} \n")
print(f"\n absolute Abweichungen zu Delrin Literatur: \n {mu-mu_De_lit} \n")
print(f"\n absolute Abweichungen zu Delrin Experiment: \n {mu-mu_De_exp} \n")