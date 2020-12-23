import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy import stats
import scipy.constants as const
from uncertainties import ufloat
from scipy.optimize import curve_fit

T1, I1 = np.genfromtxt("data/messung1.txt", unpack=True)
T2, I2 = np.genfromtxt("data/messung2.txt", unpack=True)

T1 = T1 + 273.15
T2 = T2 + 273.15
I1 = I1*1e-11
I2 = I2*1e-11

def gerade(x, m, n):
    return m*x + n 

time1 = np.arange(0, len(T1)*30, 30)
time2 = np.arange(0, len(T2)*30, 30)

temp_params1, temp_pcov1 = curve_fit(gerade, time1, T1)
temp_err1 = np.sqrt(np.diag(temp_pcov1))

temp_params2, temp_pcov2 = curve_fit(gerade, time2, T2)
temp_err2 = np.sqrt(np.diag(temp_pcov2))

time1_new = np.linspace(time1[0] - 100, time1[-1] + 100, 1000)
time2_new = np.linspace(time2[0] - 100, time2[-1] + 100, 1000)
plt.figure()
plt.xlabel(r"t / \si{\second}")
plt.ylabel(r"I / \si{\kelvin}")
plt.plot(time1, T1, "x", label="Rate 1 Daten")
plt.plot(time2, T2, "x", label=" Rate 2 Daten")
plt.plot(time1_new, gerade(time1_new, *temp_params1), label="Ausgleichsgeraden", color="k")
plt.plot(time2_new, gerade(time2_new, *temp_params2), color="k")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/rate.pdf")

b1 = ufloat(temp_params1[0], temp_err1[0])*60
T_start_1 = ufloat(temp_params1[1], temp_err1[1])

b2 = ufloat(temp_params2[0], temp_err2[0])*60
T_start_2 = ufloat(temp_params2[1], temp_err2[1])

T1_fit = np.append(T1[0:18], T1[63:])
I1_fit = np.append(I1[0:18], I1[63:])

T2_fit = np.append(T2[0:7], T2[69:])
I2_fit = np.append(I2[0:7], I2[69:])
def bkg(T, a, b):
    return a*np.exp(-b/T)

bkg_params1, bkg_pcov1 = curve_fit(bkg, T1_fit, I1_fit)
bkg_err1 = np.sqrt(np.diag(bkg_pcov1))

bkg_params2, bkg_pcov2 = curve_fit(bkg, T2_fit, I2_fit)
bkg_err2 = np.sqrt(np.diag(bkg_pcov2))

T1_new = np.linspace(T1[0]-5, T1[-1]+5, 10000)
T2_new = np.linspace(T2[0]-5, T2[-1]+5, 10000)

plt.figure()
plt.xlabel(r"T / \si{\kelvin}")
plt.ylabel(r"I / \si{\pico\ampere}")
plt.plot(T1, I1*1e12, "x", label="Datenpunkte")
plt.plot(T1_new, bkg(T1_new, *bkg_params1)*1e12, label="Ausgleichskurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/data_w_bkg1.pdf")

plt.figure()
plt.xlabel(r"T / \si{\kelvin}")
plt.ylabel(r"I / \si{\pico\ampere}")
plt.plot(T2, I2*1e12, "x", label="Datenpunkte")
plt.plot(T2_new, bkg(T2_new, *bkg_params2)*1e12, label="Ausgleichskurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/data_w_bkg2.pdf")


I1_neu = (I1 - bkg(T1, *bkg_params1)) 
T1_anlauf = T1[26:43]
I1_anlauf = I1_neu[26:43]
T1_integral = T1[26:60]
I1_integral = I1_neu[26:60]

I2_neu = (I2 - bkg(T2, *bkg_params2))
T2_anlauf = T2[11:31]
I2_anlauf = I2_neu[11:31]
T2_integral = T2[11:60]
I2_integral = I2_neu[11:60]

plt.figure()
plt.xlabel(r"T / \si{\kelvin}")
plt.ylabel(r"I / \si{\pico\ampere}")
plt.plot(T1, I1_neu*1e12, "x", label="Datenpunkte")
plt.plot(T1_integral, I1_integral*1e12, "x", label="Daten für Integral")
plt.plot(T1_anlauf, I1_anlauf*1e12, "x", label="Daten für Anlaufkurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/data_wo_bkg1.pdf")

plt.figure()
plt.xlabel(r"T / \si{\kelvin}")
plt.ylabel(r"I / \si{\pico\ampere}")
plt.plot(T2, I2_neu*1e12, "x", label="Datenpunkte")
plt.plot(T2_integral, I2_integral*1e12, "x", label="Daten für Integral")
plt.plot(T2_anlauf, I2_anlauf*1e12, "x", label="Daten für Anlaufkurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/data_wo_bkg2.pdf")


anlauf_params1, anlauf_pcov1 = curve_fit(gerade, 1/T1_anlauf, np.log(I1_anlauf*1e12))
anlauf_err1 = np.sqrt(np.diag(anlauf_pcov1))

invers_T1 = np.linspace(1/T1_anlauf[0]+2.5e-5, 1/T1_anlauf[-1]-2.5e-5, 1000)
plt.figure()
plt.xlabel(r"1/T / \si{\per\kelvin}")
plt.ylabel(r"$\ln$(I / \si{\pico\ampere})")
plt.plot(1/T1_anlauf, np.log(I1_anlauf*1e12), "x", label="Datenpunkte")
plt.plot(invers_T1, gerade(invers_T1, *anlauf_params1), label="Ausgleichskurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/anlauf1.pdf")

anlauf_params2, anlauf_pcov2 = curve_fit(gerade, 1/T2_anlauf, np.log(I2_anlauf*1e12))
anlauf_err2 = np.sqrt(np.diag(anlauf_pcov2))

invers_T2 = np.linspace(1/T2_anlauf[0]+2.5e-5, 1/T2_anlauf[-1]-2.5e-5, 1000)
plt.figure()
plt.xlabel(r"1/T / \si{\per\kelvin}")
plt.ylabel(r"$\ln$(I / \si{\pico\ampere})")
plt.plot(1/T2_anlauf, np.log(I2_anlauf*1e12), "x", label="Datenpunkte")
plt.plot(invers_T2, gerade(invers_T2, *anlauf_params2), label="Ausgleichskurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/anlauf2.pdf")

from scipy.integrate import simps
int1 = []
for  i  in range(len(T1_integral)-1):
 sim = simps(I1_integral[i:], T1_integral[i:])
 int1 = np.append(int1, unp.log(sim/(b1*I1_integral[i])))

int_params1, int_pcov1 = curve_fit(gerade, 1/T1_integral[:-1], unp.nominal_values(int1))
int_err1 = np.sqrt(np.diag(int_pcov1))

invers_T1 = np.linspace(1/T1_integral[0]+2.5e-5, 1/T1_integral[-1]-2.5e-5, 1000)
plt.figure()
plt.xlabel(r"1/T / \si{\per\kelvin}")
plt.ylabel(r"$\ln \left(\frac{\int_T^\infty i(T')\symup{d}T'}{i(T)\tau_0 b}\right)$")
plt.plot(1/T1_integral[:-1], unp.nominal_values(int1), "x",  label="Datenpunkte")
plt.plot(invers_T1, gerade(invers_T1, *int_params1), label="Ausgleichskurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/integral1.pdf")


int2 = []
for  i  in range(len(T2_integral)-1):
 sim = simps(I2_integral[i:], T2_integral[i:])
 int2 = np.append(int2, unp.log(sim/(b2*I2_integral[i])))

int_params2, int_pcov2 = curve_fit(gerade, 1/T2_integral[:-1], unp.nominal_values(int2))
int_err2 = np.sqrt(np.diag(int_pcov2))

invers_T2 = np.linspace(1/T2_integral[0]+2.5e-5, 1/T2_integral[-1]-2.5e-5, 1000)
plt.figure()
plt.xlabel(r"1/T / \si{\per\kelvin}")
plt.ylabel(r"$\ln \left(\frac{\int_T^\infty i(T')\symup{d}T'}{i(T)\tau_0 b}\right)$")
plt.plot(1/T2_integral[:-1], unp.nominal_values(int2), "x", label="Datenpunkte")
plt.plot(invers_T2, gerade(invers_T2, *int_params2), label="Ausgleichskurve")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/integral2.pdf")

T_start1 = ufloat(temp_params1[1], temp_err1[1])
T_start2 = ufloat(temp_params2[1], temp_err2[1])

bkg_a1 = ufloat(bkg_params1[0], bkg_err1[0])
bkg_b1 = ufloat(bkg_params1[1], bkg_err1[1])
m1 = ufloat(anlauf_params1[0], anlauf_err1[0])
n1 = ufloat(anlauf_params1[1], anlauf_err1[1])
k1 = ufloat(int_params1[0], int_err1[0])
l1 = ufloat(int_params1[1], int_err1[1])
W_1_1 = -m1 * const.k/const.e
W_1_2 = k1 * const.k/const.e

bkg_a2 = ufloat(bkg_params2[0], bkg_err2[0])
bkg_b2 = ufloat(bkg_params2[1], bkg_err2[1])
m2 = ufloat(anlauf_params2[0], anlauf_err2[0])
n2 = ufloat(anlauf_params2[1], anlauf_err2[1])
k2 = ufloat(int_params2[0], int_err2[0])
l2 = ufloat(int_params2[1], int_err2[1])
W_2_1 = -m2 * const.k/const.e
W_2_2 = k2 * const.k / const.e

T_max1 = 273.15 -12.4
T_max2 = 273.15 -16.3 
tau_max1_anlauf = T_max1**2 * const.k /(b1*W_1_1*const.e)
tau_max1_int = T_max1**2 * const.k /(b1*W_1_2*const.e)
tau_max2_anlauf = T_max2**2 * const.k /(b2*W_2_1*const.e)
tau_max2_int = T_max2**2 * const.k /(b2*W_2_2*const.e)

tau0_1_anlauf = tau_max1_anlauf/unp.exp( W_1_1*const.e/T_max1 / const.k)
tau0_1_int = tau_max1_int/unp.exp( W_1_2*const.e/T_max1 / const.k)
tau0_2_anlauf = tau_max2_anlauf/unp.exp( W_2_1*const.e/T_max2 / const.k)
tau0_2_int = tau_max2_int/unp.exp( W_2_2*const.e/T_max2/ const.k)

tau1 = tau0_1_anlauf.nominal_value * np.exp(W_1_1.nominal_value *const.e /T1 /const.k)
tau2 = tau0_1_int.nominal_value * np.exp(W_1_2.nominal_value *const.e /T1 /const.k)
tau3 = tau0_2_anlauf.nominal_value * np.exp(W_2_1.nominal_value *const.e /T1 /const.k)
tau4 = tau0_2_int.nominal_value * np.exp(W_2_2.nominal_value *const.e /T1 /const.k)
plt.figure()
plt.xlabel(r"1/T / \si{\per\kelvin}")
plt.ylabel(r"$\tau$")
plt.plot(1/T1, tau1, label=r"Kurve mit $\tau_\text{0, 1, Anlauf}$")
plt.plot(1/T1, tau2, label=r"Kurve mit $\tau_\text{0, 1, Integral}$")
plt.plot(1/T1, tau3, label=r"Kurve mit $\tau_\text{0, 2, Anlauf}$")
plt.plot(1/T1, tau4, label=r"Kurve mit $\tau_\text{0, 2, Integral}$")
plt.legend(loc="best") 
plt.grid()
plt.tight_layout()
plt.savefig("figures/tau.pdf")

f=  open("content/results.txt", "w")
f.write(f"Heizrate: b1 = {b1} \nb2 = {b2} \n")
f.write(f"Heizrate: T_start1 = {T_start1} \nT_start2 = {T_start2} \n")
f.write(f"Background exp: a1 = {bkg_a1}, \nb1 = {bkg_b1}, \n")
f.write(f"Background exp: a2 = {bkg_a2}, \nb2 = {bkg_b2}, \n")
f.write(f"Anlauf Gerade: m1 = {m1}, \nn1 = {n1}\n")
f.write(f"Anlauf Gerade: m2 = {m2}, \nn2 = {n2}\n")
f.write(f"Integral Gerade: m1 = {k1}, \nn1 = {l1}\n")
f.write(f"Integral Gerade: m2 = {k2}, \nn2 = {l2}\n")
f.write(f"W1_anlauf = {W_1_1}\nW1_int = {W_1_2} \nW2_anlauf = {W_2_1}\nW2_int = {W_2_2} \n")
f.write(f"T_max: T_max1 = {T_max1}, \nT_max2 = {T_max2}\n")
f.write(f"tau_max1_anlauf = {tau_max1_anlauf}\ntau_max1_int = {tau_max1_int} \ntau_max2_anlauf = {tau_max2_anlauf}\ntau_max2_int = {tau_max2_int}\n")
f.write(f"tau0_1_anlauf = {tau0_1_anlauf}\ntau0_1_int = {tau0_1_int} \ntau0_2_anlauf = {tau0_2_anlauf}\ntau0_2_int = {tau0_2_int}\n")

f.close()
