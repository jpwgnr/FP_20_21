import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy import stats
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy import constants as const

# 0 falls der Plot nicht neu gemacht werden soll oder 1 falls es neu gemacht werden soll
messung_t1 = 0
messung_t2 = 0
echo = 1
fourier = 1

# # T_1 - Messung
tau, amp = np.genfromtxt("data/data.txt", unpack=True) 

def t1_func(t, t1, m0):
    return m0 *(1 - 2*np.exp(-t/t1))

params1, pcov1 = curve_fit(t1_func, tau, amp)
err1 = np.sqrt(np.diag(pcov1))

tau_new = np.linspace(tau[0]-0.00005, tau[-1]+1, 10000)

plt.figure()
plt.xlabel(r"$\tau / \si{\second}$")
plt.xscale("log")
plt.ylabel(r"Amplitude / \si{\volt}")
plt.plot(tau, amp, "x", label="Datenpunkte")
plt.plot(tau_new, t1_func(tau_new, *params1), label="Ausgleichskurve")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.savefig("content/T1.pdf")

# T_2 - Messung

t, V1 = np.genfromtxt("data/peaks.csv", delimiter=',',unpack=True)

def t2_func(t, t2, m0, m1):
    return m0 * np.exp(-t/t2) + m1

t_peak = t[18::20]
V1_peak = V1[18::20]
params2, pcov2 = curve_fit(t2_func, t_peak, V1_peak)
err2 = np.sqrt(np.diag(pcov2))

t_peak_new = np.linspace(t_peak[0]-0.2, t_peak[-1]+0.2, 10000)
plt.figure()
plt.xlabel(r"t / \si{\second}")
plt.ylabel(r"Amplitude / $\si{\volt}$")
plt.plot(t, V1, label="Datenpunkte")
plt.plot(t_peak, V1_peak, "x", label="Maxima")
plt.plot(t_peak_new, t2_func(t_peak_new, *params2), label="Ausgleichskurve")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.savefig("content/T2.pdf")

t2 = params2[0]


t_echo, h_echo = np.genfromtxt("data/data_t2.txt", unpack=True)

t_echo = t_echo[:-2]
h_echo = h_echo[:-2]

def echo_func(t, d, m0, m1):
    return m0 * np.exp(-2*t/t2) * np.exp(-t**3/d) + m1

params3, pcov3 = curve_fit(echo_func, t_echo, h_echo, p0=[1.7e-6,1.4,3e-2])
err3 = np.sqrt(np.diag(pcov3))

t_echo_new = np.linspace(t_echo[0]-8e-3, t_echo[-1]+0.5e-3, 10000)
plt.figure()
plt.xlabel(r"$\tau^3 / \si{\micro\second}$")
plt.ylabel(r"$\ln\left(M(\tau)\right) - 2\tau/T_2$")
plt.plot(t_echo**3*1e6, np.log(h_echo)-2*t_echo/t2, "x", label="Datenpunkte")
plt.plot(t_echo_new**3*1e6, np.log(echo_func(t_echo_new, *params3))-2*t_echo_new/t2, label="Ausgleichskurve")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.savefig("content/echo.pdf")

# Fourier-Trafo:
    
#Laden der Daten aus der Datei "echo_gradient.csv"
#Die erste Spalte enthält die Zeiten in Sekunden, die zweite Spalte
#den Realteil und die dritte Spalte den Imaginärteil
data = np.genfromtxt("data/number4a.csv", delimiter=",", unpack= True)
times = data[0]
real = data[1]
imag = data[2]
#Suchen des Echo-Maximums und alle Daten davor abschneiden
start = np.argmin(real)
times = times[start:]
real = real[start:]
imag = imag[start:]
#Phasenkorrektur - der Imaginärteil bei t=0 muss = 0 sein
phase = np.arctan2(imag[0], real[0])
#Daten in komplexes Array mit Phasenkorrektur speichern
compsignal = (real*np.cos(phase)+imag*np.sin(phase))+ \
(-real*np.sin(phase)+imag*np.cos(phase))*1j
#Offsetkorrektur, ziehe den Mittelwert der letzten 512 Punkte von allen Punkten ab
compsignal = compsignal - compsignal[-512:-1].mean()
#Der erste Punkt einer FFT muss halbiert werden
compsignal[0] = compsignal[0]/2.0
#Anwenden einer Fensterfunktion (siehe z. Bsp.
#https://de.wikipedia.org/wiki/Fensterfunktion )
#Hier wird eine Gaußfunktion mit sigma = 100 Hz verwendet
apodisation = 100.0*2*np.pi
compsignal = compsignal*np.exp(-1.0/2.0*((times-times[0])*apodisation)**2)
#Durchführen der Fourier-Transformation
fftdata = np.fft.fftshift(np.fft.fft(compsignal))
#Generieren der Frequenzachse
freqs = np.fft.fftshift(np.fft.fftfreq(len(compsignal), times[1]-times[0]))
#Speichern des Ergebnisses als txt
np.savetxt("data/echo_gradient_fft.txt", np.array([freqs, np.real(fftdata), \
np.imag(fftdata)]).transpose())
#Erstellen eines Plots
plt.figure()
plt.xlabel(r"$f / \si{kHz}")
plt.ylabel(r"Anzahl")
plt.plot(freqs[(freqs>-15000) & (freqs<15000)]*1e-3, np.real(fftdata)[(freqs>-15000) & (freqs<15000)], "x", label="Fourier-Transformation")
#plt.plot(freqs, np.real(fftdata))
#plt.plot(times, real)
#plt.plot(times, imag)
plt.legend(loc="best")
plt.grid()
plt.tight_layout()
plt.savefig("content/echo_gradient.pdf")

d_f = 6.578947368420178464e+03 + 8.771929824560238558e+03
gamma = const.value("proton gyromag. ratio")
gamma_u = const.unit("proton gyromag. ratio")
durchmesser = 4.4e-3 # oder 4.2e-3 ?
g = 2 * np.pi *d_f /gamma /durchmesser # Gradient, damit dann Diffusionskoeffizient bestimmen
Diff = 1/params3[0] * (3/2) /gamma**2 /g**2 

f=  open("content/results.txt", "w")
f.write(f"T_1 =( {params1[0]} +/- {err1[0]}) s, U_0 = ({params1[1]} +/- {err1[1]}) V\n")
f.write(f"T_2 =( {params2[0]} +/- {err2[0]}) s, U_0 = ({params2[1]} +/- {err2[1]}) V, U_1 = ({params2[2]} +/- {err2[2]}) V\n ")
f.write(f"Konstante =( {params3[0]} +/- {err3[0]}) s, U_0 = ({params3[1]} +/- {err3[1]}) V, U_1 = ({params3[2]} +/- {err3[2]}) V \n")
f.write(f"d_f = {d_f*1e-3} kHz\n")
f.write(f"gamma = {gamma} {gamma_u}\n")
f.write(f"durchmesser = {durchmesser} m\n")
f.write(f"gradient = {g} 1/m?\n")
f.write(f"Diffusionskoeff. = {Diff}\n")
f.close()