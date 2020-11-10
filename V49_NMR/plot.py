import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy import stats
from uncertainties import ufloat
from scipy.optimize import curve_fit

# 0 falls der Plot nicht neu gemacht werden soll oder 1 falls es neu gemacht werden soll
messung_t1 = 0
messung_t2 = 0
echo = 1
fourier = 0

# # T_1 - Messung
if(messung_t1==True):
    tau, amp = np.genfromtxt("data/data.txt", unpack=True) 

    def t1_func(t, t1, m0):
        return m0 *(1 - 2*np.exp(-t/t1))

    params1, pcov1 = curve_fit(t1_func, tau, amp)
    err1 = np.sqrt(np.diag(pcov1))

    plt.figure()
    plt.plot(tau, amp, "x")
    plt.plot(tau, t1_func(tau, *params1))
    plt.savefig("content/T1.pdf")

# T_2 - Messung
if(messung_t2==True) or (echo==True):
    t, V1 = np.genfromtxt("data/peaks.csv", delimiter=',',unpack=True)

    def t2_func(t, t2, m0, m1):
        return m0 * np.exp(-t/t2) + m1

    t_peak = t[18::20]
    V1_peak = V1[18::20]
    params2, pcov2 = curve_fit(t2_func, t_peak, V1_peak)
    err2 = np.sqrt(np.diag(pcov2))

    plt.figure()
    plt.plot(t_peak, V1_peak, "x")
    plt.plot(t_peak, t2_func(t_peak, *params2))
    plt.savefig("content/T2.pdf")

    t2 = params2[0]

if (echo==True):
    t_echo, h_echo = np.genfromtxt("data/data_t2.txt", unpack=True)

    t_echo = t_echo[:-2]
    h_echo = h_echo[:-2]

    def echo_func(t, d, m0, m1):
        return m0 * np.exp(-2*t/t2) * np.exp(-t**3/d) + m1

    params3, pcov3 = curve_fit(echo_func, t_echo, h_echo, p0=[1.7e-6,1.4,3e-2])
    err3 = np.sqrt(np.diag(pcov3))

    plt.figure()
    plt.plot(t_echo**3, np.log(h_echo)-2*t_echo/t2, "x")
    plt.plot(t_echo**3, np.log(echo_func(t_echo, *params3))-2*t_echo/t2)
    plt.savefig("content/echo.pdf")

# Fourier-Trafo:
if (fourier==True):     
    #Laden der Daten aus der Datei "echo_gradient.csv"
    #Die erste Spalte enthält die Zeiten in Sekunden, die zweite Spalte
    #den Realteil und die dritte Spalte den Imaginärteil
    data = np.genfromtxt("data/number4a.csv", delimiter=",", unpack= True)
    times = data[0]
    real = -data[1]
    imag = data[2]
    #Suchen des Echo-Maximums und alle Daten davor abschneiden
    start = np.argmax(real)
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
    # plt.plot(freqs[(freqs>-15000) & (freqs<15000)], np.real(fftdata)[(freqs>-15000) & (freqs<15000)], "x")
    plt.figure()
    plt.plot(freqs, np.real(fftdata))
    # plt.plot(times, real)
    # plt.plot(times, imag)
    plt.savefig("content/echo_gradient.pdf")

    d_f = 3 # Bestimmen
    gamma = 5 # gyromagnetisches Verhältnis von Wasser?
    d = 4.4e-3 # oder 4.2e-3 ?
    g = 2 * np.pi *d_f /gamma /d # Gradient, damit dann Diffusionskoeffizient bestimmen