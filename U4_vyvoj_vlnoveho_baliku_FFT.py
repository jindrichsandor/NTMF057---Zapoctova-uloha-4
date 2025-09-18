import math
import numpy as np
import scipy 
import os
import time

####################  Řešení časového vývoje jednorozměrného vlnového balíku pomocí metody rozděleného propagátoru ####################
########## Nastavení programu ##########
save_folder="test"  # složka, do které se uloží data
output_file_name="test"  # názvy souborů s výslednými daty - zadáváme bez přípony (přípona je automaticky .dat)


### Výběr typu úlohy
potential_option=2 # 1 = nulový potenciál; 2 = lineární harmonický oscilátor (HO)
initial_wave_packet_option=1 # 1 = gaussovský vlnový balík; 2 = vlastní stav lineárního HO

calculate_exact_solution=True # False = nebude se počítat přesné řešení; True = vypočte se přesné řešení (včetně chyby numerického řešení), pokud je přesné řešení dostupné 
if calculate_exact_solution==True and (potential_option==1 and initial_wave_packet_option==1): exact_solution_option=1 # přesný časový vývoj volného gaussovského vlnového balíku
elif calculate_exact_solution==True and (potential_option==2 and initial_wave_packet_option==2): exact_solution_option=2 # přesný časový vývoj vlastního stavu lineárního harmonického oscilátoru
else: exact_solution_option=0 # pokud přesné řešení není dostupné


### Nastavení časového kroku
dt=0.01 # velikost časového kroku
t_steps=2000 # počet kroků kroků
t=[i*dt for i in range(t_steps+1)] # pole s časy (N.B.: t[0]=0 je počáteční čas)

### Nastavení gridu x=x_0+i*h, i=0,1,...,n
x_0=-12.775# počátek gridu
h=0.05; # mřížková konstanta
k=9
N=2**k # počet bodů gridu
n=N-1 
x=[x_0+i*h for i in range(N)] # grid

### Fyzikální parametry systemu (N.B.: v kódu automaticky všude předpokládáme hbar=1)
mu=1 # hmotnost částice
omega=1.5 # frekvence lineárního harmonického oscilátoru

# nastavení pro gaussovský vlnový balík
x_0G=-5 # počáteční střední poloha
p_0G=0 # počáteční střední hodnota hybnosti
sigma=0.5 # počáteční šířka

# nastavení pro vlastní stav harmonického oscilátoru
n_HO=0 # n_HO=0,1,2,...  (kvantové číslo určující energii vlastního stavu HO; stupeň Hermiteova polynomu)



########## Definice funkcí ##########
### Vlnové funkce
def harmonic_oscillator(n_HO,x): # vlastní stav harmonického oscilátoru (stacionární)
    temp=[0]*(n_HO+1) 
    temp[n_HO]=1 
    H_n=np.polynomial.hermite.Hermite(temp) # Hermiteův polynom stupně n
    psi=1/(np.sqrt(2**n_HO*math.factorial(n_HO)))*np.power(mu*omega/np.pi,1/4)*np.exp(-mu*omega*x**2/2)*H_n(np.sqrt(mu*omega)*x)
    return psi

def harmonic_oscillator_evol(n_HO,x,t): # časová evoluce vlastního stavu harmonického oscilátoru (v příslušném poli)
    E_n=omega*(n_HO+1/2)
    return np.exp(-1j*E_n*t)*harmonic_oscillator(n_HO,x)

def gaussian_wave_packet(x): # gaussovský vlnový balík
    psi=np.power(2*np.pi*sigma**2,-1/4)*np.exp(-(x-x_0G)**2/(4*sigma**2)+1j*p_0G*(x-x_0G)) 
    return psi

def free_gaussian_wave_packet_evol(x,t): # časová evoluce volného gaussovského vlnového balíku
    Sigma2_t = sigma**2+t**2/(4*(mu*sigma)**2) 
    X_t = x_0G+p_0G*t/mu
    
    arg=np.angle(1/np.sqrt(mu+1j*t/(2*sigma**2)))
    phi=p_0G*(x-X_t)+p_0G**2*t/(2*mu)+t*(x-X_t)**2/(8*mu*sigma**2*Sigma2_t)+arg
    
    psi=np.power(2*np.pi*Sigma2_t,-1/4)*np.exp(-(x-X_t)**2/(4*Sigma2_t)+1j*phi)
    return psi

### Numerická integrace - složené lichoběžníkové pravidlo 
def num_integration_trapezoid(f):
    f[0]=f[0]/2
    f[-1]=f[-1]/2
    return h*np.sum(f)


### Časový krok pomocí metody rozděleného propagátoru
def split_operator_method_time_step(psi):
    ## evoluce vlnového balíku pomocí metody rozděleného propagátoru
    for i in range(N): psi[i]=np.exp(-1j*V[i]*dt/2)*psi[i] # přenásobení složek vlnové funkce čísly exp(-1j*V_i*dt/2)
    
    psi=scipy.fft.fft(psi) # Fourierova transformace
    for i in range(N): psi[i]=np.exp(-1j*dt*(p[i])**2/(2*mu))*psi[i] # přenásobení složek vlnové funkce čísly exp(-1j*p**2*dt/(2*mu))
    
    psi=scipy.fft.ifft(psi) # inverzní Fourierova transformace 
    for i in range(N): psi[i]=np.exp(-1j*V[i]*dt/2)*psi[i] # přenásobení složek vlnové funkce čísly exp(-1j*V_i*dt/2)
    return psi


start_time = time.time() 
########## Vstupní data a zavedení polí pro ukládání výsledků ##########
## potenciál
if potential_option == 1:
    V=[0]*(N) # nulový potenciál
elif potential_option == 2:
    V=[mu*omega**2*x[i]**2/2 for i in range(N)] # lineární harmonický oscilátor

#x počáteční vlnový balík
if initial_wave_packet_option==1:
    psi_0=np.array([gaussian_wave_packet(x[i]) for i in range(N)], dtype=complex)
elif initial_wave_packet_option==2:
    psi_0=np.array([harmonic_oscillator(n_HO,x[i]) for i in range(N)], dtype=complex)

## pole pro ukládání výsledků
c_t=np.array([0]*(t_steps+1),dtype=complex) # autokorelační funcke (numerické řešení)
psi_t_on_boundary=np.array([[0,0] for i in range(t_steps+1)], dtype=complex) # pole pro uložení vývoje vlnové funkce na okraji gridu kvůli ověřování, že vlnová funkce zůstává v průběhu časového vývoje na okraji nulová
error=np.array([0]*(t_steps+1),dtype=float) # pole pro ukládání chyby numerického řešení

## pomocná pole pro ukládání mezivýsledků
psi_t_exact=np.array([0]*N,dtype=complex) # pole pro ukládání přesné vlnové funkce
temp_dot_product=np.array([0]*N,dtype=complex)


########## Hlavní program - evoluce vlnové funkce a výpočet autokorelační funkce ##########
### výpočet hybností dle poznámek k přednášce
p=np.array([0]*N,dtype=float)
for j in range(int(N/2-1+1)): # hybnosti pro j=0,1,...,N/2-1
    p[j]=2*np.pi*j/(N*h)
for j in range(int(N/2),int(N-1+1)): # hybnosti pro j=N/2,....,N-1
    p[j]=2*np.pi*(j-N)/(N*h)
# ------ Poznámka - k výpočtu hybnosti lze alternativně použít funkci z knihovny scipy
#p_scipy=scipy.fft.fftfreq(N,d=h) # N.B.: frekvence je ještě nutné přenásobit faktorem "2*pi"
#p=2*np.pi*p_scipy
# ------

### Uložení hodnot v čase t=0
## autokorelační funkce c(t=0)
for i in range(N): temp_dot_product[i]=np.conjugate(psi_0[i])*psi_0[i]
c_t[0]=num_integration_trapezoid(temp_dot_product)  
## hodnoty vlnové funkce na okraji gridu v čase t=0
psi_t_on_boundary[0,0]=psi_0[0] # psi(t,x_0)
psi_t_on_boundary[0,1]=psi_0[-1] # psi(t,x_n)

### Evoluce vlnového balíku a výpočet autokorelační funkce c(t)
psi=psi_0.copy()

  
if exact_solution_option==0: 
    for tt in range (1,t_steps+1):
        psi=split_operator_method_time_step(psi) # provedení evoluce stavu z času "t" do času "t+dt" pomocí metody rozděleného propagátoru
        
        ## výpočet autokorelační funkce c(t)
        for i in range(N): temp_dot_product[i]=np.conjugate(psi[i])*psi_0[i]
        c_t[tt]=num_integration_trapezoid(temp_dot_product) 

        ## uložení okrajových hodnot vlnové funkce
        psi_t_on_boundary[tt,0]=psi[0] # psi(t,x_0)
        psi_t_on_boundary[tt,1]=psi[-1] # psi(t,x_n)


elif exact_solution_option==1:
    for tt in range (1,t_steps+1):
        psi=split_operator_method_time_step(psi) # provedení evoluce stavu z času "t" do času "t+dt" pomocí metody rozděleného propagátoru
        
        ## výpočet autokorelační funkce c(t)
        for i in range(N): temp_dot_product[i]=np.conjugate(psi[i])*psi_0[i]
        c_t[tt]=num_integration_trapezoid(temp_dot_product) 

        ## uložení okrajových hodnot vlnové funkce
        psi_t_on_boundary[tt,0]=psi[0] # psi(t,x_0)
        psi_t_on_boundary[tt,1]=psi[-1] # psi(t,x_n)
        
        ## výpočet chyby
        for i in range(N): psi_t_exact[i]=free_gaussian_wave_packet_evol(x[i],t[tt])
        
        error[tt]=np.max(np.absolute(psi-psi_t_exact)) # největší hodnota chyby
            
elif exact_solution_option==2: 
    for tt in range (1,t_steps+1):
        psi=split_operator_method_time_step(psi) # provedení evoluce stavu z času "t" do času "t+dt" pomocí metody rozděleného propagátoru
        
        ## výpočet autokorelační funkce c(t)
        for i in range(N): temp_dot_product[i]=np.conjugate(psi[i])*psi_0[i]
        c_t[tt]=num_integration_trapezoid(temp_dot_product) 

        ## uložení okrajových hodnot vlnové funkce
        psi_t_on_boundary[tt,0]=psi[0] # psi(t,x_0)
        psi_t_on_boundary[tt,1]=psi[-1] # psi(t,x_n)

        ## výpočet chyby
        for i in range(N): psi_t_exact[i]=harmonic_oscillator_evol(n_HO,x[i],t[tt])
        
        error[tt]=np.max(np.absolute(psi-psi_t_exact)) # největší hodnota chyby
              
runtime=time.time()-start_time


########## Uložení výsledků do datových souborů ##########
file_directory=os.path.dirname(os.path.abspath(__file__)) # absolutní cesta k adresáři obsahující Python skript
os.chdir(file_directory) # změna pracovního adresáře

# vytvoření složky pro ukládání dat (pokud ještě neexistuje)
if os.path.exists(save_folder)==False:
    os.mkdir(save_folder)
  
### export dat   
# numerické řešení autokorelační funkce c(t)
file_name=save_folder+"/"+output_file_name+"_c_t.dat"
data=[[t[i],np.real(c_t[i]),np.imag(c_t[i])] for i in range(t_steps+1)] # formát dat: [t, Re(c(t)), Im(c(t))] 
np.savetxt(file_name,data,fmt="%1.30s",delimiter="  ", header="t   Re(c(t))    Im(c(t))", comments="")

if exact_solution_option>0:
    # chyba numerického řešení
    file_name=save_folder+"/"+output_file_name+"_error.dat"
    data=[[t[i],error[i]] for i in range(t_steps+1)] # formát dat: [t, Re(c_num(t)-c_exact(t)), Im(c_num(t)-c_exact(t))] 
    np.savetxt(file_name,data,fmt="%1.30s",delimiter="  ", header="t   error=Max(Abs(psi_num(x,t)-psi_exact(x,t)))", comments="")

# okrajové podmínky 
file_name=save_folder+"/"+output_file_name+"_BC.dat"
data=[[t[i],np.real(psi_t_on_boundary[i,0]),np.imag(psi_t_on_boundary[i,0]),np.real(psi_t_on_boundary[i,1]),np.imag(psi_t_on_boundary[i,1])] for i in range(t_steps+1)] # formát dat: [t, Re(psi(x_0,t)), Im(psi(x_0,t)), Re(psi(x_n,t)), Im(psi(x_n,t))]
np.savetxt(file_name,data,fmt="%1.30s",delimiter="  ", header="t   Re(psi(x_0,t))   Im(psi(x_0,t))    Re(psi(x_n,t))   Im(psi(x_n,t))", comments="")

# log soubor s nastavením numerického výpočtu
file_name=save_folder+"/"+output_file_name+"_log.dat"
log_file=open(file_name, "w")
log_file.write("h= "+str(h)+"\n")
log_file.write("N= "+str(N)+"\n")
log_file.write("x_0= "+str(x_0)+"\n")
log_file.write("x_n= "+str(x[-1])+"\n")

log_file.write("dt= "+str(dt)+"\n")
log_file.write("t_steps= "+str(t_steps)+"\n")

log_file.write("mu= "+str(mu)+"\n")
log_file.write("omega= "+str(omega)+"\n")

log_file.write("x_0_Gauss= "+str(x_0G)+"\n")
log_file.write("p_0= "+str(p_0G)+"\n")
log_file.write("sigma= "+str(sigma)+"\n")

log_file.write("n_HO= "+str(n_HO)+"\n")
log_file.write("runtime(s)= "+str(runtime)+"\n")

log_file.close