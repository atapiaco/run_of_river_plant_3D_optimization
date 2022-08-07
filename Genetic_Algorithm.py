# módulos de Python que vamos a utilizar
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import numpy as np
#import scipy as sp
from scipy import interpolate
#from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

from scipy import interpolate
from scipy.interpolate import interp1d

# PARA SPLINES
from scipy.interpolate import CubicSpline

# PARA INTEGRAR
from scipy import integrate

import pandas as pd



# MODO MULTI O SINGLE
SO = True

# PENA DE MUERTE
PENALIZA = 1000000

# IMPORTO EL MAPA DE ALTURAS [X Y Z] Y EL PERFIL DEL RÍO [X Y]
VX  = np.loadtxt('HeightMap_X.csv',delimiter=",")
VY  = np.loadtxt('HeightMap_Y.csv',delimiter=",")
TZ  = np.loadtxt('HeightMap_Z.csv',delimiter=",")
RR  = np.flipud(np.loadtxt('HeightMap_R.csv',delimiter=","))
RX  = np.array([ RR[i][0] for i in range(len(RR)) ])
RY  = np.array([ RR[i][1] for i in range(len(RR)) ])
RS  = np.array(([0]))
for i in range(1,len(RR)):
    RS = np.append(RS,RS[i-1]+((RX[i]-RX[i-1])**2+(RY[i]-RY[i-1])**2)**0.5)
RSmax = RS[-1]
F_RX = interpolate.interp1d(RS,RX)
F_RY = interpolate.interp1d(RS,RY)

# PARA PASAR DE S A X-Y EN EL RÍO
def F_RIVER(s):
    return np.array((F_RX(s), F_RY(s)))

# PARA PASAR DE X-Y A Z EN EL TERRENO
F_TERRAIN = interpolate.interp2d(VX, VY, np.transpose(TZ), kind='linear')


# CTES PARA GENERAR INDIVIDUOS
Dmax  = 40*0.01

# CTES DEL MODELO DE LA PLANTA
F     = 0.010
RHO   = 1000
G     = 9.80
DNOZ  = 22e-3
SNOZ  = (np.pi*DNOZ**2)/4
REND  = 0.90
Pmin  = 7.0E3
dzT   = 1
dzD   = 0.5

# COSTE DE LA TUBERÍA, SOPORTES Y EXCAVACIONES
Ksop = 9.0
Kexc = 8.0
Xsop = 0.2
Bexc = np.pi/180*35


def crea_individuo(Hmin=100, Noise=4.2):
    valido = False
    while valido == False:        
        # DIÁMETRO
        D = Dmax*(0.2+0.8*random.random())
        # POSICION DE TURBINA Y PRESA
        ST = RSmax*random.random()
        SD = RSmax*random.random()
        if ST>SD:
            S0 = ST
            ST = SD
            SD = S0
        # SI NO CUMPLE ALTURA, LA EXTIENDO
        XT,YT = F_RIVER(ST)
        XD,YD = F_RIVER(SD)
        Hg    = F_TERRAIN(XD,YD) - F_TERRAIN(XT,YT)
        while Hg<Hmin:
            XT,YT = F_RIVER(ST)
            XD,YD = F_RIVER(SD)
            Hg    = F_TERRAIN(XD,YD) - F_TERRAIN(XT,YT)
            ST    = np.max( [ST - 5, 0] )
            SD    = np.min( [SD + 5, RSmax] )
            # NÚMERO DE NODOS
        #N = random.randint(Nmin,Nmax)
        N = random.randint(3,4)
            # NODOS INTERIORES
        IND = [D, ST, SD]
        for i in range(N):
            Snew = ST + (1+i)*(SD-ST)/(N+2)
            Xnew, Ynew = F_RIVER(Snew)
            Znew = 3*random.random()
            IND.append(Xnew + np.random.normal(loc=0.0, scale=Noise))
            IND.append(Ynew + np.random.normal(loc=0.0, scale=Noise))
            IND.append(Znew)
        fitn = fitness_function(IND,depurar=False)
        if fitn[0] < PENALIZA:
            valido = True
    return IND

def crea_individuo_original(Hmin=200, Noise=1):
    # DIÁMETRO
    D = Dmax*(0.2+0.8*random.random())
    # POSICION DE TURBINA Y PRESA
    ST = RSmax*random.random()
    SD = RSmax*random.random()
    if ST>SD:
        S0 = ST
        ST = SD
        SD = S0
    # SI NO CUMPLE ALTURA, LA EXTIENDO
    XT,YT = F_RIVER(ST)
    XD,YD = F_RIVER(SD)
    Hg    = F_TERRAIN(XD,YD) - F_TERRAIN(XT,YT)
    while Hg<Hmin:
        XT,YT = F_RIVER(ST)
        XD,YD = F_RIVER(SD)
        Hg    = F_TERRAIN(XD,YD) - F_TERRAIN(XT,YT)
        ST    = np.max( [ST - 5, 0] )
        SD    = np.min( [SD + 5, RSmax] )
        # NÚMERO DE NODOS
    #N = random.randint(Nmin,Nmax)
    N = random.randint(2,6)
        # NODOS INTERIORES
    IND = [D, ST, SD]
    for i in range(N):
        Snew = ST + (1+i)*(SD-ST)/(N+2)
        Xnew, Ynew = F_RIVER(Snew)
        Znew = 3*random.random()
        IND.append(Xnew + np.random.normal(loc=0.0, scale=Noise))
        IND.append(Ynew + np.random.normal(loc=0.0, scale=Noise))
        IND.append(Znew)
    return IND

def mutacion(IND, sigma=4.2, Pmut01=0.05, Pmut10=0.05, PmutM=0.05):
    N = int((len(IND)-3)/3)
        # MUTACION DEL DIAMETRO
    IND[0] = IND[0] + np.random.normal(scale=0.01)
    if IND[0]<=.1*Dmax:
        IND[0]=0.1*Dmax
    if IND[0]>=Dmax:
        IND[0]=Dmax
        # MUTACION DE LOS NODOS EXTREMOS
    Lp = IND[2]-IND[1]
    if random.random() < PmutM:
        IND[1] = np.random.normal(loc=IND[1], scale=sigma)
    if random.random() < PmutM:        
        IND[2] = np.random.normal(loc=IND[2], scale=sigma)
    if IND[1]<0:
        IND[1]=0
    if IND[2]>RSmax:
        IND[2]=RSmax
        # MUTACION DE NODOS INTERIORES
        # MUTACIÓN MOV
    for i in range(3,len(IND)):
        if random.random() < PmutM:
            if (i+2)%3 == 0: # Es la Z
                IND[i] = np.random.normal(loc=IND[i], scale=sigma*0.2)
            else:   # Es la x o y
                IND[i] = np.random.normal(loc=IND[i], scale=sigma)
        # MUTACIÓN 1-0
    if random.random() < Pmut10 and N>=2:
        i = 3+random.randint(0,N-2)
        del IND[i]
        del IND[i]
        del IND[i]
        # MUTACION 0-1
    if random.random() < Pmut01:
        s0 = IND[1]+Lp*random.random()
        if s0 > RSmax:
            s0 = RSmax
        x0,y0 = F_RIVER(s0)
        IND.append(np.random.normal(loc=x0, scale=sigma))
        IND.append(np.random.normal(loc=y0, scale=sigma))
        IND.append(np.random.normal(loc=0, scale=sigma*0.1))
    return IND,

def cruce(IND1, IND2, alpha=0.5):
    gamma = (1+2*alpha)*random.random() - alpha
    D3 = (1-gamma)*IND1[0] + gamma*IND2[0]
    D4 = (1-gamma)*IND2[0] + gamma*IND1[0]
    ST = np.min([IND1[1],IND2[1]])
    SD = np.max([IND1[2],IND2[2]])
    IND3 = [D3,ST,SD]
    IND4 = [D4,ST,SD]
    NODOSX = [ IND1[i] for i in np.arange(3,len(IND1),3) ] + [ IND2[i] for i in np.arange(3,len(IND2),3) ]
    NODOSY = [ IND1[i] for i in np.arange(4,len(IND1),3) ] + [ IND2[i] for i in np.arange(4,len(IND2),3) ]
    NODOSZ = [ IND1[i] for i in np.arange(5,len(IND1),3) ] + [ IND2[i] for i in np.arange(5,len(IND2),3) ]
    for i in range(len(NODOSX)):
        if random.random()>0.5:
            IND3.append(NODOSX[i])
            IND3.append(NODOSY[i])
            IND3.append(NODOSZ[i])
        else:
            IND4.append(NODOSX[i])
            IND4.append(NODOSY[i])
            IND4.append(NODOSZ[i])
    return IND1, IND2

def fitness_function(IND, depurar=False):
    # EL CROMOSOMA TIENE TAMAÑO: 1+2+3*N
    # DIAM, TURB, PRESA, CODOS
    DIAM = IND[0]
    
    XYZ_T = np.append( F_RIVER(IND[1]), dzT )
    XYZ_D = np.append( F_RIVER(IND[2]), dzD )
    XYZ_N = np.array([ [IND[i],IND[i+1],IND[i+2]] for i in np.arange(3,len(IND),3) ])
    
    
    NODES   = np.append(np.append(XYZ_T,XYZ_N),XYZ_D)
    NODES   = NODES.reshape(len(NODES)//3,3)    
    NODES3D = np.array([ np.append(NODES[i][0:2], NODES[i][2] + F_TERRAIN(NODES[i][0],NODES[i][1])) for i in range(len(NODES)) ])
    
    # AQUI SE DEBEN ORDENAR LOS NODOS SEGUN Z CRECIENTE
    ZNODES = NODES3D[1:-1].transpose()[2]
    orden  = np.argsort(ZNODES)
    NODES3D[1:-1] = NODES3D[orden+1]

    # CON SPLINES
    T0   = np.linspace(0,1,len(NODES3D))
    cs_x = CubicSpline(T0,NODES3D[:,0],bc_type='natural')
    cs_y = CubicSpline(T0,NODES3D[:,1],bc_type='natural')
    cs_z = interpolate.PchipInterpolator(T0,NODES3D[:,2])
    def INTERPOLA(t):
        return cs_x(t), cs_y(t), cs_z(t)
    
    # INTEGRAL DE LÍNEA
    tt  = np.linspace(0,1,200) # ESTABA EN 1000
    TT  = INTERPOLA(tt)
    dTT = np.array(( np.gradient(TT[0],tt[1]-tt[0]), np.gradient(TT[1],tt[1]-tt[0]), np.gradient(TT[2],tt[1]-tt[0]) ))
    ss  = integrate.cumtrapz( np.sqrt( dTT[0]**2 + dTT[1]**2 + dTT[2]**2 ), tt )
    ss = np.append(0,ss)
    
    # COMPROBAMOS CURVATURA
    ttr = np.linspace(0,1,100)
    TTr  = INTERPOLA(ttr)
    r = np.zeros(len(ttr)-3)
    for i in range(len(r)):
        x1, y1 = TTr[0][i  ], TTr[1][i  ]
        x2, y2 = TTr[0][i+1], TTr[1][i+1]
        x3, y3 = TTr[0][i+2], TTr[1][i+2]
        cr = ( (x1-x2)**2 + (y1-y2)**2 )**0.5
        ar = ( (x2-x3)**2 + (y2-y3)**2 )**0.5
        br = ( (x3-x1)**2 + (y3-y1)**2 )**0.5
        r[i]  = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5
    r_min = min(r)
    r_lim = 200E9*DIAM/(2*250E6)
    if r_min < r_lim:
        return PENALIZA,
    
    # ARC-LENGTH PARAMETRIZATION
    fun_s2t = interpolate.interp1d(ss, tt)
    def INTERPOLA_S(s):
        tnew = fun_s2t(s)
        return INTERPOLA(tnew)
    
    # AÑADO EL LAYOUT SOBRE EL TERRENO A TT
    # TT CONTIENE [XX, YY, ZZpenstck, ZZproy]
    TT_terr = np.zeros(len(TT[0]))
    for i in range(0,len(TT_terr)):
        TT_terr[i] = F_TERRAIN(TT[0][i], TT[1][i])
    TT = np.vstack([TT, TT_terr])
    
    # ALTURA DE LA TURBINA Y ALTURA DE LA PRESA
    zT = TT[2][ 0]
    zD = TT[2][-1]
    
    # SI HAY NODOS MAS BAJOS QUE LA TURBINA O MAS ALTOS QUE LA PRESA, NO VALE
    if np.min(TT[2]) != zT or np.max(TT[2]) != zD:
        return PENALIZA+1,
    
    # MHPP PERFORMANCE
    Lp = ss[-1]
    Hg = zD-zT
    P  = REND * (RHO/(2*SNOZ**2))*(Hg/(1/(2*G*SNOZ**2)+F*Lp/(DIAM**5)))**(3/2)
    if P<Pmin:
        return PENALIZA+2,
    
    # COST OF CIVIL WORKS
    dz_sop = np.where(TT[2]>TT[3], TT[2]-TT[3], 0)
    dz_exc = np.where(TT[3]>TT[2], TT[3]-TT[2], 0)
    cc_sop = Ksop*Xsop*np.power(dz_sop*4/3, 1)
    cc_exc = Kexc*np.tan(Bexc)*np.power(dz_exc, 2) + DIAM*dz_exc
    C_sop = np.trapz(cc_sop,ss)
    C_exc = np.trapz(cc_exc,ss)
    
    # COST OF PENSTOCK
    C_tub = Lp*(616.1*DIAM**2 + 99.76*DIAM + 13.14)
    
    # TOTAL COST
    C = C_tub + C_exc + C_sop
    Q = 1000*(Hg/(1/(2*9.8*SNOZ**2)+0.01*Lp/DIAM**5))**(1/2)
    
    if depurar:
        
        print('Height: \t',"%.2f" % Hg,' m')
        print('Flow: \t',"%.2f" % Q,' L/s')
        print('Long: \t',"%.2f" % Lp,' m')
        print('Diam: \t',"%.2f" % DIAM,' m')
        print('r min: \t',"%.2f" % r_min,' m')
        print('r_lim: \t',"%.2f" % r_lim,' m')
        print('Cost: \t',"%.2f" % C,' c.u.')
        print('C_sop \t', C_sop)
        print('C_exc \t', C_exc)
        print('C_tub \t', C_tub)
        print('Potencia', P)
        
        VY2 = VY[:45]
        TZ2 = TZ.transpose()[:45]
        XX,YY = np.meshgrid(VX,VY2)
        plt.close('all')
        
        
            # SUPERFICIE 2D-A
        fig2, ax2 = plt.subplots()
        ax2.contour(XX,YY,TZ2,20,colors='grey',inline=1,alpha=1,linewidths=0.6)
        ax2.set_aspect('equal')
        
        
        
        distance = np.cumsum( np.sqrt(np.sum( np.diff(RR, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        alpha = np.linspace(0, 1, 1000)
        interpolator =  interp1d(distance, RR, kind='cubic', axis=0)
        interpolated_points = interpolator(alpha)
        ax2.plot(*interpolated_points.T, color='blue', linewidth=2);
        # ax2.plot(RX,RY,'b')
        ax2.plot(TT[0],TT[1],'r', linewidth=3)
        ax2.plot(NODES3D[0,0],NODES3D[0,1],'ko')
        ax2.plot(NODES3D[-1,0],NODES3D[-1,1],'ko')
        
        plt.text(530,500, 'Powerhouse', fontsize=12)
        plt.text(830,80, 'Extraction', fontsize=12)
        
        ax2.legend(('River', 'MHPP Layout'))
        plt.xlabel('x-coordinate (m)')
        plt.ylabel('y-coordinate (m)')
        
            # SUPERFICIE 2D-B
        fig3, ax3 = plt.subplots()
        ax3.plot(tt,TT[3])
        ax3.plot(tt,TT[2])

            # SUPERFICIE 2D-C
        fig4, ax4 = plt.subplots()
        ax4.plot(tt,TT[2]-TT[3])
        
            # SUPERFICIE 2D-C
        fig5, ax5 = plt.subplots()
        ax5.plot(ttr[:-3],r)

    return C,

   
    
    # PARA SO
    # Paso1: creación del problema
creator.create("Problema1", base.Fitness, weights=(-1,))
    # Paso2: creación del individuo
creator.create("individuo", list, fitness=creator.Problema1)
    # Creamos la caja de herramientas
toolbox = base.Toolbox() 
    # Registramos nuevas funciones
toolbox.register("individuo", tools.initIterate, creator.individuo, crea_individuo )
toolbox.register("ini_poblacion", tools.initRepeat, list, toolbox.individuo)
    # Operaciones genéticas
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", cruce)
toolbox.register("mutate", mutacion)
toolbox.register("select", tools.selTournament, tournsize = 3)



def unico_objetivo_ga(c, m):
    """ Los parámetros de entrada son la probabilidad de cruce y la
    probabilidad de mutación """
    
    NGEN   = 400
    MU     = 2000
    LAMBDA = 2000
    CXPB   = c
    MUTPB  = m
    
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB,
                                             MUTPB, NGEN, stats=stats,
                                             halloffame=hof, verbose=True)
    
    return pop, logbook

#%%############################################################################
# LANZAMIENTO ÚNICO
###############################################################################

pop, logbook = unico_objetivo_ga(c=0.3, m=0.7)