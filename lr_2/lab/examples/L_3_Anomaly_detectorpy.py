#---------------- Лекція 3 детектування та очищення вибірки від АВ-------------------------
# Функціонал:
# генерація випадкової вибірки з аномалиями
# нормальні виміри - нормальний законами розподілу випадкової величини.
# аномальні ивміри - рівномірно розподілені за вибіркою.
# виявлення аномельних вимірів;
# розрахунок статистичних характеристик випадкової величини;
#---- підключення модулів (бібліотек)  Python методи яких буде використано в програмі ---
import numpy as np
import math as mt
import matplotlib.pyplot as plt
# -------------------------------- ФУНКЦІЇ МОДЕЛЬ ВИМІРІВ ---------------------------
def Stat_characteristics (SL, Text):
    # статистичні характеристики ВВ з урахуванням тренду
    Yout = MNK(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter ))
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print('------------', Text ,'-------------')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(SL0, bins=20, facecolor="blue", alpha=0.5)
    plt.ylabel(Text)
    plt.show()
    return
# ----------- рівномірний закон розводілу номерів АВ в межах вибірки ----------------
def randomAM (n, iter):
    SAV = np.zeros((nAV))
    S = np.zeros((n))
    for i in range(n):
        S[i] = np.random.randint(0, iter)  # параметри закону задаются межами аргументу
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    # -------------- генерація номерів АВ за рівномірним законом  -------------------
    for i in range(nAV):
        SAV[i] = mt.ceil(np.random.randint(1, iter))  # рівномірний розкид номерів АВ в межах вибірки розміром 0-iter
    print('номери АВ: SAV=', SAV)
    print('----- статистичны характеристики РІВНОМІРНОГО закону розподілу ВВ -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return SAV

# ------------------------- нормальний закон розводілу ВВ ----------------------------
def randoNORM (dm, dsig, iter):
    S = np.random.normal(dm, dsig, iter)  # нормальний закон розподілу ВВ з вибіркою єбємом iter та параметрами: dm, dsig
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('------- статистичны характеристики НОРМАЛЬНОЇ похибки вимірів -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('------------------------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S

# ------------------- модель ідеального тренду (квадратичний закон)  ------------------
def Model (n):
    S0=np.zeros((n))
    for i in range(n):
        S0[i]=(0.0000005*i*i)    # квадратична модель реального процесу
    return S0

# ---------------- модель виміру (квадратичний закон) з нормальний шумом ---------------
def Model_NORM (SN, S0N, n):
    SV=np.zeros((n))
    for i in range(n):
        SV[i] = S0N[i]+SN[i]
    return SV

# ----- модель виміру (квадратичний закон) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
def Model_NORM_AV (S0, SV, nAV, Q_AV):
    SV_AV = SV
    SSAV = np.random.normal(dm, (Q_AV * dsig), nAV)  # аномальна випадкова похибка з нормальним законом
    for i in range(nAV):
        k=int (SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]        # аномальні вимірів з рівномірно розподіленими номерами
    return SV_AV

# --------------- графіки тренда, вимірів з нормальним шумом  ---------------------------
def Plot_AV (S0_L, SV_L, Text):
    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return

# ------------------------------ МНК згладжування -------------------------------------
def MNK (S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    return Yout

# ------------------------------ МНК згладжування -------------------------------------
def MNK_AV_Detect (S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    return C[1,0]
# ------------------------------ Виявлення АВ за алгоритмом medium -------------------------------------
def Sliding_Window_AV_Detect_medium (S0, n_Wind, Q):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind=mt.ceil(iter-n_Wind)+1
    S0_Wind=np.zeros((n_Wind))
    # -------- еталон  ---------
    j=0
    for i in range(n_Wind):
        l = (j + i)
        S0_Wind[i] = S0[l]
        dS_ethadone = np.var(S0_Wind)
        scvS_ethadone = mt.sqrt(dS_ethadone)
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l=(j+i)
            S0_Wind[i] = S0[l]
    # - Стат хар ковзного вікна --
        mS = np.median(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
    # --- детекція та заміна АВ --
        if scvS  > (Q*scvS_ethadone):
            # детектор виявлення АВ
            # print('S0[l] !!!=', S0[l])
            S0[l]=mS
        # print('----- Вікно -----')
        # print('mS=', mS)
        # print('scvS=',scvS)
        # print('-----------------')
    return S0

# ------------------------------ Виявлення АВ за МНК -------------------------------------
def Sliding_Window_AV_Detect_MNK (S0, Q, n_Wind):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind=mt.ceil(iter-n_Wind)+1
    S0_Wind=np.zeros((n_Wind))
    # -------- еталон  ---------
    Speed_ethadone = MNK_AV_Detect(SV_AV)
    Yout_S0 = MNK(SV_AV)
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l=(j+i)
            S0_Wind[i] = S0[l]
    # - Стат хар ковзного вікна --
        Speed=MNK_AV_Detect(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
    # --- детекція та заміна АВ --
        Speed_ethadone_1 = abs(Speed_ethadone * mt.sqrt(iter))
        # Speed_1 = abs(Speed / (Q*scvS))
        Speed_1 = abs(Q * Speed_ethadone * mt.sqrt(n_Wind) * scvS)
        # print('Speed_ethadone=', Speed_ethadone_1)
        # print('Speed_1=', Speed_1)
        if Speed_1  > Speed_ethadone_1:
            # детектор виявлення АВ
            # print('S0[l] !!!=', S0[l])
            S0[l]=Yout_S0[l,0]
    return S0

# ------------------------------ Виявлення АВ за алгоритмом sliding window -------------------------------------
def Sliding_Window_AV_Detect_sliding_wind (S0, n_Wind):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind=mt.ceil(iter-n_Wind)+1
    S0_Wind=np.zeros((n_Wind))
    Midi = np.zeros(( iter))
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l=(j+i)
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        Midi[l] = np.median(S0_Wind)
    # ---- очищена вибірка  -----
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi

# -------------------------------- БЛОК ОСНОВНИХ ВИКЛИКІВ ------------------------------
# ------------------------------ сегмент API (вхідних даних) ---------------------------
n=10000; iter=int(n)                     # кількість реалізацій ВВ
Q_AV=3                                   # коефіціент переваги АВ
nAVv=10; nAV=int ((iter*nAVv)/100)       # кількість АВ у відсотках та абсолютних одиницях
dm=0; dsig=5                             # параметри нормального закону розподілу ВВ: середне та СКВ
# ------------ виклики функцій моделей: тренд, аномального та нормального шуму  ----------
S0=Model (n)                             # модель ідеального тренду (квадратичний закон)
SAV=randomAM (n, iter)                   # модель рівномірних номерів АВ
S=randoNORM (dm, dsig, iter)             # модель нормальних помилок
# ----------------------------- Нормальні похибки ------------------------------------
SV=Model_NORM (S, S0, n)               # модель тренда + нормальних помилок
Plot_AV (S0, SV, 'квадратична модель + Норм. шум')
Stat_characteristics (SV, 'Вибірка + Норм. шум')
Yout_SV=MNK (SV)
Stat_characteristics (Yout_SV, 'МНК Вибірка + Норм. шум')
# ----------------------------- Аномальні похибки ------------------------------------
SV_AV=Model_NORM_AV (S0, SV, nAV, Q_AV)  # модель тренда + нормальних помилок + АВ
Plot_AV (S0, SV_AV, 'квадратична модель + Норм. шум + АВ')
Stat_characteristics (SV_AV, 'Вибірка з АВ')
Yout_SV_AV=MNK (SV_AV)
Stat_characteristics (Yout_SV_AV, 'МНК Вибірка + Норм. шум + АВ')
# ----------------- Очищення від аномальних похибок ковзним вікном --------------------
print('Оберіть іметод виявлення та очищення вибірки від АВ:')
print('1 - метод medium')
print('2 - метод MNK')
print('3 - метод sliding window')
mode = int(input('mode:'))

if (mode == 1):
    print('Вибірка очищена від метод medium АВ')
    # --------- Увага!!! якість результату залежить від якості еталонного вікна -----------
    N_Wind_Av = 5   # розмір ковзного вікна для виявлення АВ
    Q = 1.6         # коефіціент виявлення АВ
    S_AV_Detect_medium=Sliding_Window_AV_Detect_medium (SV_AV, N_Wind_Av, Q)
    Plot_AV (S0, S_AV_Detect_medium, 'Вибірка очищена від АВ алгоритм medium')
    Stat_characteristics (S_AV_Detect_medium, 'Вибірка очищена від алгоритм medium АВ')
    Yout_SV_AV_Detect=MNK (S_AV_Detect_medium)
    Stat_characteristics (Yout_SV_AV_Detect, 'МНК Вибірка відчищена від АВ алгоритм medium')

if (mode == 2):
    print('Вибірка очищена від АВ метод MNK')
    # ------------------- Відчищення від аномальних похибок МНК --------------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    Q_MNK = 7   # коефіціент виявлення АВ
    S_AV_Detect_MNK = Sliding_Window_AV_Detect_MNK(SV_AV, Q_MNK, n_Wind)
    Plot_AV(S0, S_AV_Detect_MNK, 'Вибірка очищена від АВ алгоритм MNK')
    Stat_characteristics(S_AV_Detect_MNK, 'Вибірка очищена від АВ алгоритм MNK')
    Yout_SV_AV_Detect_MNK = MNK(S_AV_Detect_MNK)
    Stat_characteristics(Yout_SV_AV_Detect_MNK, 'МНК Вибірка очищена від АВ алгоритм MNK')

if (mode == 3):
    print('Вибірка очищена від АВ метод MNK')
    # --------------- Відчищення від аномальних похибок sliding window -------------------
    n_Wind = 5  # розмір ковзного вікна для виявлення АВ
    S_AV_Detect_sliding_wind = Sliding_Window_AV_Detect_sliding_wind (SV_AV, n_Wind)
    Plot_AV(S0, S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind')
    Stat_characteristics(S_AV_Detect_sliding_wind, 'Вибірка очищена від АВ алгоритм sliding_wind')
    Yout_SV_AV_Detect_sliding_wind = MNK(S_AV_Detect_sliding_wind)
    Stat_characteristics(Yout_SV_AV_Detect_sliding_wind, 'МНК Вибірка очищена від АВ алгоритм sliding_wind')