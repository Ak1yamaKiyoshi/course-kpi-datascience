#---------- Лекція 2 - статистичний аналіз характеристик ВВ --------------
import numpy as np
import math as mt
import matplotlib.pyplot as plt
# ----------------------- МОДЕЛЬ випадкової похибки ----------------------
n=10000; iter=int(n)                      # кількість реалізацій ВВ ------
# --------------------- рівномірний закон розводілу ВВ
S=np.random.rand(iter)                    # генерація ВВ в масив S - параметри закону за замовченням
mS=np.median(S)                           # математичне сподівання ВВ
dS=np.var(S)                              # дисперсія ВВ
scvS=mt.sqrt(dS)                          # дисперсія ВВ
print('матриця реалізацій ВВ=',S)
print('матиматичне сподівання ВВ=',mS)
print('дисперсія ВВ =',dS)
print('СКВ ВВ=',scvS)
                                          # гістограма закону розподілу ВВ
plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
plt.show()

S=np.zeros((n))
for i in range(n):
    S[i]=np.random.randint(0, 100)         # параметри закону задаются межами аргументу
mS=np.median(S)
dS=np.var(S)
scvS=mt.sqrt(dS)
print('матриця реалізацій ВВ=',S)
print('матиматичне сподівання ВВ=',mS)
print('дисперсія ВВ =',dS)
print('СКВ ВВ=',scvS)
                                           # гістограма закону розподілу ВВ
plt.hist(S,  bins=20, facecolor="blue", alpha=0.5)
plt.show()

# --------------------- нормальний закон розводілу ВВ ---------------------
dm=0; dsig=5                               # параметри закону розподілу ВВ із систематикою dsig
#S = np.random.normal(dm, dsig, iter)      # коректура параметрів закону розподілу (1 спосіб)
S1 = np.random.randn(n)
S = ((np.random.randn(n))*dsig)+dm         # коректура параметрів закону розподілу (2 спосіб)
mS=np.median(S)
dS=np.var(S)
scvS=mt.sqrt(dS)
print('матриця реалізацій ВВ=',S)
print('матиматичне сподівання ВВ=',mS)
print('дисперсія ВВ =',dS)
print('СКВ ВВ=',scvS)
                                           # гістограма закону розподілу ВВ
plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
plt.show()

# --------------------- модель виміру (квадратичний закон) з нормальний шумом
S4=np.zeros((n)); S3=np.zeros((n)); S0=np.zeros((n))
for i in range(n):
    S0[i]=(0.0000005*i*i)                 # квадратична модель реального процесу
    S3[i] = S0[i]+S[i]
                                          # графік моделі реального процесу
plt.plot(S3)
plt.plot(S0)
plt.ylabel('динаміка продажів')
plt.show()

                                           # гістограми законів розподілу ВВ
plt.hist(S, bins=20, alpha=0.5, label='S')
plt.hist(S1, bins=20, alpha=0.5, label='S1')
plt.hist(S3, bins=20, alpha=0.5, label='S3')
plt.show()
                                           # статистичні характеристики трендової вибірки (зміщені)
mS3=np.median(S3)
dS3=np.var(S3)
scvS3=mt.sqrt(dS3)
print('матиматичне сподівання ВВ3=',mS3)
print('дисперсія ВВ3 =',dS3)
print('СКВ ВВ3=',scvS3)

# --------------------- оцінка статистичних характеристик ВВ з урахуванням динаміки зміни контрольованої велечини
for i in range(n):
      S4[i] = (S3[i]-S0[i])              # позбавлення квадратичної складової
                                            # ПРОБЛЕМАТИКА:
                                            # 1. Де в реальних задача взяти S0 - Л3
                                            # 2. Як встановити наявність систематики - dm по формі гістограми
                                            # 3. Якщо відняти dm гітограм S4 прийде до S - ні, ЧОМУ?

plt.hist(S, bins=20, alpha=0.5, label='S')
plt.hist(S1, bins=20, alpha=0.5, label='S1')
plt.hist(S3, bins=20, alpha=0.5, label='S3')
plt.hist(S4, bins=20, alpha=0.5, label='S4')
plt.show()
