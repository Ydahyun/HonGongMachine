import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

# 로지스틱 회귀 학습 전 k-최근접 이웃 분류기의 확률 예측
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()

print(pd.unique(fish['Species']))

fish_input = fish[['Weight', 'Length', 'Diagonal','Height','Width']].to_numpy()
print(fish_input[:5])

# 타깃 데이터에 2개 이상의 클래스가 포함된 문제를 다중분류(Multi Class Classification)라고 한다.
fish_target = fish['Species'].to_numpy()  # pd -> np 로 변환,  타겟 데이터에는 7가지의 생선 종류가 들어감.


# 분류(훈련, 타겟)
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 훈련세트와 테스트세트를 표준화 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃 분류기의 확률 예측
# 최근접 이웃개수 k를 3으로 지정하여 훈련세트와 테스트 점수 확인
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)  # 훈련세트로 fit
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
print(kn.classes_)
print(kn.predict(test_scaled[:5]))

proba = kn.predict_proba(test_scaled[:5])
print()
print("'Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish'")
print(np.round(proba, decimals=4))  # 각 클래스의 확률

# 로지스틱 회귀는 이름은 회귀이지만 분류 모델이다. 이 알고리즘은 선형 회귀와 동일하게 선형 방정식을 학습한다.
# ex. z = a*(Weight) + b*(Length) + c*(Diagonal) + d*(Height) + e*(Width) + f
# a,b,c,d,e는 가중치 혹은 계수이며 z는 어떤 값도 가능하다. 확률로 표현하기위해(0과 1사이) 시그모이드함수를 사용.
z = np.arange(-5,5,0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# 로지스틱 회귀로 이진 분류 수행하기

# 넘파이 배열은 불리언 인덱싱이라고 True만 전달할 수 있다.
char_arr = np.array(['A','B','C','D','E'])
print(char_arr[[True, False, True, False, False]])
# 이와같이 도미(Bream)과 빙어(Smelt)의 행만 골라내면
bream_smelt_indexes = (train_target =='Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.classes_)
print(lr.coef_, lr.intercept_)

# z값 계산
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)  # 이 z값을 시그모이드 함수에 통과시키면 확률을 얻을 수 있다. expit(), np.exp()
print(expit(decisions))