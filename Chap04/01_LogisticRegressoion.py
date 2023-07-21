import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
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