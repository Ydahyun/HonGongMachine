import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 특성의 스케일을 조정하는 방법은 표준점수 말고도 더 있다.
# 하지만 대부분의 경우 표준점수로 충분하다.

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros((14))))

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

print(test_target)  # 도미:빙어 = 35:14 = 2.5:1 그러나 테스트 세트의 도미:빙어 = 3.3:1 -> 샘플링 편향

train_input_re, test_input_re, train_target_re, test_target_re = train_test_split(fish_data, fish_target,
                                                                      stratify=fish_target, random_state=42)
print(test_target_re)  #  2.25:1 로 적합해짐.

kn = KNeighborsClassifier()
kn.fit(train_input_re, train_target_re)
kn.score(test_input_re, test_target_re)
kn.predict([[25, 150]])  # [0.]

plt.scatter(train_input_re[:,0], train_input_re[:,1])
plt.scatter(25, 150, marker='^')
plt.xlim((0, 1000))  # x축 범위를 1000까지 늘려줌
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

mean = np.mean(train_input_re, axis=0)          #  평균
std = np.std(train_input_re, axis=0)            #  표준편차
train_scaled = (train_input_re - mean) / std    #  표준점수 변환

new = ([25,150]-mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

test_scaled = (test_input_re - mean) / std
kn.score(test_scaled, test_target_re)
print(kn.predict([new]))