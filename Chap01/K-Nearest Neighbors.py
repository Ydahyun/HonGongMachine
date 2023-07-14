# K-최근접 이웃 알고리즘을 사용하여 도미와 빙어데이터를 구분.
# K-최근접 알고리즘은 가장 가까운 5개의 데이터를 보고 다수결의 원칙에 따라 데이터를 예측한다.
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# 첫 번째 머신러닝 프로그램
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]

print(fish_data)
# 정답주기
fish_target = [1]*35 + [0]*14
print(fish_target)

kn = KNeighborsClassifier()
# 훈련
kn.fit(fish_data, fish_target)
# 평가
print(kn.score(fish_data, fish_target))
# 예측
print(kn.predict([[30, 600]]))


"""
# K-최근접 알고리즘은 가장 가까운 5개의 데이터를 보고 다수결의 원칙에 따라 데이터를 예측한다.
# 참고 데이터수를 5개에서 다른 수로 바꾸는 방법
kn49 = KNeighborsClassifier(n_neighbors=49)  # 참고 데이터를 49개로 한 kn49 모델
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
"""