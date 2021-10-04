# Notebook에서 Modeling 부분을 보며...

## 인상적이었던 부분

1. 각 model의 prediciton 결과를 연결해 dataframe으로 만들어서 'Heat Map'으로 출력함으로써 '각 모델의 예측이 얼마나 비슷하게 이루어지는 지' 확인하는 것이 인상적이었다.
2. Train size에 대한 learning curve를 출력한 notebook은 이번에 처음 리뷰한 것 같다. 새로웠고, 유익했다.

## 아쉬웠던 부분

1. One-hot encoding 을 수행해서 data가 많이 sparse 해졌다.
  - Tree 기반 모델이 잘 동작하지 않을 것 같은데, Ensemble에 Tree 기반 모델을 4개나 포함시킨 이유가 뭘까...? feature importance 출력해보려고 했나..?
  - 이렇게 sparse 해진 data에서 Tree 기반모델로 feature importance 출력하는 게 맞는 방법인 지 궁금하다. 이렇게 해도 되는 건가..?
