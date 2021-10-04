# Titanic Top 4% with ensemble modeling

## Notebook summary

### 1. Data Load & Check

#### 1-1. outlier detection

> **'Tukey method'** 를 사용하여 outlier Search
- **'Tukey method'** 란?
    - 'Q1 - (1.5 * IQR) 미만' & 'Q3 + (1.5 * IQR)' 의 범위에 존재하는 Data를 Outlier로 판별하는 method이다.
    - 이때, 
        - **'Q1'** 은 1st quartile(첫번째 사분위)로써, `Q1 = np.percentile(df[col], 25)`와 같이 구할 수 있다.
        - **'Q3'** 는 3rd quartile(세번째 사분위)로써, `Q3 = np.percentile(df[col], 75)`와 같이 구할 수 있다.
        - **'IQR'** 은 interquartile range(사분위 범위)로써, `Q3 - Q1`와 같이 구할 수 있다.
    - 저자는, 두 개 이상의 feature에서 outlier를 가지는 observations(row)를 전체 dataset에서의 outlier로 판별하고 이를 제외하고자 하였다.

#### 1-2. Train과 Test set을 병합

> Categorical feature를 실수화 할 때, 동일하게 처리하기 위함인 듯 하다.

#### 1-3. Null과 missing value 확인

> **'Cabin'** 과 **'Age'** 에서 많은 missing values를 확인했다.

- **'Suvived'** 에서 발생한 missing values는 data set을 병합할 때 test에는 없는 target value이기 때문에 발생한 것이므로 신경쓰지 않아도 된다.

### 2. Feature 분석

#### 2-1. Numerival features

> **'Heat Map'** 을 통해 'SibSp', 'Parch', 'Age', 'Fare'등의 feature와 'Survived(target feature)'와의 상관계수를 확인하고 이에 따라 feature를 하나씩 살펴보는 방식!

- SibSP

    > 많은 형제자매와 배우자를 데리고 탑승한 승객이 더 적은 생존확률을 보였다.
    - **'sns.factorplot()'** 이용하여 분석하였다.
        - 이를 통해 괜찮은 관찰을 할 수 있었으므로, 이렇게 계속 분석해보자.
- Parch
    > family size가 적은 쪽이 sigle (Parch 0)이거나 medium family(Parch 3, 4) 혹은 large family(Parch 5, 6) 보다 더 높은 생존률을 나타냈다.
    - Parch 3에서 std가 큰 것에 유의해야 할 것 같다.
- Age
    > 'Age' feature가 HeatMap에서 target과 높은 상관계수 값을 보이지 않더라도 나이대에 따라 생존률의 차이를 보인다는 것을 확인하였다.
    - 매우 어린 나이대 (0~5세)에서 생존률 peak 값이 나타남.
    - 60~80 나이대 승객의 생존률이 높지 않다.
- Fare
    > 분포가 상당히 치우쳐져 있어 (skewness) 이대로 모델에 넣으면 overweight가 발생할 수 있으므로 'log function'을 통해 skewness를 줄여준다.

#### 2-2. Categorical features

- sex
    > 여성의 생존률이 남성의 생존률보다 매우 높았다.
    - "Women and children first."
- Pclass
    > 높은 Passenger class를 가진 승객이 더 높은 생존률을 보였다.
- Embark
    > 생존률이 C Q S 순으로 높았는데, Pclass와 함께 분포를 살펴보니, 높은 생존률을 보인 탑승처의 승객이 더 높은 Passenger class를 가진 것으로 밝혀졌다.
    - mising value는 2개 뿐이어서 최빈값이 S로 채웠다.

### 3. Filling missing Values

#### Age

> 앞서 missing values를 확인했을 때, Age feature에서 256개의 mising values를 확인했다. 

> Age의 특정 분포대에서 더 높은 생존률을 나타내는 것을 확인했으므로(ex. childern), Age feature를 유지하고 이를 missing values를 처리하기 위해 이용하는 것이 좋을 것 같다고 판단.

> 이를 위해, Age와 가장 관계가 깊은 feature를 찾기로 하였고, 'sex', 'parch','pclass', 'SibSp'와 비교하여 분포를 확인하였다.

- 그 결과, 
    - 남성과 여성에서 Age 분포가 거의 같게 관찰되었다.
        - 즉, 'sex' feature는 Age를 예측하기에 도움이 되지 않는다는 것을 의미한다.
    - 1st class > 2nd class > 3rd class 순으로 나이대가 높게 분포하는 것을 확인하였다.
    - 승객의 나이가 높을 수록 부모와 자식을 많이 데리고 탑승하고, 승객의 나이가 낮을 수록 형제자매와 배우자를 많이 데리고 탑승한 것을 확인하였다.

> 따라서, missing values를 채우는 전략을 다음과 같이 결정하였다.

> "'pclass', 'parch', 'sibsp' feature가 비슷한 row들의 age 평균으로 채고,  계산해내지 못하는 경우 전체 평균으로 채운다."

### 4. Feature Engineering

#### 4-1. Name / Title

> 저자는 승객의 title 정보 (ex. Mr, Ms ...)는 탈출할 때에 우선권을 가지는 정보가 되었을 수 있으므로, 이를 feature로써 추가하였다.

> 생존률을 출력해보니 "Women and children first"의 경향을 보인 것으로 보아, 유의미한 feature를 만들어 냈다는 것을 알 수 있었다.

#### 4-2. Family Size

> `dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1` 와 같이 feature를 생성해보았다.

> 앞선 분석 결과를 통해 나이가 많을 수록 생존 확률이 낮다는 것을 확인하였는데, 이러한 경향을 나타내는 feature임을 확인하였다.

> 이후, 4가지 family size로 분류하여 feature로 추가하였다.

#### 4-3. Cabin

> Cabin을 가지지 못한 승객은 Cabin 정보가 missing 된 것을 알 수 있다.

> Cabin의 첫번째 문자는 Titanic의 Dest number를 나타낼 가능성이 높은데, 이는 선실의 위치 정보를 포함하고 있으므로 이것만을 추가하는 것으로 저자는 결정하였다.

> 오직 적은 인원 수 만이 cabin을 가지고 있기 때문에, std가 높게 나타나고 다른 desk에 있던 승객과 생존률을 구별해낼 수 없지만, cabin을 가지고 있는 승객이 가지고 있지 못한 승객 보다 생존률이 높다는 것을 확인하였다. (B, C, D, F 에서 확인된다.)

#### 4-4. Ticket

> Ticket이 함께 prefix된 경우 함께 배치된 객실에 예약될 수 있다. 이는 pclass와 생존률이 비슷할 수 있다는 것을 의미한다. 따라서 ticket feature를 prefix 로 바꾸기로 했다. 그게 더 유으미 할 수 있을 것이라고 저자는 예상했다.

### 5. Modeling

#### 5-1. Simple Modeling

1. Cross Validate models
    > 10 가지의 classifiers를 비교하고, 이들을 stratified kfold cross validation을 이용하여 mean accuracy를 평가하는 방식으로 비교하였다.

        - SVC
        - Decision Tree
        - AdaBoost
        - Random Forest
        - Extra Trees
        - Gradient Boosting
        - Multiple layer perceptron (NN)
        - KNN
        - Logistic regression
        - Linear Discriminant Analysis
    > 이를 통해, SVC, AdaBoost, RandomForest, ExtraTress, Gradient Boosting 분류기를 ensemble modeling에 사용하기 로 결정하였다.
2. Hyperparameter tunning for best models
    > 앞서 선정한 model들에 대해 Grid Search optimization을 수행하여 hyperparameter tunning을 수행하였다.

        - 이 작업은 4 cpu로 진행해도 15분이 걸리는 작업이었다....
3. Plot Learning Curves
    > **learning curve는 train set에 대한 overfitting 효과와 train size가 accuracy에 미치는 영향을 확인 할 수 있는 좋은 방법이다!!**

    > GradientBoosting 과 Adaboost 분류기는 train set에 Overfitting되는 경향이 있다. cross validation 곡선이 증가하는 것으로 보아 GradientBoosting과 Adaboost는 더 많은 training example로 더 나은 성능을 발휘할 수 있을 것으로 판단된다.

    > SVC와 ExtraTree 분류기는 Train curve와 cross validation curve가 가까운 듯 보인다. 이는 더 general한 prediction을 도출하는 것으로 이해할 수 있다.
4. Tree based classifier에서 Feature importance 살펴보기
    > 4가지 tree based classifier가 서로 다른 top feature를 보이는 것으로 확인했다. 이는 그들이 서로 다른 feature에 대해 기초하여 prediction되었다는 것을 의미한다.

    > 그럼에도 불구하고, 'Fare', 'Title_2', 'Age', 'Sex'와 같이 공통으로 높은 feature importance를 가지는 경향을 보였다.

    > 이 결과에 대해 저자는 다음과 같이 분석하였다.

        - Pc_1, Pc_2, Pc_3, Fare는 승객의 사회적 지위를 의미한다.
        - Sex, Title_2(Mrs/Mlle/Mme/Miss/Ms), Title_3(Mr)는 성별을 의미한다.
        - Age, Title_1(Master)는 승객의 나이를 의미한다.
        - Fsize, LargeF, MedF, Single은 승객의 가족 규모를 의미한다.
    > 이러한 feature importance를 살펴보았을 때, 생존률을 예측하는 것은 'Age', 'Sex', 'Family size', '승객의 사회적 지위'에 더 관계가 있지, titanic 안에서의 위치와는 상관이 없는 것을 알 수 있다.

    > 각각 분류기의 예측 결과를 이용하여 Heat Map을 찍어본 결과..
    
        - Adaboost가 조금 다른 예측을 내는 것을 제외하고는 5개의 분류기 예측이 매우 유사한 것으로 나타났다.
        - 5개의 분류자는 다소 비슷한 예측 결과를 나타냈지만 다소 차이가 있다. 이러한 예측 사이의 차이점들은 Ensemble voting을 고려할 만하다.

#### 5-2. Ensemble Modeling

> Voting Classifier를 이용하여 models들을 결합하였다. "soft" voting방식을 채택하였다.