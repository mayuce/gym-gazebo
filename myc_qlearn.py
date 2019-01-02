import random

class QLearn:
    # Constructor yapisi - Actions degiskeni var olan aksiyonlari alir, Epsilon gozlemleme degerini alir.
    # Epsilon degiskeni robotun daha fazla yol kesfetmek icin ne kadar random deger alacagidir.
    # Alpha ogrenme oranini ve Gamma ise gelecek odul tahminlerine ne kadar onem verilecegi degerini alir.
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    # istenilen durumdaki aksiyonlari getiren fonksiyon
    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)


    # Q-learn Bellman denkleminin fonksiyonu: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))  denklemi ile ogrenmeyi saglar
    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        # Eger eski deger bos ise kazanci yerine yazar. Eger dolu ise eski deger ile yeni degeri ogrenme katsayisi ile carpar ve yerine yazar.
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)


    def chooseAction(self, state):
        # Verilen durumda en yuksek odul degerine sahip aksiyon seciliyor
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        # random bir sayi seciliyor ve eger epsilon degerinden kucukse en kucuk odul degerine sahip deger q matirisinden aliniyor
        # min ve max Q degerlerinden yuksek olanin mutlak degeri seciliyor ve bu deger ile aksiyonlara random degerler eklenerek tekrar maxQ degeri hesaplaniyor
        # Bunun amaci araya stokastik degerler katmaktir. Bu da sinif olusturulurken verilen epsilon degeri ile ne kadar katilacagi ayarlanir.
        # Epsilon degerinin artmasi ile daha fazla yol kesfedilir ve robot daha iyi ogrenme saglar.
        if random.random() < self.epsilon:
            minQ = min(q);
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        # birden fazla maxQ degeri olabilmekte bazen bunlarin sayisi aliniyor.
        # eger birden fazla ise deger aksiyonlar arasindan rastgele bir tanesi seciliyor
        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        # secilen aksiyon geri donuduruluyor
        # Buradaki amac hedefe giden yola en yakin aksiyonu secip, uygulamaktir.
        action = self.actions[i]
        return action

    # Verilen durum aksiyon odul ve sonraki adim icin
    # sonraki adimdaki en yuksek Q matrisindeki deger secilir ve o durum icin gamma degeriyle carpilarak Q matrisine eklenir.
    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)