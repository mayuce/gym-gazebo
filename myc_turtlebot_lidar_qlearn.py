#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy
import random
import time
import liveplot
import myc_qlearn as qlearn

def render():
    render_skip = 0
    render_interval = 50
    render_episodes = 10

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':

    # Gazebo simulasyonunda 'GazeboCircuitTurtlebotLidar-v0' evreni yukenir.
    env = gym.make('GazeboCircuitTurtlebotLidar-v0')

    # Sonuclarin tutuldugu klasor gosterilir
    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)
    last_time_steps = numpy.ndarray(0)

    # Q Learn ilk deger atamalari yapilir.
    qlearn = qlearn.QLearn(actions=range(env.action_space.n), alpha=0.1, gamma=0.8, epsilon=0.9)
    initial_epsilon = qlearn.epsilon
    epsilon_discount = 0.999
    start_time = time.time()
    total_episodes = 2000 # kac kez tekrarlanacagi
    highest_reward = 0

    # Burada total episode sayisi kadar robot cevre sifirlaranarak aksiyon secer ve bu aksiyon uzerinden harita uzerinde gezmeye baslar
    # Her bolum robot bir yere carpincaya kadar veya hedefe ulasana kadar devam eder
    # Bolum sonunda robotun hareketine gore odul degeri belirlenir ve Q matrisine yazilir.
    # Bu sonuclar sonucunda robot haritadaki hedeflere nasil gidecegini ve hangi noktalar uzerinden gidebilecegini ogrenir.


    for x in range(total_episodes):
        observation = env.reset()
        state = ''.join(map(str, observation))
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        done = False
        cumulated_reward = 0
        for i in range(500):
            # Su anki durumdan bir durum secer
            action = qlearn.chooseAction(state)
            # Aksiyon calistirilir ve geri donus atamalari yapilir aksiyona gore
            observation, reward, done, info = env.step(action)

            # odul kumulatif odule eklenir
            cumulated_reward += reward

            # eger kumulatif odul en yusek odulden yuksek ise en yuksek odul kumlatif odul degerini alir
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            # Bir sonraki durum harita uzerindeki gozlemden secilir
            nextState = ''.join(map(str, observation))

            # Bir sonraki durum icin ogrenme fonksiyonu calistirilir
            qlearn.learn(state, action, reward, nextState)


            # Evren temizlenir
            env._flush(force=True)

            # Eger dongu tamamlanmaz ise su anki durum sonraki durumun degerini alir ve dongu tekrarlanir
            # eger dongu tamamlanirsa son adim degiskeni atanir.
            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        # X bolumu 100'un katsayisi ise cikti ekranina sonuc basilir.
        if x % 100 == 0:
            plotter.plot(env)

        # Bolum ile ilgili bilgiler terminal ekranina yazilir.
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    l = last_time_steps.tolist()
    l.sort()


    #Tum skor ve en iyi 100 skor terminal ekranina yazilir.
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # Evren kapatilir.
    env.close()
