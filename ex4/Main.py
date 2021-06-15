from plotnine import *
import ex4_tools
import adaboost
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def Q10_to_Q13(noise_ratio):
    # generate train & test samples:
    train_num_samples = 5000
    train_x, train_y = ex4_tools.generate_data(train_num_samples, noise_ratio)
    test_num_samples = 200
    test_x, test_y = ex4_tools.generate_data(test_num_samples, noise_ratio)

    # train classifier
    T = 500
    ada = adaboost.AdaBoost(ex4_tools.DecisionStump, T)
    D = ada.train(train_x, train_y)

    # compute errors:
    max_t = T
    ada_train_error = [ada.error(train_x, train_y, t) for t in range(max_t + 1)]
    ada_test_error = [ada.error(test_x, test_y, t) for t in range(max_t + 1)]

    # plot of the training error and test error, as a function of T:
    df = pd.DataFrame({'T': range(T + 1), 'error': ada_train_error, 'set': 'train'})
    df = pd.concat([df, pd.DataFrame({'T': range(T + 1), 'error': ada_test_error, 'set': 'test'})])

    # run questions 10-13:
    print("noise ratio is: ", noise_ratio)
    Q10(df, noise_ratio)
    Q11(ada, test_x, test_y, noise_ratio)
    Q12(df, ada, test_x, test_y, noise_ratio, ada_test_error)
    Q13(ada, train_x, train_y, D, T, noise_ratio)


def Q10(df, noise):
    title = "Adaboost's Train/Test Error VS Number Of Iterations.\n noise = " + str(noise)
    p = (ggplot(df) +
         geom_line(aes(x='T', y='error', color='set'), data=df[df.set == 'test'], size=0.5) +
         geom_line(aes(x='T', y='error', color='set'), data=df[df.set == 'train'], size=0.5) +
         scale_linetype_manual(values=['blue', 'red'], labels=['test', 'train']) +
         labs(x="T - iterations", y="Error", title=title))
    ggsave(p, "Q10_noise_" + str(noise) + ".png", verbose=False)
    print(p)


def Q11(ada, test_x, test_y, noise):
    i = 1
    for t in [5, 10, 50, 100, 200, 500]:
        plt.subplot(2, 3, i)
        ex4_tools.decision_boundaries(ada, test_x, test_y, t, 0.5)
        i += 1
    plt.suptitle("Adaboost desicions for different T's. noise = " + str(noise))
    plt.savefig("Q11_noise_" + str(noise) + ".png")
    plt.show()


def Q12(df, ada, test_x, test_y, noise, test_error):
    t = np.argmin(test_error)[0]
    print("T_hat is: ", t)
    print("T_hat's error: ", min(test_error))
    ex4_tools.decision_boundaries(ada, test_x, test_y, t)
    plt.title("Adaboost best classifier desicion.\nnoise = " + str(noise))
    plt.savefig("Q12_noise_" + str(noise) + ".png")
    plt.show()


def Q13(ada, train_x, train_y, D, T, noise):
    D_T = D[T] / np.max(D) * 10
    ex4_tools.decision_boundaries(ada, train_x, train_y, T, D_T)
    plt.title("Adaboost last classifier desicion.\nnoise = " + str(noise))
    plt.savefig("Q13_noise_" + str(noise) + ".png")
    plt.show()


def Q14():
    for noise_ratio in (0.01, 0.4):
        Q10_to_Q13(noise_ratio)


def main():
    Q10_to_Q13(0)
    Q14()


main()
