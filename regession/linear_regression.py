# _*_coding:utf-8_*_
import random
import matplotlib.pyplot as plt


def gen_data(k=2, b=1):
    y_true = []
    y_gen = []
    x_true = [x for x in range(10)]
    for x in range(10):
        y_true1 = k * x + b
        y_true.append(y_true1)
        # change = random.randint(-2, 2)
        plus_minus = random.choice([2, -2])
        change = plus_minus * random.random()
        y_gen1 = y_true1 + change
        y_gen.append(y_gen1)
    return x_true, y_gen, y_true


def optimizer(data, init_b, init_k, learning_rate, num_iter):
    b = init_b
    k = init_k
    plt.ion()
    x, y_gen, y_true = gen_data()
    for i in range(num_iter):
        b, k = compute_gradient(b, k, data, learning_rate)
        if i % 100 == 0:
            plt.cla()
            plt.ylim((0, 20))
            plt.xlim((0, 10))
            print(i, computer_loss(b, k, data))  # 损失函数

            x = [x for x in range(10)]
            y = []
            for x1 in range(10):
                y1 = k * x1 + b
                y.append(y1)
            plt.text(7, 3, 'k:%.2f' % k,fontdict={'size': 15})
            plt.text(7, 2, 'b:%.2f' % b, fontdict={'size': 15})
            plt.plot(x, y)
            plt.scatter(x, y_gen)
            plt.draw()
            # if glo == 1:
            #     time.sleep(2)
            #     glo += 1
            plt.pause(0.1)
    plt.ioff()
    plt.show()
    return [b, k]


def compute_gradient(old_b, old_k, data, learning_rate):
    b_gradient = 0
    k_gradient = 0

    N = float(10)

    # 偏导数， 梯度
    for i in range(10):
        x = data[0][i]
        y = data[1][i]

        b_gradient += -(2 / N) * (y - ((old_k * x) + old_b))
        k_gradient += -(2 / N) * x * (y - ((old_k * x) + old_b))  # 偏导数

    new_b = old_b - (learning_rate * b_gradient)
    new_k = old_k - (learning_rate * k_gradient)
    return [new_b, new_k]


def computer_loss(b, k, data):
    loss = 0
    x = data[0]
    y = data[1]
    for i in range(10):
        x1 = x[i]
        y1 = y[i]
        loss += (y1 - k * x1 - b) ** 2
    return loss / 10


def linear_regression():
    learning_rate = 0.0001
    init_b = 0
    init_k = 0
    num_iter = 3000
    data = gen_data()
    [b, k] = optimizer(data, init_b, init_k, learning_rate, num_iter)
    return b, k


if __name__ == '__main__':
    b, k = linear_regression()
    print('k:', k, 'b:', b)
