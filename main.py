import numpy as np
from layers import FCLayer, ActivationLayer, Network
from activation_functions import relu, relu_prime, tanh, tanh_prime
from loss_functions import cross_entropy_softmax, cross_entropy_softmax_prime, mean_square_error, mean_square_error_prime
from keras.datasets import mnist
from keras.utils import np_utils

import time
import sys
import tkinter

def tanh_tanh_tanh_mse_network():
    # The network provided from the site where I stole all this code from: Omar Aflak, https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
    net = Network()
    net.add(FCLayer(784, 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 50))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(50, 10))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.use(mean_square_error, mean_square_error_prime)
    return net

# sample: 1000, epoch: 35
# 1. accuracy: 79.90%, time: 42.4911s
# 2. accuracy: 81.13%, time: 42.3893s
# 3. accuracy: 79.65%, time: 42.3747s

# sample: 2000, epoch: 35
# 1. accuracy: 86.61%, time: 84.7035s
# 2. accuracy: 85.04%, time: 85.1274s
# 3. accuracy: 84.22%, time: 84.0151s

# sample: 4000, epoch: 35
# 1. accuracy: 88.95%, time: 168.2937s
# 2. accuracy: 89.42%, time: 172.6208s
# 3. accuracy: 88.30%, time: 169.3265s

# sample: 8000, epoch: 35
# 1. accuracy: 91.50%, time: 335.0033s
# 2. accuracy: 91.09%, time: 337.2939s
# 3. accuracy: 91.56%, time: 333.9224s

# sample: 16000, epoch: 35
# 1. accuracy: 93.28%, time: 674.4695s
# 2. accuracy: 93.56%, time: 673.6062s
# 3. accuracy: 93.23%, time: 667.7904s

def relu_relu_softmax_cross_entropy_network():
    # Network that almost every other tutorial uses
    net = Network()
    net.add(FCLayer(784, 100))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(FCLayer(100, 50))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(FCLayer(50, 10))
    #softmax is supposed to be the last layer, but I combined it with the loss function
    net.use(cross_entropy_softmax, cross_entropy_softmax_prime)
    return net

# sample: 1000, epoch: 35
# 1. accuracy: 78.39%, time: 44.8030s
# 2. accuracy: 79.67%, time: 44.7586s
# 3. accuracy: 79.22%, time: 44.2722s

# sample: 2000, epoch: 35
# 1. accuracy: 85.27%, time: 91.7738s
# 2. accuracy: 85.00%, time: 89.7378s
# 3. accuracy: 86.01%, time: 90.1282s

# sample: 4000, epoch: 35
# 1. accuracy: 88.31%, time: 181.8210s
# 2. accuracy: 88.45%, time: 177.0105s
# 3. accuracy: 88.66%, time: 177.7447s

# sample: 8000, epoch: 35
# 1. accuracy: 90.98%, time: 353.3294s
# 2. accuracy: 91.83%, time: 356.6858s
# 3. accuracy: 91.32%, time: 358.8333s

# sample: 16000, epoch: 35
# 1. accuracy: 93.70%, time: 709.4794s
# 2. accuracy: 94.13%, time: 715.7597s
# 3. accuracy: 94.37%, time: 705.6836s

# sample: 32000, epoch: 35
# 1. accuracy: 95.38%, time: 1421.4080s

def get_data():
    # Most of the code was from: Omar Aflak, https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_train = x_train.astype('float')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_test = x_test.astype('float')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)

def train():
    target_network = sys.argv[2]
    output_file = sys.argv[3]
    start_index = int(sys.argv[4])
    end_index = int(sys.argv[5])
    epochs = int(sys.argv[6])
    learning_rate = float(sys.argv[7])

    net = globals()[target_network]()

    (x_train, y_train), (x_test, y_test) = get_data()

    start_time = time.time()
    net.train(x_train[start_index:end_index], y_train[start_index:end_index], epochs, learning_rate)
    elapsed_time = time.time() - start_time
    print(f"time={elapsed_time:.4f}")

    net.save(output_file)

def predict():
    target_network = sys.argv[2]
    input_file = sys.argv[3]
    start_index = int(sys.argv[4])
    end_index = int(sys.argv[5])

    net = globals()[target_network]()
    net.load(input_file)

    (x_train, y_train), (x_test, y_test) = get_data()

    matches = 0
    samples = end_index - start_index
    for i in range(start_index, end_index):
        guess = np.argmax(net.predict(x_test[i]))
        actual = np.argmax(y_test[i])
        print(f"guess={guess}, actual={actual}, match={guess==actual}")
        if actual == guess:
            matches += 1
    print(f"accuracy={matches * 100/samples:.2f}%")


def draw():
    def increase_thickness(event):
        if event.char != 't': return

        nonlocal thickness
        nonlocal max_thickness
        thickness += 1
        if thickness > max_thickness:
            thickness = max_thickness

        thickness_label.config(text=f"THICKNESS: {thickness}")


    def decrease_thickness(event):
        if event.char != 'r': return

        nonlocal thickness
        nonlocal min_thickness
        thickness -= 1
        if thickness < min_thickness:
            thickness = min_thickness
        
        thickness_label.config(text=f"THICKNESS: {thickness}")

    def switch_erasing(event):
        if event.char != 'e': return

        nonlocal is_erasing
        is_erasing = not is_erasing
        erasing_label.config(text=f"IS ERASING: {is_erasing}")

    def turn_on_draw(event):
        nonlocal hold_down
        hold_down = True
    
    def turn_off_draw(event):
        nonlocal hold_down
        hold_down = False

    def draw_on_canvas(event):
        nonlocal hold_down
        if not hold_down: return
        
        x_index = event.x // 10
        y_index = event.y // 10

        nonlocal thickness
        start_x_index = x_index - (thickness // 2)
        start_y_index = y_index - (thickness // 2)

        for y_index in range(start_y_index, start_y_index + thickness):
            for x_index in range(start_x_index, start_x_index + thickness):
                if y_index >= 0 and y_index < 28 and x_index >= 0 and x_index < 28:
                    x_coord = x_index * 10
                    y_coord = y_index * 10
                    nonlocal is_erasing
                    nonlocal saved_points
                    if not is_erasing:
                        canvas.create_rectangle(x_coord, y_coord, x_coord + 10, y_coord + 10, fill="white")
                        saved_points[y_index][x_index] = 1
                    else:
                        # This is not great, but I rarely need to erase, soooo.....
                        # enjoy a laggy experience until you clear :D
                        canvas.create_rectangle(x_coord, y_coord, x_coord + 10, y_coord + 10, fill="black")
                        saved_points[y_index][x_index] = 0

    def predict_from_canvas(event):
        if event.char != 'p': return

        x = saved_points.flatten()
        predictions = net.predict(x)
        guess = np.argmax(predictions)

        predict_label.config(text=f"PREDICTION: {guess}")
        print(saved_points)
        print(predictions)
        print(guess)

    def clear_canvas(event):
        if event.char != 'c': return

        canvas.delete("all")
        nonlocal saved_points
        saved_points = np.zeros((28, 28))
        
    target_network = sys.argv[2]
    input_file = sys.argv[3]

    net = globals()[target_network]()
    net.load(input_file)

    is_erasing = False
    hold_down = False
    saved_points = np.zeros((28, 28))
    thickness = 2
    max_thickness = 4
    min_thickness = 1

    root = tkinter.Tk()

    root.title("Predict Digit Draw")

    controls_label = tkinter.Label(root, text="CONTROLS\nMouse-1: Draw, Mouse-1 Release: Stop Draw\nT: increase thickness, R: decrease thickness\nC: Clear, E: switch erase/draw, P: predict")
    controls_label.pack()

    erasing_label = tkinter.Label(root, text=f"IS ERASING: {is_erasing}")
    erasing_label.pack()

    thickness_label = tkinter.Label(root, text=f"THICKNESS: {thickness}")
    thickness_label.pack()

    predict_label = tkinter.Label(root, text="PREDICTION: ?", fg='red')
    predict_label.pack()

    canvas = tkinter.Canvas(root, bg="black", height=280, width=280)
    canvas.pack()

    root.bind("<ButtonPress-1>", lambda e: (turn_on_draw(e), draw_on_canvas(e)))
    root.bind("<ButtonRelease-1>", turn_off_draw)
    root.bind("<Motion>", draw_on_canvas)
    root.bind("<Key>", lambda e: (predict_from_canvas(e), clear_canvas(e), increase_thickness(e), decrease_thickness(e), switch_erasing(e)))

    root.mainloop()

if __name__ == "__main__":
    
    if sys.argv[1] == "train": 
        # 1. target_network
        # 2. output_file
        # 3. start_index
        # 4. end_index
        # 5. epochs
        # 6. learning_rate
        train()
    elif sys.argv[1] == "predict": 
        # 1. target_network
        # 2. input_file
        # 3. start_index
        # 4. end_index
        predict()
    elif sys.argv[1] == "draw": 
        # 1. target_network
        # 2. input_file
        draw()
    else:
        print("You did not provide a valid command!")
        
