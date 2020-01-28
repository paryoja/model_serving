import matplotlib.pyplot as plt
import numpy as np


def show_single_image(img, title, recover_func=None):
    plt.figure()
    plt.title(title)
    # print(type(img[0]))
    # print(dir(img[0]))
    if recover_func:
        recovered = recover_func(img[0])
        # print(recovered)
        plt.imshow(np.uint8(recovered))
    else:
        plt.imshow(img[0])
    # plt.title(class_names[label_batch[n] == True][0].title())
    plt.axis('off')

    plt.show()


def show_images(image_batch, title="Train", recover_func=None):
    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize="x-large")
    n = 0
    for img in image_batch:
        ax = plt.subplot(5, 5, n + 1)
        # print(type(img[0]))
        # print(dir(img[0]))
        if recover_func:
            recovered = recover_func(img[0])
            # print(recovered)
            plt.imshow(np.uint8(recovered))
        else:
            plt.imshow(img[0])
        # plt.title(class_names[label_batch[n] == True][0].title())
        plt.axis('off')
        n += 1
    plt.show()
