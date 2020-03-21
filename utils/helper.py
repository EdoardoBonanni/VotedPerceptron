import numpy as np

# sprite is a computer graphic that may be moved on-screen
def create_sprite_image(images):
    # Returns a sprite image consisting of images passed as argument. Images should be count x width x height
    # isistance() is a function that allows to control if the specified object is of the specified type
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # numpy ceil() returns the whole upper part of a number
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))
    # create the image element by element
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img
    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    # Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    # Makes black white, and white black
    return 255 - mnist_digits


def get_sprite_image(to_visualise, do_invert=True):
    # function that allows you to view the requested object
    to_visualise = vector_to_matrix_mnist(to_visualise)
    if do_invert:
        to_visualise = invert_grayscale(to_visualise)
    return create_sprite_image(to_visualise)
