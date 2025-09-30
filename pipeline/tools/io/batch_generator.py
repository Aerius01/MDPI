def batch_generator(images, images_mean, nr_images, batch_size, index):
    if index + batch_size > nr_images:
        x_batch = (images[index:] - images_mean) / 255.
        return x_batch
    else:
        end = index + batch_size
        x_batch = (images[index:end] - images_mean) / 255.
        return x_batch
