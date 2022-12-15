import keras
import matplotlib.pyplot as plt
import tensorflow as tf

import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

# print("config:\n", config)

DTYPE = config["DTYPE"]

#################### Utility functions ####################

############## Diverse ##############


def path_join(folder0, folder1):
    if not (len(folder0)):
        return folder1
    if not (len(folder1)):
        return folder0
    else:
        if folder0[-1] == '/' and folder1[0] == '/':
            return folder0+folder1[1:]
        if folder0[-1] == '/' or folder1[0] == '/':
            return folder0+folder1
        return folder0+'/'+folder1


def print_status_bar(iteration, total, mean_losses_and_metrics):
    metrics = " - ".join(['{}: {:.3f}'.format(name, metric.result())
                          for loss_or_metric in mean_losses_and_metrics.keys() for name, metric in mean_losses_and_metrics[loss_or_metric].items()])
    print("{}/{} - ".format(iteration, total)+metrics)


def save_models(epoch, generator, discriminator, dst_folder):
    generator.save_weights(path_join(dst_folder, f'epoch_{epoch}_generator'))
    discriminator.save_weights(
        path_join(dst_folder, f'epoch_{epoch}_discriminator'))


def save_history(history, dst_folder):
    with open(path_join(dst_folder, f'last_epoch_history.yaml'), 'w') as f:
        yaml.dump(history, f, sort_keys=False,
                  default_flow_style=False)

############## Common optimizer ##############


opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)


############## PSNR (Peak Signal to Noise Ratio) ##############

def compute_psnr(original_image, generated_image):
    original_image = tf.convert_to_tensor(original_image, dtype=DTYPE)
    generated_image = tf.convert_to_tensor(generated_image, dtype=DTYPE)

    psnr = tf.image.psnr(original_image, generated_image, max_val=1.0)

    return tf.math.reduce_mean(psnr, axis=None, keepdims=False, name=None)


def plot_psnr(psnr):
    psnr_means = psnr['psnr_quality']
    plt.figure(figsize=(10, 8))

    plt.plot(psnr_means)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title('PSNR')
    plt.show()


############## SSIM - Structural Similarity Index ##############

def compute_ssim(original_image, generated_image):
    original_image = tf.convert_to_tensor(original_image, dtype=tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype=tf.float32)

    ssim = tf.image.ssim(original_image, generated_image,
                         max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, )

    return tf.math.reduce_mean(ssim, axis=None, keepdims=False, name=None)


def plot_ssim(ssim):
    ssim_means = ssim['ssim_quality']

    plt.figure(figsize=(10, 8))
    plt.plot(ssim_means)
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title('SSIM')
    plt.show()


############## Loss functions - perceptual loss ##############

def plot_loss(losses):
    d_loss = losses['d_history']
    g_loss = losses['g_history']

    plt.figure(figsize=(10, 8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.title("Loss")
    plt.legend()
    plt.show()


############## Save images ##############


def save_images(original_image, lr_image, sr_image, path, metrics, epoch):
    """
    Save LR, HR (original) and generated SR images in one panel
    """
    _, ax = plt.subplots(1, 3, figsize=(10, 6))

    images = [lr_image, sr_image, original_image]
    titles = ['LR', f'SR-generated/epoch {epoch}', 'HR']

    for i, img in enumerate(images):
        # (X + 1)/2 to scale back from [-1,1] to [0,1]
        ax[i].imshow((img + 1)/2.0)
        ax[i].axis("off")
        ax[i].set_title('{}'.format(titles[i]))

    if 'psnr' in metrics.keys() and 'ssim' in metrics.keys():
        psnr = metrics['psnr']
        ssim = metrics['ssim']
        ax[1].text(0.5, -0.1, 'psnr: %3.3f, ssim: %3.3f' % (psnr, ssim),
                   verticalalignment='bottom', horizontalalignment='center',
                   transform=ax[1].transAxes,
                   color='black', fontsize=15)
    else:
        print("metrics is missing psnr and ssim")

    plt.savefig(path)
    plt.close()
