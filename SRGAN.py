
#################### LIBRAIRIES ####################
import keras
import tensorflow as tf
# common optimizer to all networks
from utils import opt, compute_psnr, compute_ssim, print_status_bar, save_images, path_join, save_models, save_history
import wandb

#################### CONFIG ####################
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)
# print("config:\n", config)
print()

#################### SET GPU ####################
print("tf.__version__:", tf.__version__)

physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(physical_devices))
# print(device_lib.list_local_devices())
tf.config.experimental.set_memory_growth(physical_devices[0], True)


DTYPE = config["DTYPE"]
tf.keras.backend.set_floatx(DTYPE)


#################### Discriminator ####################
def myLeakyReLU(z):
    return tf.keras.activations.relu(z, alpha=0.2, max_value=None, threshold=0.0)


def get_discriminator(input_shape):
    # TO FILL
    discriminator = tf.keras.Sequential([
        ...  # TO FILL
    ], name="discriminator")

    n_blocks = config['MODEL']['DISCRIMINATOR']['N_BLOCKS']
    n_filters = config['MODEL']['DISCRIMINATOR']['N_FILTERS_START']
    for i in range(n_blocks):
        n_filters = ...  # TO FILL
        stride_unit = ...  # TO FILL
        # add convolution, batch normalization and myLeakyReLU layers
        discriminator.add(...)  # TO FILL

    # add the head of the discriminator
    discriminator.add(...)  # TO FILL

    return discriminator


input_shape = config['HIGH_RESOLUTION_SHAPE']


#################### Generator ####################
class ResidualBlock(keras.layers.Layer):
    # understand it for coding get_generator
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(
                filters, 3, strides=strides, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(
                filters, 3, strides=strides, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
        ]
        self.skip_layers = []

        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(
                    filters, 1, strides=strides, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z+skip_Z)


def get_generator(input_shape):
    # # TO FILL
    input = tf.keras.layers.Input(shape=input_shape)

    start = tf.keras.Sequential([
        ...  # TO FILL
    ], name='start')

    residual_blocks = tf.keras.Sequential([])
    # TO FILL

    out = tf.keras.Sequential([
        ...  # TO FILL
    ])

    after_start = start(input)
    A1 = residual_blocks(after_start)
    A2 = tf.keras.layers.Add()([A1, after_start])
    A2 = tf.keras.layers.PReLU()(A2)
    output = out(A2)

    return tf.keras.Model(inputs=[input], outputs=[output], name="generator")


def test_get_generator():
    input_shape = config["LOW_RESOLUTION_SHAPE"]
    gen = get_generator(input_shape)


#################### VGG19 ####################


def get_VGG19(input_shape):
    # the download takes time the first time only
    VGG19_base = tf.keras.applications.vgg19.VGG19(
        weights="imagenet", include_top=False, input_shape=input_shape)
    # block5_conv2 as the SRGAN paper 2017 suggests
    VGG19_base.outputs = [VGG19_base.get_layer('block5_conv2').output]
    model = tf.keras.Model(inputs=[VGG19_base.inputs], outputs=[
                           VGG19_base.outputs], name="VGG19")

    return model


def test_get_VGG19():
    VGG19_base = get_VGG19(config["HIGH_RESOLUTION_SHAPE"])
    VGG19_base.trainable = False

    VGG19_base.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    VGG19_base.summary()


#################### SRGAN ####################
# Final Adversarial Network
def get_adversarial_model(generator, discriminator, feature_extractor):
    input_high_resolution = tf.keras.layers.Input(
        shape=config["HIGH_RESOLUTION_SHAPE"])  # only to fit the dataset format
    input_low_resolution = tf.keras.layers.Input(
        shape=config["LOW_RESOLUTION_SHAPE"])

    generated_high_resolution_images = generator(input_low_resolution)

    features = feature_extractor(generated_high_resolution_images)

    # make a discriminator non trainable
    discriminator.trainable = False
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=opt, metrics=['accuracy'])

    # discriminator will give a prob estimation for generated high-resolution images
    probs = discriminator(generated_high_resolution_images)

    # create and compile
    adversarial_model = tf.keras.Model(
        [input_low_resolution, input_high_resolution], [probs, features], name='SRGAN')
    # not needed... useless thx to my_train_on_batch
    adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[
                              1e-3, 1], optimizer=opt)

    return adversarial_model


def test_set_adversarial_model():
    print("\nusing the function:")
    gen = get_generator(config["LOW_RESOLUTION_SHAPE"])
    dis = get_discriminator(config["HIGH_RESOLUTION_SHAPE"])
    fe_extr = get_VGG19(config["HIGH_RESOLUTION_SHAPE"])
    adv = get_adversarial_model(gen, dis, fe_extr)


class SRGAN(keras.Model):
    def __init__(self, generator, discriminator, feature_extractor):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.feature_extractor = feature_extractor

    def compile(self, loss, loss_weight, optimizer, metrics=[]):
        '''
        Args:
        -loss (list of len 2 of str): [discriminator loss identifier, feature extractor loss identifier]
        -loss_weight (list of len 2)
        -optimizer (function)
        -metrics (list of fct)
        '''
        super().compile()
        assert len(
            loss) == 2, "loss of wrong length. Only the two first elements will be used."
        self.loss = [tf.keras.losses.get(loss[i]) for i in range(len(loss))]
        assert len(loss) == len(
            loss_weight), "loss and loss_weight have not the same length."
        self.loss_weight = loss_weight
        self.optimizer = optimizer
        # make metrics more flexible
        # self.myMetrics = metrics

    @tf.function
    def train_step(self, lr_images, hr_images):
        # TODO
        batch_size = tf.shape(lr_images)[0]

        # print("train step")
        # print("lr_images:", lr_images)

        # Decode them to fake images
        generated_high_resolution_images = self.generator(lr_images)
        # print(generated_high_resolution_images[0].numpy())

        # # Combine them with real images
        # combined_images = tf.concat(
        #     [generated_high_resolution_images, hr_images], axis=0)

        # generate a batch of true and fake labels
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        if config["LABEL_NOISE"]:
            # Add random noise to the labels - important trick!
            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
            fake_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))

        # Train the discriminator on real high-resolution images
        with tf.GradientTape() as tape:
            ...  # TO FILL
        ...
        d_loss_real = tf.math.reduce_mean(d_loss_real)

        # Train the discriminator on fake high-resolution images
        with tf.GradientTape() as tape:
            ...  # TO FILL
        ...
        d_loss_fake = tf.math.reduce_mean(d_loss_fake)

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!

        # deceive the discriminator
        with tf.GradientTape() as tape:
            ...  # TO FILL
        ...
        g_loss_discr = tf.math.reduce_mean(g_loss_discr)

        # deceive the feature extractor
        # extract feature maps for true high-resolution images
        hr_features = self.feature_extractor(hr_images)

        with tf.GradientTape() as tape:
            ...  # TO FILL
        ...
        g_loss_fe = tf.math.reduce_mean(g_loss_fe)

        psnr = compute_psnr(hr_images, generated_high_resolution_images)
        ssim = compute_ssim(hr_images, generated_high_resolution_images)

        return {'losses': {"d_loss_real": d_loss_real, "d_loss_fake": d_loss_fake, "g_loss_discr": g_loss_discr, "g_loss_fe": g_loss_fe}, 'metrics': {'psnr': psnr, 'ssim': ssim}}

    def fit(self, dataset, epochs, validation_dataset, steps_per_epoch=None, save_val_period=None, dst_folder='.', use_wandb=False):
        if steps_per_epoch == None:
            steps_per_epoch = len(dataset)
        else:
            steps_per_epoch = min(len(dataset), steps_per_epoch)

        if save_val_period != None:
            for val_lr_image_batch, val_hr_image_batch in validation_dataset.take(1):
                break

        mean_losses_and_metrics = {
            "losses": {
                "d_loss_real": keras.metrics.Mean(),
                "d_loss_fake": keras.metrics.Mean(),
                "g_loss_discr": keras.metrics.Mean(),
                "g_loss_fe": keras.metrics.Mean()
            },
            "metrics": {
                "psnr": keras.metrics.Mean(),
                'ssim': keras.metrics.Mean()
            }
        }

        history = {}
        last_values = {}
        for first_key, dic in mean_losses_and_metrics.items():
            history[first_key] = {
                metric_name: [] for metric_name in dic.keys()}
            last_values[first_key] = {
                metric_name: 0 for metric_name in dic.keys()}

        print("begin training...")
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            step = 1
            for lr_images, hr_images in dataset.take(steps_per_epoch):
                batch_history = self.train_step(lr_images, hr_images)
                for first_key, dict in batch_history.items():
                    for second_key, value in dict.items():
                        mean_losses_and_metrics[first_key][second_key](
                            value)
                print_status_bar(
                    step*config["BATCH_SIZE"], steps_per_epoch*config["BATCH_SIZE"], mean_losses_and_metrics)
                step += 1

            for first_key, dict in mean_losses_and_metrics.items():
                for metric_name, metric in dict.items():
                    res = metric.result()
                    history[first_key][metric_name].append(res)
                    last_values[first_key][metric_name] = res
                    metric.reset_states()

            save_history(history, dst_folder)

            if use_wandb:
                wandb.log({"epoch": epoch})
                wandb.log(last_values)

            if save_val_period != None and epoch % save_val_period == save_val_period-1:
                generated_images = self.generator.predict_on_batch(
                    val_lr_image_batch)
                for i, img in enumerate(generated_images):
                    psnr = compute_psnr(val_hr_image_batch[i], img)
                    ssim = compute_ssim(val_hr_image_batch[i], img)
                    save_images(val_hr_image_batch[i], val_lr_image_batch[i], img, path=path_join(
                        dst_folder, f"/img_{epoch}_{i}"), metrics={"psnr": psnr, "ssim": ssim}, epoch=epoch)

                # save_models(epoch, self.generator,
                #             self.discriminator, dst_folder)

                if use_wandb:
                    for i in range(config["VAL_BATCH_SIZE"]):
                        wandb.log({"validation": wandb.Image(path_join(
                            dst_folder, f"/img_{epoch}_{i}.png"))})
        print("training is over")
        return history


def test_final():
    gen = get_generator(config["LOW_RESOLUTION_SHAPE"])
    dis = get_discriminator(config["HIGH_RESOLUTION_SHAPE"])
    fe_extr = get_VGG19(config["HIGH_RESOLUTION_SHAPE"])
    adv_model = SRGAN(gen, dis, fe_extr)
    adv_model.compile(loss=["binary_crossentropy", "mse"],
                      loss_weight=[1e-3, 1], optimizer=opt)
