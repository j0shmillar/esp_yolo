# https://www.researchgate.net/figure/YOLOv5-architecture-The-YOLO-network-consists-of-three-main-parts-Backbone-Neck-and_fig5_355962110
import time
import tensorflow as tf

class CBL(tf.keras.layers.Layer):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride ):
        super(CBL, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform()

        self.act1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv2D(mid_channels, kernel_size, strides=stride, padding='same', use_bias=False, kernel_initializer=initializer)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.act2 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(out_channels, kernel_size, strides=stride, padding='same', use_bias=False, kernel_initializer=initializer)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return self.maxpool(x)

class YOLO(tf.keras.Model):
    def __init__(self, first_out, nc=80, anchors=(), grid_size=4, inference=False):
        super(YOLO, self).__init__()
        self.inference = inference
        self.grid_size = grid_size
        self.nc = nc  # number of classes
        self.num_boxes = len(anchors)  # number of detection layers

        self.backbone = tf.keras.Sequential()
        self.backbone.add( CBL(in_channels=3, mid_channels=first_out, out_channels=first_out, kernel_size=3, stride=1) )
        self.backbone.add( CBL(in_channels=first_out, mid_channels=first_out, out_channels=first_out*2, kernel_size=3, stride=1) )
        self.backbone.add( CBL(in_channels=first_out*2, mid_channels=first_out*2, out_channels=first_out*4, kernel_size=3, stride=1) )
        self.backbone.add( CBL(in_channels=first_out*4, mid_channels=first_out*4, out_channels=first_out*4, kernel_size=3, stride=1) )
        self.backbone.add( CBL(in_channels=first_out*4, mid_channels=first_out*8, out_channels=first_out*8, kernel_size=3, stride=1) )

        self.head = tf.keras.Sequential()
        self.head.add( tf.keras.layers.Flatten() )
        self.head.add( tf.keras.layers.Dense(256, activation='relu') )
        self.head.add( tf.keras.layers.Dense( grid_size*grid_size*self.num_boxes *(5 + self.nc) ) )
    
    def call(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = tf.reshape(x, (-1, self.grid_size*self.grid_size, self.num_boxes, 5 + self.nc))
        x = tf.sigmoid(x)

        return x

if __name__ == "__main__":
    batch_size = 32
    image_height = 88
    image_width = 88
    nc = 4
    anchors = [ [(11,8), (25,18), (46,43)]]
    num_boxes = len(anchors)

    x = tf.random.uniform((batch_size, image_height, image_width, 3), dtype=tf.float32)
    print(f'x type = {type(x)}')
    first_out = 16

    model = YOLO(first_out=first_out, nc=nc, anchors=anchors, inference=False)

    model.build(input_shape = x.shape)
    model.summary()

    start = time.time()
    out = model(x)
    end = time.time()
    assert out.shape == (batch_size, 4*4, num_boxes, 5 + nc)

    print("Success!")
    print("feedforward took {:.2f} seconds".format(end - start))


