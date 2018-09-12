from keras.models import Model
from keras.layers import *

def CVN_model(num_classes):
    def inception(x, name='Inception',num_filters=[16,32,96,128,32,64]):
        branch11 = Conv2D(num_filters[0], (1, 1), activation='relu', padding='same', name=name+'_branch11')(x)
        branch11 = Conv2D(num_filters[1], (5, 5), activation='relu', padding='same', name=name+'_branch12')(branch11)
        branch12 = Conv2D(num_filters[2], (1, 1), activation='relu', padding='same', name=name+'_branch21')(x)
        branch12 = Conv2D(num_filters[3], (3, 3), activation='relu', padding='same', name=name+'_branch22')(branch12)
        branch13 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=name+'_branch3mp')(x)
        branch13 = Conv2D(num_filters[4], (1, 1), padding='same', activation='relu', name=name+'_branch31')(branch13)
        branch14 = Conv2D(num_filters[5], (1, 1), padding='same', activation='relu', name=name+'_branch32')(x)

        x = Concatenate(axis=-1, name=name+'_Concatenate')([branch11, branch12, branch13, branch14])
        return x

    def subnet(x,name):
        x = Conv2D(64, (7, 7), activation='relu', strides=(2, 2), name=name + 'Conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=name + 'Pool1')(x)
        x = Conv2D(64, (1, 1), activation='relu', name=name + 'Conv2')(x)
        x = Conv2D(192, (3, 3), activation='relu', name=name + 'Conv3')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=name + 'Pool2')(x)

        x = inception(x, name=name + 'Inception')

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=name + 'Pool3')(x)
        return x

    inputx = Input(shape=(80,100,1),name='inputx')
    inputy = Input(shape=(80,100,1),name='inputy')

    x = subnet(inputx,'x')
    y = subnet(inputy,'y')

    merge = Concatenate(axis=-1)([x, y])
    merge = inception(merge,name='mergeInception',num_filters=[48,128,192,384,128,384])
    merge = AveragePooling2D(pool_size=(3, 4))(merge)
    merge = Flatten()(merge)
    merge = Dropout(0.4)(merge)
    merge = Dense(1024,activation='relu')(merge)
    out = Dense(num_classes,activation='softmax',name='out')(merge)

    model = Model(inputs=[inputx,inputy],outputs=out)

    return model
