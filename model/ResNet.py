from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model, Input

#%% 
"""""""""""""""""""""""""""""""""""""""""
    previous ResNet for #layers <= 34

"""""""""""""""""""""""""""""""""""""""""

def residual_unit_previous(inputs, filters=64, initializer='glorot_uniform', regularizer=l2(0.01), downsample=False, name="residual_unit"):
    if downsample == False:
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(inputs)
        residual = inputs
        
    else:
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(inputs)
        
        residual = Conv2D(filters=filters, kernel_size=(1,1), strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv_resid".format(name))(inputs)
        residual = BatchNormalization(name="{}_bn_resid".format(name))(residual)
        residual = Activation('relu', name="{}_act_resid".format(name))(residual)
    
    
    x = BatchNormalization(name="{}_bn1".format(name))(x)
    x = Activation('relu', name="{}_act1".format(name))(x)
    
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv2".format(name))(x) 
    x = BatchNormalization(name="{}_bn2".format(name))(x)
    
    x = Add(name="{}_add".format(name))([x, residual])

    return Activation('relu', name="{}_act2".format(name))(x)



"""""""""""""""""""""""""""""""""""
    default as ResNet34

"""""""""""""""""""""""""""""""""""

def ResNet_previous(inputs, layers=(3,4,6,3), filters=(64,128,256,512), num_class=1000, initializer='glorot_uniform', l2_weight=0.05):
    x = Conv2D(filters=filters[0], kernel_size=(7,7), strides=(2,2), kernel_initializer=initializer, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    for num in range(layers[0]):
        x = residual_unit_previous(x, filters=filters[0], initializer=initializer, regularizer=l2(l2_weight), downsample=False, name="resid1_{}".format(num+1))

    for num in range(layers[1]):
        if num == 0:
            x = residual_unit_previous(x, filters=filters[1], initializer=initializer, regularizer=l2(l2_weight), downsample=True, name="resid2_{}".format(num+1))
           
        else:
            x = residual_unit_previous(x, filters=filters[1], initializer=initializer, regularizer=l2(l2_weight), downsample=False, name="resid2_{}".format(num+1))
           
    for num in range(layers[2]):
        if num == 0:
            x = residual_unit_previous(x, filters=filters[2], initializer=initializer, regularizer=l2(l2_weight), downsample=True, name="resid3_{}".format(num+1))
           
        else:
            x = residual_unit_previous(x, filters=filters[2], initializer=initializer, regularizer=l2(l2_weight), downsample=False, name="resid3_{}".format(num+1))
            
    for num in range(layers[3]):
        if num == 0:
            x = residual_unit_previous(x, filters=filters[3], initializer=initializer, regularizer=l2(l2_weight), downsample=True, name="resid4_{}".format(num+1))
            
        else:
            x = residual_unit_previous(x, filters=filters[3], initializer=initializer, regularizer=l2(l2_weight), downsample=False, name="resid4_{}".format(num+1))
        

    
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = Dense(units=num_class, kernel_initializer=initializer, activation='softmax', name="class_output")(x)
    
    return Model(inputs=inputs, outputs=outputs)


model = ResNet_previous(inputs=Input(shape=(224,224,3)), layers=(2,2,2,2), filters=(64,128,256,512), num_class=3, initializer='he_normal')

model.summary()


#%%
"""""""""""""""""""""""""""""""""""
    ResNet for #layers <= 34

"""""""""""""""""""""""""""""""""""

def residual_unit(inputs, filters=64, initializer='glorot_uniform', regularizer=l2(0.01), downsample=False, first_block=False, name="residual_unit"):
    if first_block == False:
        x = BatchNormalization(name="{}_bn1".format(name))(inputs)
        x = Activation('relu', name="{}_act1".format(name))(x)
        
    else:
        x = inputs
        
    
    if downsample == False:
        residual = x
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(x)
        
    else:
        residual = Conv2D(filters=filters, kernel_size=(1,1), strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv_resid".format(name))(x)
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(x)
            
    
    x = BatchNormalization(name="{}_bn2".format(name))(x)
    x = Activation('relu', name="{}_act2".format(name))(x)
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv2".format(name))(x)
    

    return Add(name="{}_add".format(name))([x, residual])



"""""""""""""""""""""""""""""""""""
    default as ResNet34

"""""""""""""""""""""""""""""""""""

def ResNet(inputs, layers=(3,4,6,3), filters=(64,128,256,512), num_class=1000, initializer='glorot_uniform', l2_weight=0.05):
    x = Conv2D(filters=filters[0], kernel_size=(7,7), strides=(2,2), kernel_initializer=initializer, padding='same')(inputs)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    for num in range(layers[0]):
        if num == 0:
            x = residual_unit(x, filters=filters[0], initializer=initializer, regularizer=l2(l2_weight), downsample=False, first_block=True, name="resid1_{}".format(num+1))

        else:
            x = residual_unit(x, filters=filters[0], initializer=initializer, regularizer=l2(l2_weight), downsample=False, first_block=False, name="resid1_{}".format(num+1))
                    
    for num in range(layers[1]):
        if num == 0:
            x = residual_unit(x, filters=filters[1], initializer=initializer, regularizer=l2(l2_weight), downsample=True, first_block=False, name="resid2_{}".format(num+1))
                        
        else:
            x = residual_unit(x, filters=filters[1], initializer=initializer, regularizer=l2(l2_weight), downsample=False, first_block=False, name="resid2_{}".format(num+1))
                        
    for num in range(layers[2]):
        if num == 0:
            x = residual_unit(x, filters=filters[2], initializer=initializer, regularizer=l2(l2_weight), downsample=True, first_block=False, name="resid3_{}".format(num+1))
                        
        else:
            x = residual_unit(x, filters=filters[2], initializer=initializer, regularizer=l2(l2_weight), downsample=False, first_block=False, name="resid3_{}".format(num+1))
                        
    for num in range(layers[3]):
        if num == 0:
            x = residual_unit(x, filters=filters[3], initializer=initializer, regularizer=l2(l2_weight), downsample=True, first_block=False, name="resid4_{}".format(num+1))
                        
        else:
            x = residual_unit(x, filters=filters[3], initializer=initializer, regularizer=l2(l2_weight), downsample=False, first_block=False, name="resid4_{}".format(num+1))
                    
    # x = BatchNormalization(name="final_bn")(x)
    # x = Activation('relu', name="final_act")(x)
    
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = Dense(units=num_class, kernel_initializer=initializer, activation='softmax', name="class_output")(x)
    
    return Model(inputs=inputs, outputs=outputs)


    

model = ResNet(inputs=Input(shape=(224,224,3)), layers=(2,2,2,2), filters=(64,128,256,512), num_class=3, initializer='he_normal')

model.summary()


#%%
"""""""""""""""""""""""""""""""""""""""""""""""""""
    If the layers are more than 50, by following the statement of paper, we should use the following code
    ex. ResNet50, ResNet101, ResNet152
    
    previous ResNet

"""""""""""""""""""""""""""""""""""""""""""""""""""

def residual_unit_bottleneck_previous(inputs, filters=64, initializer='glorot_uniform', regularizer=l2(0.01), downsample=False, change_resid_filter=False, name="residual_unit_bottleneck"):
    if downsample == False and change_resid_filter == False:
        x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(inputs)
        residual = inputs
    
    elif downsample == False and change_resid_filter == True:
        x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(inputs)
        
        residual = Conv2D(filters=filters*4, kernel_size=(1,1), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv_resid".format(name))(inputs)
        residual = BatchNormalization(name="{}_bn_resid".format(name))(residual)
    
    elif downsample == True and change_resid_filter == True:
        x = Conv2D(filters=filters, kernel_size=(1,1), strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(inputs)

        residual = Conv2D(filters=filters*4, kernel_size=(1,1), strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv_resid".format(name))(inputs)
        residual = BatchNormalization(name="{}_bn_resid".format(name))(residual)
    
    
    x = BatchNormalization(name="{}_bn1".format(name))(x)
    x = Activation('relu', name="{}_act1".format(name))(x) 
    
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv2".format(name))(x)
    x = BatchNormalization(name="{}_bn2".format(name))(x)
    x = Activation('relu', name="{}_act2".format(name))(x)
    
    x = Conv2D(filters=filters*4, kernel_size=(1,1), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv3".format(name))(x)   
    x = BatchNormalization(name="{}_bn3".format(name))(x)
    
    
    x = Add(name="{}_add".format(name))([x, residual])
       
    return Activation('relu', name="{}_act3".format(name))(x)



"""""""""""""""""""""""""""""""""""
    default as ResNet50

"""""""""""""""""""""""""""""""""""

def ResNet_bottleneck_previous(inputs, layers=(3,4,6,3), filters=(64,128,256,512), num_class=1000, initializer='glorot_uniform', l2_weight=0.05):
    x = Conv2D(filters=filters[0], kernel_size=(7,7), strides=(2,2), kernel_initializer=initializer, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    for num in range(layers[0]):
        if num == 0:
            x = residual_unit_bottleneck_previous(x, filters=filters[0], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=True, name="resid1_{}".format(num+1))

        else:
            x = residual_unit_bottleneck_previous(x, filters=filters[0], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=False, name="resid1_{}".format(num+1))
            
    for num in range(layers[1]):
        if num == 0:
            x = residual_unit_bottleneck_previous(x, filters=filters[1], initializer=initializer, regularizer=l2(l2_weight), downsample=True, change_resid_filter=True, name="resid2_{}".format(num+1))
            
        else:
            x = residual_unit_bottleneck_previous(x, filters=filters[1], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=False, name="resid2_{}".format(num+1))
            
    for num in range(layers[2]):
        if num == 0:
            x = residual_unit_bottleneck_previous(x, filters=filters[2], initializer=initializer, regularizer=l2(l2_weight), downsample=True, change_resid_filter=True, name="resid3_{}".format(num+1))
            
        else:
            x = residual_unit_bottleneck_previous(x, filters=filters[2], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=False, name="resid3_{}".format(num+1))
            
    for num in range(layers[3]):
        if num == 0:
            x = residual_unit_bottleneck_previous(x, filters=filters[3], initializer=initializer, regularizer=l2(l2_weight), downsample=True, change_resid_filter=True, name="resid4_{}".format(num+1))
            
        else:
            x = residual_unit_bottleneck_previous(x, filters=filters[3], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=False, name="resid4_{}".format(num+1))

    
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = Dense(units=num_class, kernel_initializer=initializer, activation='softmax', name="class_output")(x)
    
    return Model(inputs=inputs, outputs=outputs)



model = ResNet_bottleneck_previous(inputs=Input(shape=(224,224,3)), layers=(2,2,2,2), filters=(64,128,256,512), num_class=3, initializer='he_normal')

model.summary()




#%%
"""""""""""""""""""""""""""""""""""""""""""""""""""
    ResNet for ResNet50, ResNet101, ResNet152...

"""""""""""""""""""""""""""""""""""""""""""""""""""

def residual_unit_bottleneck(inputs, filters=64, initializer='glorot_uniform', regularizer=l2(0.01), downsample=False, change_resid_filter=False, name="residual_unit_bottleneck"):
    if downsample == False and change_resid_filter == False:
        residual = inputs
        
        x = BatchNormalization(name="{}_bn1".format(name))(inputs)
        x = Activation('relu', name="{}_act1".format(name))(x)
        x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(inputs)
        
    elif downsample == False and change_resid_filter == True:
        residual = Conv2D(filters=filters*4, kernel_size=(1,1), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv_resid".format(name))(inputs)
        
        x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(inputs)

    elif downsample == True and change_resid_filter == True:
        x = BatchNormalization(name="{}_bn1".format(name))(inputs)
        x = Activation('relu', name="{}_act1".format(name))(x)
        
        residual = Conv2D(filters=filters*4, kernel_size=(1,1), strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv_resid".format(name))(x)

        x = Conv2D(filters=filters, kernel_size=(1,1), strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv1".format(name))(x)


    x = BatchNormalization(name="{}_bn2".format(name))(x)
    x = Activation('relu', name="{}_act2".format(name))(x)
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv2".format(name))(x)

    x = BatchNormalization(name="{}_bn3".format(name))(x)
    x = Activation('relu', name="{}_act3".format(name))(x)
    x = Conv2D(filters=filters*4, kernel_size=(1,1), strides=(1,1), kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same', name="{}_conv3".format(name))(x)   


    return Add(name="{}_add".format(name))([x, residual])



def ResNet_bottleneck(inputs, layers=(3,4,6,3), filters=(64,128,256,512), num_class=1000, initializer='glorot_uniform', l2_weight=0.05):
    x = Conv2D(filters=filters[0], kernel_size=(7,7), strides=(2,2), kernel_initializer=initializer, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    for num in range(layers[0]):
        if num == 0:
            x = residual_unit_bottleneck(x, filters=filters[0], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=True, name="resid1_{}".format(num+1))
                        
        else:
            x = residual_unit_bottleneck(x, filters=filters[0], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=False, name="resid1_{}".format(num+1))
                        
    for num in range(layers[1]):
        if num == 0:
            x = residual_unit_bottleneck(x, filters=filters[1], initializer=initializer, regularizer=l2(l2_weight), downsample=True, change_resid_filter=True, name="resid2_{}".format(num+1))
                        
        else:
            x = residual_unit_bottleneck(x, filters=filters[1], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=False, name="resid2_{}".format(num+1))
                        
    for num in range(layers[2]):
        if num == 0:
            x = residual_unit_bottleneck(x, filters=filters[2], initializer=initializer, regularizer=l2(l2_weight), downsample=True, change_resid_filter=True, name="resid3_{}".format(num+1))
                        
        else:
            x = residual_unit_bottleneck(x, filters=filters[2], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=False, name="resid3_{}".format(num+1))
                        
    for num in range(layers[3]):
        if num == 0:
            x = residual_unit_bottleneck(x, filters=filters[3], initializer=initializer, regularizer=l2(l2_weight), downsample=True, change_resid_filter=True, name="resid4_{}".format(num+1))
                       
        else:
            x = residual_unit_bottleneck(x, filters=filters[3], initializer=initializer, regularizer=l2(l2_weight), downsample=False, change_resid_filter=False, name="resid4_{}".format(num+1))

    
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = Dense(units=num_class, kernel_initializer=initializer, activation='softmax', name="class_output")(x)
    
    return Model(inputs=inputs, outputs=outputs)



model = ResNet_bottleneck(inputs=Input(shape=(224,224,3)), layers=(2,2,2,2), filters=(64,128,256,512), num_class=3, initializer='he_normal')

model.summary()







