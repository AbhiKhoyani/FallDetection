import tensorflow as tf

class MLP(tf.keras.Model):
  def __init__(self, units = [1024,512,256,64,32,16], dropout=0, n_classes = 2): 
    super().__init__()
    self.num_layers = len(units)
    for i,n in enumerate(units):
      setattr(self, f"dense{i}", tf.keras.layers.Dense(n, activation = 'relu'))
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.final = tf.keras.layers.Dense(1, activation = 'sigmoid')

  def call(self, inputs, training = False):
    x = inputs
    
    for i in range(self.num_layers):
        x = getattr(self, f'dense{i}')(x)
        if training:
            x = self.dropout(x)
    return self.final(x)

class CNN(tf.keras.Model):
  def __init__(self, units = [300,200,100], kernels=[9,7,5],strides=[1,1,1], pool = True ,dropout=0, n_classes = 2): 
    super().__init__()

    assert len(units)==len(kernels)==len(strides), "All the arguments shall be of same length."
    self.num_layers = len(units)
    self.pool = pool
    for i in range(self.num_layers):
      setattr(self, f"cnn{i}", tf.keras.layers.Conv1D(units[i], kernels[i], strides = strides[i], padding= 'same', activation = 'relu'))
      if self.pool:
        setattr(self, f'avgPool{i}', tf.keras.layers.MaxPooling1D(3, strides =1, padding ='valid'))
    

    self.dropout = tf.keras.layers.Dropout(dropout)
    self.globalPool = tf.keras.layers.GlobalAveragePooling1D(data_format = 'channels_first')
    self.final = tf.keras.layers.Dense(1, activation = 'sigmoid')

  def call(self, inputs, training = False):
    x = tf.expand_dims(inputs, -1)
    for i in range(self.num_layers):
        x = getattr(self, f'cnn{i}')(x)
        if self.pool:
            x = getattr(self, f'avgPool{i}')(x)
        if training:
            x = self.dropout(x)
    
    x = self.globalPool(x)
    return self.final(x)

class Transformer_Head(tf.keras.Model):
  def __init__(self, input_shape, head_size, num_heads, ff_dim, dropout=0): 
    super().__init__()
    self.layerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6) 
    # self.layerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6) 
    self.multiHead = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout) 
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.conv1 = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
    self.conv2 = tf.keras.layers.Conv1D(filters=input_shape, kernel_size=1)
  
  def call(self, inputs, training = False):
    x = inputs
    x = self.layerNorm(x)
    x = self.multiHead(x, x)
    x = self.dropout(x)
    res = x + inputs

    # Feed Forward Part
    x = self.layerNorm(res)
    x = self.conv1(x)
    x = self.dropout(x)
    x = self.conv2(x)
    return x + res

    
class Transformer(tf.keras.Model):
  def __init__(self, d_model, 
    head_size = 256,
    num_heads = 2,
    ff_dim = 2,
    num_transformer_blocks = 2,
    mlp_units = [64],
    dropout=0,
    mlp_dropout=0,
    n_classes = 2): 
    super().__init__()
    
    self.d_model = d_model
    self.head_size = head_size
    self.num_heads = num_heads
    self.ff_dim = ff_dim
    self.num_transformer_blocks = num_transformer_blocks
    self.units = mlp_units
    self.num_mlp = len(mlp_units)

    for i in range(num_transformer_blocks):
      setattr(self, f'txEnc{i}', Transformer_Head(self.d_model, self.head_size, self.num_heads, self.ff_dim, dropout))
    
    self.globalPool = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")
    self.mlp_dropout = tf.keras.layers.Dropout(mlp_dropout)
    for i,n in enumerate(self.units):
      setattr(self, f'dense{i}', tf.keras.layers.Dense(i, activation = 'relu'))
    self.final = tf.keras.layers.Dense(1, activation = 'sigmoid')

  def call(self, inputs, training = False):
    x = tf.expand_dims(inputs, -1)
    for i in range(self.num_transformer_blocks):
        x = getattr(self, f'txEnc{i}')(x)
  
    x = self.globalPool(x)
    for i in range(self.num_mlp):
        x = getattr(self, f'dense{i}')(x)
        if training:
          x = self.mlp_dropout(x)
    return self.final(x)

