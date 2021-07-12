# ArcFace 

1. **演算法**: 假設樣本 $i$ 的 $label$ 為 $y_i$，傳統 **Softmax** 的第 $y_i$ 類輸出機率為
   $$
   \frac{e^{W_{y_i}^T x_i + b_{y_i}}}{\sum_{j=1}^{n} e^{W_j^{T}x_i + b_j}} \tag{1}
   $$
   <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210712133620541.png" alt="image-20210712133620541" style="zoom: 67%;" />
   
   （Source：https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2）
   
   其中 $W, b$ 分別為 $weight$, $bias$。公式 $(1)$ 可改寫成
   $$
   \frac{e^{||W_{y_i}|| \cdot ||x_i|| \cos\theta_{y_{i}} + b_{y_i}}}{\sum_{j=1}^{n} e^{||W_j||\cdot ||x_i|| \cos\theta_{j}  + b_j}} \tag{2}
   $$
   其中 $\cos\theta_j$ 為 $W_j, x_i$ 間的夾角。假設 $||W_j|| = 1,\ ||x_i|| =s,\ b_j = 0$，公式變成
   $$
   \frac{e^{s \cos\theta_{y_{i}}}}{\sum_{j=1}^{n} e^{s\cos\theta_{j}}} \tag{3}
   $$
   相當於將 $x_i$ 投射到半徑為 $s$ 的超球面上，為了進一步提高模型分辨能力，使得 **intra-class** 間更為緊湊，**inter-class** 間更為分散，**ArcFace** 引進 **Additive Angular Margin Penalty** $m$，最後的公式為
   $$
   \frac{e^{s \cos(\theta_{y_{i}}+m)}}{e^{s \cos(\theta_{y_{i}}+m)} + \sum_{j=1, j \neq y_i}^{n} e^{s\cos\theta_{j}}} \tag{4}
   $$
   其中，**$s, m$ 為模型超參數**。因此，**ArcFace** 的損失函數為
   $$
   L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cos(\theta_{y_{i}}+m)}}{e^{s \cos(\theta_{y_{i}}+m)} + \sum_{j=1, j \neq y_i}^{n} e^{s\cos\theta_{j}}} \tag{5}
   $$
   <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210711113950574.png" alt="image-20210711113950574" style="zoom: 200%;" />

   （Source：https://arxiv.org/pdf/1801.07698.pdf）

   

   ![image-20210711114113036](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210711114113036.png)
   
   （Source：https://arxiv.org/pdf/1801.07698.pdf）
   
   
   
2. **意義**:  使用公式 $(3)$ 所訓練的模型會使 $x_i$ 與 $W_{y_{i}}$ 在超球面上靠近，公式 $(4)$ 加入**Additive Angular Margin Penalty** 能進一步提高 **intra-class** 緊湊度，降低 **inter-class** 分散度，增加 **Softmax** 在 $label \ index$ 上所對應的機率值。 

   ![image-20210711115054583](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210711115054583.png)

   （Source：https://arxiv.org/pdf/1801.07698.pdf）

3. **延伸**: **SphereFace** 公式與 **ArcFace** 相近，差別在於 **SphereFace** 使用 **Multiplicative Angular Margin Penalty**，公式為
   $$
   \frac{e^{s \cos(m \theta_{y_{i}})}}{e^{s \cos(m \theta_{y_{i}})} + \sum_{j=1, j \neq y_i}^{n} e^{s\cos\theta_{j}}} \tag{5}
   $$
   ![image-20210711161607949](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210711161607949.png)

   （Source：https://arxiv.org/pdf/1801.07698.pdf）

4. **程式實作**:

   ```python
   import tensorflow.keras.backend as K
   
   class ArcFace(Layer):
       """
       Implementation of ArcFace layer. Reference: https://arxiv.org/abs/1801.07698
       
       Arguments:
         num_classes: number of classes to classify
         s: scale factor
         m: margin
         regularizer: weights regularizer
       """
       def __init__(self,
                    num_classes,
                    s=30.0,
                    m=0.5,
                    regularizer=None,
                    name='arcface',
                    **kwargs):
           
           super().__init__(name=name, **kwargs)
           self._n_classes = num_classes
           self._s = float(s)
           self._m = float(m)
           self._regularizer = regularizer
   
       def build(self, input_shape):
           embedding_shape, label_shape = input_shape
           self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                     initializer='glorot_uniform',
                                     trainable=True,
                                     regularizer=self._regularizer,
                                     name='cosine_weights')
   
       def call(self, inputs, training=None):
           """
           During training, requires 2 inputs: embedding (after backbone+pool+dense),
           and ground truth labels. The labels should be sparse (and use
           sparse_categorical_crossentropy as loss).
           """
           embedding, label = inputs
   
           # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
           label = tf.reshape(label, [-1], name='label_shape_correction')
   
           # Normalize features and weights and compute dot product
           x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
           w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
           cosine_sim = tf.matmul(x, w, name='cosine_similarity')
   
           if not training:
               # We don't have labels if we're not in training mode
               return self._s * cosine_sim
           else:
               one_hot_labels = tf.one_hot(label,
                                           depth=self._n_classes,
                                           name='one_hot_labels')
               theta = tf.math.acos(K.clip(
                       cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
               selected_labels = tf.where(tf.greater(theta, math.pi - self._m),
                                          tf.zeros_like(one_hot_labels),
                                          one_hot_labels,
                                          name='selected_labels')
               final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                                      theta + self._m,
                                      theta,
                                      name='final_theta')
               output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
               return self._s * output
           
       def get_config(self):
           config = {
                     "num_classes" : self._n_classes,
                     "s" : self._s,
                     "m" : self._m,
           }
           base_config = super().get_config()
           return dict(list(base_config.items()) + list(config.items()))  
   ```
   
   

1. **參考連結**
   + https://arxiv.org/pdf/1801.07698.pdf

   + https://www.kaggle.com/josealways123/understanding-arcface-and-adacos

   + https://www.kaggle.com/chankhavu/keras-layers-arcface-cosface-adacos







