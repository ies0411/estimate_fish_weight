from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam , RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint , LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.applications import Xception, ResNet50V2, EfficientNetB0, EfficientNetB1
from tensorflow.keras.applications import MobileNet
import tensorflow as tf
from keras_radam import RAdam

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import sklearn
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess_input
import albumentations as A

def create_model(model_type='efficientnetb7', in_shape=(224, 224, 3), n_classes=1):
    input_tensor = Input(shape=in_shape)
    if model_type == 'resnet50v2':
        base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif model_type == 'xception':
        base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif model_type == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif model_type == 'efficientnetb1':
        base_model = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif model_type == 'efficientnetb7':
        base_model = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=input_tensor)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation=LeakyReLU(0.2))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation=LeakyReLU(0.2))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation=LeakyReLU(0.2))(x)
    x = Dropout(0.5)(x)
#     preds = Dense(1)(x)
    preds = Dense(1, activation='linear')(x)
    model = Model(inputs=input_tensor, outputs=preds)

    return model
IMAGE_DIR = './dataset/train'
target = pd.read_csv('./dataset/train.csv')
def MakeDataFrame(image_dir=IMAGE_DIR):
  paths = []
  label_gubuns = []
  for dirname, _, filenames in os.walk(IMAGE_DIR):
      for filename in filenames:
          # there are some file which is not .jpg
          if '.jpg' in filename:
              file_path = dirname+'/'+ filename
              #appending full directory of image
              paths.append(file_path)
              #folder name of label
              label = file_path.split('/')[3]
              #average weight of label
              label_n = target.loc[target['ImageDir']==label,'AvgWeight'].values[0]
              label_gubuns.append(label_n)

  return pd.DataFrame({'path':paths, 'label':label_gubuns})

def GetTrainValid(train_df, valid_size=0.1):
    train_path = train_df['path'].values
    train_label = train_df['label'].values
    tr_path = train_path
    tr_label = train_label
    # tr_path, val_path, tr_label, val_label \
    # = train_test_split(train_path, train_label, test_size=valid_size)

#     print('tr_path shape:', tr_path.shape, 'tr_label shape:', tr_label.shape, 'val_path shape:', val_path.shape, 'val_label shape:', val_label.shape)
    return tr_path,  tr_label

# test = MakeDataFrame()
# tr,val,tr_l,val_l = GetTrainValid(test)
# print(tr_l)


from keras.models import load_model
import sklearn.metrics# import accuracy_score


#make data flow
class Dataset(Sequence):
    def __init__(self, image_filenames, labels, image_size=(224, 400), batch_size=64,
                 augmentor=None, shuffle=True, pre_func=None):

        self.image_filenames = image_filenames
        self.labels = labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.pre_func = pre_func
        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()
            pass

    def __len__(self):
        # iteration number
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):

        image_name_batch = self.image_filenames[index*self.batch_size:(index+1)*self.batch_size]
        if self.labels is not None:
            label_batch = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        else:
            label_batch = None

        image_batch = np.zeros((image_name_batch.shape[0], self.image_size[0], self.image_size[1], 3), dtype='float32')

        for image_index in range(image_name_batch.shape[0]):
            image = cv2.cvtColor(cv2.imread(image_name_batch[image_index]),\
                                 cv2.COLOR_BGR2RGB)
            if self.augmentor is not None:
                image = self.augmentor(image=image)['image']
#             image = cv2.resize(image, (self.image_size, self.image_size))
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            if self.pre_func is not None:
                image = self.pre_func(image)

            image_batch[image_index] = image

        return image_batch, label_batch

    def on_epoch_end(self):
        if(self.shuffle):
            self.image_filenames, self.labels =\
            sklearn.utils.shuffle(self.image_filenames, self.labels)
        else:
            pass

# learning rate scheduler에 적용할 함수 선언.
def lrfn_01(epoch):
    LR_START = 1e-5
    LR_MAX = 1e-4
    LR_RAMPUP_EPOCHS = 2
    LR_SUSTAIN_EPOCHS = 1
    LR_STEP_DECAY = 0.75

    def calc_fn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//2)
        return lr

    return calc_fn(epoch)

def lrfn_02(epoch):
    LR_START = 1e-6
    LR_MAX = 2e-5
    LR_RAMPUP_EPOCHS = 2
    LR_SUSTAIN_EPOCHS = 1
    LR_STEP_DECAY = 0.75

    def calc_fn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//2)
        return lr

    return calc_fn(epoch)


lr01_cb = tf.keras.callbacks.LearningRateScheduler(lrfn_01, verbose=1)
lr02_cb = tf.keras.callbacks.LearningRateScheduler(lrfn_02, verbose=1)
rlr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, mode='min', verbose=1)
ely_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)



from keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error
from keras.callbacks import ModelCheckpoint

def train_model(model_type, train_df, initial_lr=0.00001, augmentor=None, input_pre_func=None,
                loss='mse',batch_size=64,image_size=(224, 400)):
    tr_path, tr_label = GetTrainValid(train_df, valid_size=0.05)
    tr_ds = Dataset(tr_path, tr_label, image_size=image_size, batch_size=batch_size,
                          augmentor=augmentor, shuffle=True, pre_func=input_pre_func)
    # val_ds = Dataset(val_path, val_label, image_size=image_size, batch_size=batch_size,
    #                       augmentor=None, shuffle=False, pre_func=input_pre_func)

    # model = create_model(model_type=model_type)
    model = tf.keras.models.load_model('./ver-5checkpoint-epoch-15-batch-4-finetune-False-trial-001.h5')
    model.compile(optimizer=Adam(initial_lr), loss=loss,  metrics=['mae','mse'])
#RAdam()

    # 만일 Fine tuning 일 경우 아래 로직 적용.
#     if config.IS_FINE_TUNING:
    # print('####### Fine tuning ########')
    # 첫번째 Fine Tuning. Feature Extractor를 제외한 classification layer를 학습.(Feature Extractor layer들을 trainable=False 설정)
    # for layer in model.layers[:-6]:
    #     layer.trainable = False

    # rlr_cb = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=30, mode='min', verbose=1)
    # ely_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

    print('####### Classification Layer들의 학습을 시작합니다. ########')


    filename = './ver-{}-checkpoint-trial.h5'.format(6)
    checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             monitor='loss',   # val_loss 값이 개선되었을때 호출됩니다
                             verbose=1,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                             mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                            )


    # history = model.fit(tr_ds, epochs=5, steps_per_epoch=int(np.ceil(tr_path.shape[0]/batch_size)),
    #                    validation_data=None,
    #                    callbacks=([rlr_cb,ely_cb,checkpoint]), verbose=1)

#     model.save('./myungsun_rev2.h5')
    # model.save('./myungsun_rev5.h5')

    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
                layer.trainable = True

    # filename = './ver-{}checkpoint-epoch-{}-batch-{}-finetune-{}-trial-001.h5'.format(5,15, batch_size,False)
    # checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
    #                          monitor='loss',   # val_loss 값이 개선되었을때 호출됩니다
    #                          verbose=1,            # 로그를 출력합니다
    #                          save_best_only=True,  # 가장 best 값만 저장합니다
    #                          mode='auto'           # auto는 알아서 best를 찾습니다. min/max
    #                         )
    print('####### 전체 Layer들의 학습을 시작합니다. ########')
    # for index in range(10):
    history = model.fit(tr_ds, epochs=30, steps_per_epoch=int(np.ceil(tr_path.shape[0]/batch_size)),
                      validation_data=None,
                      callbacks=([checkpoint]), verbose=1)
    # model.save('./myungsun_rev-{}.h5'.format(index))

## Not Fine Tuning
#     else:
#         print('####### 학습을 시작합니다. ########')
#         history = model.fit(tr_ds, epochs=config.N_EPOCHS, steps_per_epoch=int(np.ceil(tr_path.shape[0]/config.BATCH_SIZE)),
#                        validation_data=val_ds, validation_steps=int(np.ceil(val_p




#     history = model.fit(tr_ds, epochs=epochs, steps_per_epoch=tr_path.shape[0]//batch_size,
#                    validation_data=val_ds, validation_steps=val_path.shape[0]//batch_size,
#                    callbacks=([lr02_cb,ely_cb]), verbose=1)
    model.save('./myungsun_rev7.h5')

    return model, history

augmentor_01 = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.VerticalFlip(p=0.2),
    # A.ToGray(p=0.2),
    # A.ToSepia(p=0.2),
    # A.Equalize(p=0.2),
    # A.Sharpen(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.2),
    A.OneOf(
        [
        #  A.GaussNoise(p=1, var_limit=(100, 200)),
         A.Blur(blur_limit=(10, 15), p=1),
        #  A.RandomShadow(p=1),
        #  A.MotionBlur(p=1),
        #  A.MedianBlur(p=1),
        #  A.ISONoise(p=1),
        #  A.CLAHE(p=1, clip_limit=4),
        #  A.Posterize(p=1)
        ], p=0.1)
])


from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess_input
# from tensorflow.keras.applications.efficientnet import preprocess_input as xcp_preprocess_input


IMAGE_DIR = './dataset/train'

train_df = MakeDataFrame(image_dir=IMAGE_DIR)


# model, history = train_model(model_type='efficientnetb7',train_df=train_df, initial_lr=0.00001,image_size=(280, 500),
#                              augmentor=augmentor_01, input_pre_func=eff_preprocess_input,batch_size=4)


print("###inference###")
paths = []
label_gubuns = []
IMAGE_DIR = './dataset/test'
for dirname, _, filenames in os.walk(IMAGE_DIR):
    for filename in filenames:
        # there are some file which is not .jpg
        if '.jpg' in filename:

            file_path = dirname+'/'+ filename
            paths.append(file_path)

            label = file_path.split('/')[3]
            label_gubuns.append(label)

test_df=pd.DataFrame({'path':paths})
test_path = test_df['path'].values

# test_ds = Dataset(image_filenames=test_path,labels= None,batch_size=4, augmentor=augmentor_01, shuffle=None, pre_func=eff_preprocess_input)

# len(test_path)
model = tf.keras.models.load_model('./myungsun_rev7.h5')

print("##load model completed##")
test_ds = Dataset(image_filenames=test_path,labels= None,batch_size=6, augmentor=None, shuffle=None, image_size=(250, 480),pre_func=eff_preprocess_input)
preds = model.predict(test_ds)
# print("1 set was completed")

# test_ds_2 = Dataset(image_filenames=test_path,labels= None,batch_size=6, augmentor=augmentor_01, shuffle=None, pre_func=eff_preprocess_input)
# preds_2 = model.predict(test_ds_2)
# print("2 set was completed")

# test_ds_3 = Dataset(image_filenames=test_path,labels= None,batch_size=6, augmentor=augmentor_01, shuffle=None, pre_func=eff_preprocess_input)
# preds_3 = model.predict(test_ds_3)
# print("3 set was completed")

# test_ds_4 = Dataset(image_filenames=test_path,labels= None,batch_size=6, augmentor=augmentor_01, shuffle=None, pre_func=eff_preprocess_input)
# preds_4 = model.predict(test_ds_4)
# print("4 set was completed")

# preds = (preds_1+preds_2+preds_3+preds_4)/4.0


sample = pd.read_csv('./submission_sample .csv')


image_dir_df = pd.DataFrame(label_gubuns)
image_dir_df.columns=['ImageDir']
predics_df = pd.DataFrame(preds)
predics_df.columns=['AvgWeight']
concat_df=pd.concat([image_dir_df,predics_df],axis=1)


send_df_dir=[]
send_df_value=[]
idx=0
sum_idx=0
sum=0
for dirname in concat_df['ImageDir']:
  # print(dirname)
  if idx is not 0:
    if concat_df.loc[idx,'ImageDir']==concat_df.loc[idx-1,'ImageDir']:
        if idx == 7332:
            print("complete")
            sum = sum +concat_df.loc[idx,'AvgWeight']
            send_df_dir.append(concat_df.loc[idx,'ImageDir'])
            send_df_value.append(sum/sum_idx)
        else :
            sum = sum +concat_df.loc[idx,'AvgWeight']

    else:

     send_df_dir.append(concat_df.loc[idx-1,'ImageDir'])
     send_df_value.append(sum/sum_idx)

     sum = concat_df.loc[idx,'AvgWeight']
     sum_idx=0

  else:
    sum = sum + concat_df.loc[idx,'AvgWeight']

  idx=idx+1
  sum_idx=sum_idx+1

send_df = pd.DataFrame({'ImageDir':send_df_dir,'AvgWeight':send_df_value})

for dirname in sample['ImageDir']:
    sample.loc[sample['ImageDir']==dirname,'AvgWeight']=send_df.loc[send_df['ImageDir']==dirname,'AvgWeight'].values

send_df.to_csv('./submit_rev6.csv', index=False)


