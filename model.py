import csv
import cv2
import imutils
import argparse
import numpy as np
from Net import model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, model=None, base_path='', epochs=5):
        self.dataset = []
        self.model = model
        self.epochs = epochs
        self.training_dataset = []
        self.validation_dataset = []
        self.steering_correction = 0.2
        self.base_path = base_path
        self.image_path = self.base_path + '/IMG/'
        self.log_path = self.base_path +'driving_log.csv'
         
    def LoadDataset(self):
        with open(self.log_path) as csvfile:
            reader = csv.reader(csvfile)
            
            next(reader)
            
            for row in reader:
                self.dataset.append(row)
                
        return None
   
    def ProcessData(self, data_batch):
        steering_angle = np.float32(data_batch[3])
        images = []
        steering_angles = []
        
        for column_index in range(3):
            filename = data_batch[column_index].split('/')[-1]
            #print(filename)
            image = cv2.imread(self.image_path + filename)
            image = imutils.preprocess(image)
  
            images.append(image)
                
            if column_index ==1:
                steering_angle = steering_angle + self.steering_correction                               
                steering_angles.append(steering_angle)   
            elif column_index ==2:
                steering_angle = steering_angle - self.steering_correction
                steering_angles.append(steering_angle) 
            else:     
                steering_angles.append(steering_angle)
                
            if column_index==0:
                images.append(imutils.flip(image))
                steering_angles.append( - steering_angle)
        return images, steering_angles
        
    def DataGenerator(self, datatype, BS = 32):
        while True:
            shuffle(datatype)
            # Split dataset into batches of size BS
            for i in range(0, len(datatype), BS):
                data_batches = datatype[i: i + BS]
                images, angles = [], []
                
                for data_batch in data_batches:
                    aug_images, aug_angles = self.ProcessData(data_batch)
                    images.extend(aug_images)
                    angles.extend(aug_angles)
                    
                images = np.array(images) 
                angles  = np.array(angles)
                
                yield shuffle(images, angles)
            
            
    def split_dataset(self):
        training, validation = train_test_split(self.dataset, test_size=0.2)
        self.training_dataset, self.validation_dataset = training, validation
        print('size of the dataset is:', len(self.dataset))
        print('size of training dataset is:', len(self.training_dataset))
        print('size of validation dataset is:', len(self.validation_dataset)) 
        
        return None
    
    def training_generator(self):
        return self.DataGenerator(datatype = self.training_dataset, BS = 32)
    
    def validation_generator(self):
        return self.DataGenerator(datatype = self.validation_dataset, BS = 32)
            
            
    def train(self):
        self.split_dataset()
        history = self.model.fit_generator(generator=self.training_generator(),
                                 validation_data=self.validation_generator(),
                                 epochs=self.epochs,
                                 steps_per_epoch=len(self.training_dataset) // 32,
                                 validation_steps=len(self.validation_dataset) // 32 )
        self.model.save('model.h5')
        
        
            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--data-base-path',
        type=str,
        default='/home/workspace/CarND-Behavioral-Cloning-P3/data/',
        help='Path to image directory and driving log'
    )
    args = ap.parse_args()

    # Instantiate the pipeline
    pipeline = Pipeline(model=model(), base_path=args.data_base_path, epochs=5)

    pipeline.LoadDataset()
   
    pipeline.train()

if __name__ == '__main__':
    main()
    
    
    