import numpy as np
import matplotlib.pyplot as plt
import os
import csv

class Kalman_Filter:
    def __init__(self, Q, R): # 构造函数
        self.Q = Q
        self.R = R

        self.P_k_k1 = 1
        self.Kg = 0
        self.P_k1_k1 = 1
        self.x_k_k1 = 0
        self.ADC_OLD_Value = 0
        self.Z_k = 0
        self.kalman_adc_old=0

    def Kalman(self, ADC_Value):
        self.Z_k = ADC_Value

        if ( abs(self.kalman_adc_old - ADC_Value) >= 60 ):
            self.x_k1_k1 = ADC_Value * 0.382 + self.kalman_adc_old * 0.618
        else:
            self.x_k1_k1 = self.kalman_adc_old

        self.x_k_k1 = self.x_k1_k1
        self.P_k_k1 = self.P_k1_k1 + self.Q
        self.Kg = self.P_k_k1 / (self.P_k_k1 + self.R)

        kalman_adc = self.x_k_k1 + self.Kg * (self.Z_k - self.kalman_adc_old)
        self.P_k1_k1 = (1 - self.Kg) * self.P_k_k1
        self.P_k_k1 = self.P_k1_k1

        self.kalman_adc_old = kalman_adc
        return kalman_adc

def GetSampleTime(timelist):
    "获取时间列表的平均取样时间间隔"
    sampleNum = len(timelist)
    timeInterval = [timelist[j+1] - timelist[j] for j in range(0, sampleNum-1)]
    _timeInterval = np.asarray(timeInterval)
    avgSampleTime = np.mean(_timeInterval)
    return avgSampleTime

if __name__ == '__main__':
    number = ['1','2','3','4','5','6','7','8','9','10']
    for i in range(1,11):
        for j in range(1,11):
            # 读取数据
            dataType = 'reference'
            subject = number[i-1]
            trial = number[j-1]

            fileName = f'code\\{dataType}\\{subject}\\{trial}.txt'
            df = np.loadtxt(fileName)
            
        
            time0 = list(range(1, df.shape[1] + 1)) # 时间节点
            time = [x * 0.002 for x in time0]
            accx = df[0].tolist() # x轴加速度
            accy = df[1].tolist() # y轴加速度
            accz = df[2].tolist() # z轴加速度
            gyrox = df[3].tolist() # x轴角速度
            gyroy = df[4].tolist() # y轴角速度
            gyroz = df[5].tolist() # z轴角速度

            
            
            # 卡尔曼滤波
            AccX_Filter = Kalman_Filter(0.001, 0.1)
            AccY_Filter = Kalman_Filter(0.001, 0.1)
            AccZ_Filter = Kalman_Filter(0.001, 0.1)
            GyroX_Filter = Kalman_Filter(0.001, 0.1)
            GyroY_Filter = Kalman_Filter(0.001, 0.1)
            GyroZ_Filter = Kalman_Filter(0.001, 0.1)

            sampleTime = GetSampleTime(time)
            sampleNum = len(time)
            timeAxis = np.linspace(0, sampleNum*sampleTime, sampleNum, True)

            filteredAccX = []
            filteredAccY = []
            filteredAccZ = []
            filteredGyroX = []
            filteredGyroY = []
            filteredGyroZ = []
            for k in range (0, sampleNum):
                filteredAccX.append(AccX_Filter.Kalman(accx[k]))
                filteredAccY.append(AccY_Filter.Kalman(accy[k]))
                filteredAccZ.append(AccZ_Filter.Kalman(accz[k]))
                filteredGyroX.append(GyroX_Filter.Kalman(gyrox[k]))
                filteredGyroY.append(GyroY_Filter.Kalman(gyroy[k]))
                filteredGyroZ.append(GyroZ_Filter.Kalman(gyroz[k]))
                
            df1 = np.array([filteredAccX,filteredAccY,filteredAccZ,filteredGyroX,filteredGyroY,filteredGyroZ])
            
            folder1 = 'code\\referenceKalman'
            folder2 = subject
            file_name = f'{trial}.txt'

            
            path1 = os.path.join(folder1, folder2)
            if not os.path.exists(path1):
                os.makedirs(path1)
            path2 = os.path.join(path1, file_name)
            np.savetxt(path2,df1)