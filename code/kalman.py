import numpy as np
import matplotlib.pyplot as plt
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
    # 读取数据
    dataType = 'reference'
    subject = '1'
    trial = '2'

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
    for i in range (0, sampleNum):
        filteredAccX.append(AccX_Filter.Kalman(accx[i]))
        filteredAccY.append(AccY_Filter.Kalman(accy[i]))
        filteredAccZ.append(AccZ_Filter.Kalman(accz[i]))
        filteredGyroX.append(GyroX_Filter.Kalman(gyrox[i]))
        filteredGyroY.append(GyroY_Filter.Kalman(gyroy[i]))
        filteredGyroZ.append(GyroZ_Filter.Kalman(gyroz[i]))
        
    df1 = np.array([filteredAccX,filteredAccY,filteredAccZ,filteredGyroX,filteredGyroY,filteredGyroZ])
    
    '''
    # 折线图
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))
    fig.suptitle(f'{dataType} {subject}-{trial}')
    subtitles = ['Accel', 'Gyro']
    Axes = ['X','Y','Z']
    for i in range(2):
        for j in range(3):
            x = np.arange(df1.shape[1])
            y = df1[3*i+j, :]
            axs[i].plot(x, y, label=Axes[j])
        axs[i].set_title(subtitles[i])
    plt.tight_layout()
    plt.legend()
    plt.show()
    '''



    # 绘图
    l1, = plt.plot(timeAxis, accx, label="Acceleration X")
    l2, = plt.plot(timeAxis, filteredAccX, label="Kalman Filtered Acceleration X")
    plt.xlabel("Time(s)")
    plt.ylabel("Acceleration X(m/s²)")
    plt.legend(handles=[l1,l2])
    plt.grid()
    plt.savefig("code\\images\\kalman.png")
    plt.show()
'''
    l1, = plt.plot(timeAxis, accy, label="Acceleration Y")
    l2, = plt.plot(timeAxis, filteredAccY, label="Kalman Filtered Acceleration Y")
    plt.xlabel("Time(s)")
    plt.ylabel("Acceleration Y(m/s²)")
    plt.legend(handles=[l1,l2])
    plt.grid()
    plt.show()

    l1, = plt.plot(timeAxis, accz, label="Acceleration Z")
    l2, = plt.plot(timeAxis, filteredAccZ, label="Kalman Filtered Acceleration Z")
    plt.xlabel("Time(s)")
    plt.ylabel("Acceleration Z(m/s²)")
    plt.legend(handles=[l1,l2])
    plt.grid()
    plt.show()

    l1, = plt.plot(timeAxis, gyrox, label="Gyroscope X")
    l2, = plt.plot(timeAxis, filteredGyroX, label="Kalman Filtered Gyroscope X")
    plt.xlabel("Time(s)")
    plt.ylabel("Gyroscope X(rad/s)")
    plt.legend(handles=[l1,l2])
    plt.grid()
    plt.show()

    l1, = plt.plot(timeAxis, gyroy, label="Gyroscope Y")
    l2, = plt.plot(timeAxis, filteredGyroY, label="Kalman Filtered Gyroscope Y")
    plt.xlabel("Time(s)")
    plt.ylabel("Gyroscope Y(rad/s)")
    plt.legend(handles=[l1,l2])
    plt.grid()
    plt.show()

    l1, = plt.plot(timeAxis, gyroz, label="Gyroscope Z")
    l2, = plt.plot(timeAxis, filteredGyroZ, label="Kalman Filtered Gyroscope Z")
    plt.xlabel("Time(s)")
    plt.ylabel("Gyroscope Z(rad/s)")
    plt.legend(handles=[l1,l2])
    plt.grid()
    plt.show()
    '''

    