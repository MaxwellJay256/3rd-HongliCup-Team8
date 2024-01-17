import numpy as np
from scipy.stats import kurtosis, skew

def get_features(y):
    """
    Args:
        y : wave data
    Returns:
        max_y（最大值）, min_y（最小值）, median_y（中位数）, mean_y（均值）, var_y（方差）, peak（峰值）, peak2peak（峰峰值）, rms（有效值），
        crestf（峰值因子）, margin（裕度因子）, pulse（脉冲因子）, waveform（波形因子）, kur（峭度因子）, sk（偏度因子）
    """
    max_y = np.max(y)
    min_y = np.min(y)
    median_y = np.median(y)
    mean_y = np.mean(y)
    var_y = np.var(y)
 
    rms = np.sqrt(np.mean(np.square(y)))
    peak = max(abs(max_y), abs(min_y))
    peak2peak = max_y - min_y
    # 峰值因子
    crestf = max_y / rms if rms != 0 else 0
    xr = np.square(np.mean(np.sqrt(np.abs(y))))
    # 裕度
    margin = (max_y / xr) if xr != 0 else 0 
    yr = np.mean(np.abs(y))
    # 脉冲因子
    pulse = max_y / yr if yr != 0 else 0 
    # 波形因子
    waveform = rms / yr if yr !=0 else 0  
    # 峭度
    kur = kurtosis(y) 
    # 偏斜度
    sk = skew(y)   
    features = np.array([max_y, min_y, median_y, mean_y, var_y, peak, peak2peak, rms, crestf, margin, pulse,waveform, kur, sk]) 
    return features

# 读取数据

data = []

number = ['1','2','3','4','5','6','7','8','9','10']
for i in range(1,11):
    for j in range(1,11):
        dataType = 'referenceKalman'
        subject = number[i-1]
        trial = number[j-1]
        fileName = f'code\\{dataType}\\{subject}\\{trial}.txt'
        df = np.loadtxt(fileName)
        feature_v = np.array([get_features(df[0]),get_features(df[1]),get_features(df[2]),get_features(df[3]),get_features(df[4]),get_features(df[5])]).flatten()
        data.append(feature_v)
data1 = np.array(data)
print(data1.shape)
np.savetxt("code\\reference.txt",data1)