import ee
import geemap
ee.Authenticate()
# Initialize Earth Engine
ee.Initialize(
    project='ee-lbwnb331161',
    opt_url='https://earthengine-highvolume.googleapis.com'
)

# 加载缓冲区 shapefile
buffer = ee.FeatureCollection('projects/ee-lbwnb331161/assets/mangrove_2020_1km_buffer')

# 加载红树林影像
mangrove = ee.Image('projects/ee-lbwnb331161/assets/GreatBay_Mangrove_10m_WGS84')

# 将红树林影像转换为二值图像(1表示红树林,其他为0或mask)
mangrove_binary = mangrove.eq(1)

# 在缓冲区内裁剪红树林
mangrove_clipped = mangrove_binary.clip(buffer)

# ========== 定义缩放和处理函数 ==========

# 定义缩放函数 - Landsat 5/7
def applyScaleFactorsL57(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True) \
                .addBands(thermalBand, None, True)

# 定义缩放函数 - Landsat 8/9
def applyScaleFactorsL89(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True) \
                .addBands(thermalBand, None, True)

def maskL57Clouds(image):
    qa = image.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1 << 1).eq(0)   # dilated cloud
            .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
            .And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud shadow
            .And(qa.bitwiseAnd(1 << 5).eq(0))) # snow
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])

# 定义云掩码函数 - Landsat 8/9
# 另加 Cirrus(2)
def maskL89Clouds(image):
    qa = image.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1 << 1).eq(0)   # dilated cloud
            .And(qa.bitwiseAnd(1 << 2).eq(0))  # cirrus
            .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
            .And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud shadow
            .And(qa.bitwiseAnd(1 << 5).eq(0))) # snow
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])


# 波段重命名函数 - Landsat 5/7
def renameL57(image):
    return image.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'],
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL']
    ).copyProperties(image, ["system:time_start"])

# 波段重命名函数 - Landsat 8/9
def renameL89(image):
    return image.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'],
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL']
    ).copyProperties(image, ["system:time_start"])

# ========== 基础过滤函数 ==========
def filterLandsat(collection, start_date, end_date):
    return collection \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lte('CLOUD_COVER', 50)) \
        .filterBounds(buffer)

# ========== 2000-2002: Landsat-5 TM + Landsat-7 (SLC-on) ==========
print("正在处理 2000-2002: Landsat-5 TM + Landsat-7 (SLC-on)...")
l5_2000_2002 = filterLandsat(
    ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'),
    '2000-01-01', '2002-12-31'
).map(applyScaleFactorsL57).map(renameL57).map(maskL57Clouds)

l7_2000_2002 = filterLandsat(
    ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
    '2000-01-01', '2002-12-31'
).map(applyScaleFactorsL57).map(renameL57).map(maskL57Clouds)

landsat_2000_2002 = l5_2000_2002.merge(l7_2000_2002)


# ========== 2003-2011: 优先 Landsat-5 TM ==========
print("\n正在处理 2003-2011: 优先 Landsat-5 TM...")
l5_2003_2011 = filterLandsat(
    ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'),
    '2003-01-01', '2011-12-31'
).map(applyScaleFactorsL57).map(renameL57).map(maskL57Clouds)

# L7 仅用于填充(云量要求更严格)
l7_gap_fill = filterLandsat(
    ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
    '2003-01-01', '2011-12-31'
).filter(ee.Filter.lte('CLOUD_COVER', 1)) \
 .map(applyScaleFactorsL57).map(renameL57).map(maskL57Clouds)

landsat_2003_2011 = l5_2003_2011.merge(l7_gap_fill)


# ========== 2012: L7 (SLC-off) 多景填隙,允许 ±1 年滚动 ==========
print("\n正在处理 2012: L7 (SLC-off) + ±1年滚动复合...")
# 主要时间范围
l7_2012 = filterLandsat(
    ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
    '2012-01-01', '2012-12-31'
).map(applyScaleFactorsL57).map(renameL57).map(maskL57Clouds)

# ±1 年扩展范围(用于填充数据缺口)
l7_extended_2011 = filterLandsat(
    ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
    '2011-01-01', '2011-12-31'
).map(applyScaleFactorsL57).map(renameL57).map(maskL57Clouds)

l7_extended_2013 = filterLandsat(
    ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
    '2013-01-01', '2013-03-31'
).map(applyScaleFactorsL57).map(renameL57).map(maskL57Clouds)

landsat_2012 = l7_2012.merge(l7_extended_2011).merge(l7_extended_2013)

# ========== 2013: L7 (SLC-off) + L8 (2013.4开始) ==========
print("\n正在处理 2013: L7 (SLC-off) + L8 (2013.4起)...")
l7_2013 = filterLandsat(
    ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
    '2013-01-01', '2013-03-31'
).map(applyScaleFactorsL57).map(renameL57).map(maskL57Clouds)

# Landsat 8 从 2013-04-01 开始
l8_2013 = filterLandsat(
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'),
    '2013-04-01', '2013-12-31'
).map(applyScaleFactorsL89).map(renameL89).map(maskL89Clouds)

landsat_2013 = l7_2013.merge(l8_2013)

# ========== 2014-2022: Landsat-8 OLI ==========
print("\n正在处理 2014-2022: Landsat-8 OLI...")
l8_2014_2022 = filterLandsat(
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'),
    '2014-01-01', '2022-12-31'
).map(applyScaleFactorsL89).map(renameL89).map(maskL89Clouds)


# ========== 2023-2025: Landsat-8 OLI + Landsat-9 OLI-2 ==========
print("\n正在处理 2023-2025: Landsat-8 OLI + Landsat-9 OLI-2...")
l8_2023_2025 = filterLandsat(
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'),
    '2023-01-01', '2025-12-31'
).map(applyScaleFactorsL89).map(renameL89).map(maskL89Clouds)

# Landsat 9 从 2021-10-31 开始,但主要用于 2023-2025
l9_2023_2025 = filterLandsat(
    ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'),
    '2023-01-01', '2025-12-31'
).map(applyScaleFactorsL89).map(renameL89).map(maskL89Clouds)

landsat_2023_2025 = l8_2023_2025.merge(l9_2023_2025)


# ========== 合并所有时段 ==========
landsat_masked = landsat_2000_2002 \
    .merge(landsat_2003_2011) \
    .merge(landsat_2012) \
    .merge(landsat_2013) \
    .merge(l8_2014_2022) \
    .merge(landsat_2023_2025)

#print(f"\n合并后的总影像数量: {landsat_masked.size().getInfo()} 景")
print("\n数据源策略:")
print("  2000-2002: L5 TM + L7 SLC-on")
print("  2003-2011: 优先 L5 TM, L7 填隙")
print("  2012: L7 SLC-off + ±1年滚动")
print("  2013: L7 SLC-off (1-3月) + L8 OLI (4月起)")
print("  2014-2022: L8 OLI")
print("  2023-2025: L8 OLI + L9 OLI-2")

# ========== 计算光谱指数 ==========

# 1. NDWI (归一化差异水体指数)
def addNDWI(image):
    ndwi = image.normalizedDifference(['Green', 'NIR']).rename('NDWI')  # (Green - NIR) / (Green + NIR)
    return image.addBands(ndwi)

landsat_NDWI = landsat_masked.map(addNDWI)

# 2. NDVI (归一化差异植被指数)
def addNDVI(image):
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')  # (NIR - Red) / (NIR + Red)
    return image.addBands(ndvi)

landsat_NDVI = landsat_NDWI.map(addNDVI)

# 3. EVI (增强型植被指数)
def addEVI(image):
    evi = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)', {
            'NIR': image.select('NIR'),
            'RED': image.select('Red'),
            'BLUE': image.select('Blue')
        }).rename('EVI')
    return image.addBands(evi)

landsat_EVI = landsat_NDVI.map(addEVI)

# 4. mNDWI (修正归一化差异水体指数)
def addmNDWI(image):
    mndwi = image.normalizedDifference(['Green', 'SWIR1']).rename('mNDWI')  # (Green - SWIR1) / (Green + SWIR1)
    return image.addBands(mndwi)

landsat_mNDWI = landsat_EVI.map(addmNDWI)

# 5. MVI (红树林植被指数)
def addMVI(image):
    mvi = image.expression(
        '(NIR - GREEN) / (SWIR - GREEN)', {
            'NIR': image.select('NIR'),
            'GREEN': image.select('Green'),
            'SWIR': image.select('SWIR1')
        }).rename('MVI')
    return image.addBands(mvi)

landsat_MVI = landsat_mNDWI.map(addMVI)

# 6. AWEI_nsh (自动水体提取指数 - 无阴影)
def addAWEI_nsh(image):
    awei_nsh = image.expression(
        '4 * (GREEN - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)', {
            'GREEN': image.select('Green'),
            'SWIR1': image.select('SWIR1'),
            'NIR': image.select('NIR'),
            'SWIR2': image.select('SWIR2')
        }).rename('AWEI_nsh')
    return image.addBands(awei_nsh)

landsat_AWEI_nsh = landsat_MVI.map(addAWEI_nsh)

# 7. AWEI_sh (自动水体提取指数 - 含阴影)
def addAWEI_sh(image):
    awei_sh = image.expression(
        'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2', {
            'BLUE': image.select('Blue'),
            'GREEN': image.select('Green'),
            'NIR': image.select('NIR'),
            'SWIR1': image.select('SWIR1'),
            'SWIR2': image.select('SWIR2')
        }).rename('AWEI_sh')
    return image.addBands(awei_sh)

landsat_AWEI_sh = landsat_AWEI_nsh.map(addAWEI_sh)

# 8. NDMI (归一化差异湿度指数)
def addNDMI(image):
    ndmi = image.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')  # (NIR - SWIR1) / (NIR + SWIR1)
    return image.addBands(ndmi)

landsat_indices = landsat_AWEI_sh.map(addNDMI)
# 9. EMVI
def addEMVI(image):
    emvi = image.expression(
        '(GREEN - SWIR2) / (SWIR1 - GREEN)', {
            'GREEN': image.select('Green'),
            'SWIR1': image.select('SWIR1'),
            'SWIR2': image.select('SWIR2')
        }).rename('EMVI')
    return image.addBands(emvi)
landsat_indices = landsat_indices.map(addEMVI)

# 10. CMRI (NDVI - NDWI)
def addCMRI(image):
    cmri = image.select('NDVI').subtract(image.select('NDWI')).rename('CMRI').toFloat()
    return image.addBands(cmri)

landsat_indices = landsat_indices.map(addCMRI)
# 11. kNDVI (核归一化植被指数 - 使用tanh方法)
def addkNDVI(image):
    nir = image.select('NIR')
    red = image.select('Red')
    # 计算 (NIR - Red)^2
    D2 = nir.subtract(red).pow(2)
    # Sigma值需要根据数据调整，Landsat反射率使用0.15
    sigma = ee.Number(0.15)
    # 使用tanh方法计算kNDVI
    kndvi = D2.divide(sigma.multiply(2.0).pow(2)).tanh().rename('kNDVI')
    return image.addBands(kndvi)

landsat_indices = landsat_indices.map(addkNDVI)

print("光谱指数计算完成!")
print(f"最终影像集合包含波段: {landsat_indices.first().bandNames().getInfo()}")
# ========= 训练标签（0/1）与硬负样本（来自 GWL_FCS30） =========
# 0: Other (Background)
# 1: Mangrove (Target - 使用用户提供的 mangrove 二值影像)
# 2: Tidal Flat (Hard Negative - 来自 GWL)
# 3: Water (Hard Negative - 来自 GWL)

# 加载 GWL_FCS30 先验知识
gwl_fcs30 = ee.ImageCollection("projects/sat-io/open-datasets/GWL_FCS30")
gwl = gwl_fcs30.filterBounds(buffer).mosaic().clip(buffer).select([0], ['classification'])

# 定义先验类别
# 180: Water, 187: Tidal Flat (Wetland)
water_prior = gwl.eq(180)
tidal_prior = gwl.eq(187)

# 构建 4 类标签
# 优先级：Mangrove > Water/Tidal > Other
# mangrove_binary 已在前方定义 (mangrove.eq(1))
label = ee.Image(0).byte().rename('class') \
    .where(tidal_prior.And(mangrove_binary.Not()), 2) \
    .where(water_prior, 3) \
    .where(mangrove_binary, 1) \
    .clip(buffer)


# ========= 随机森林 & 抽样参数 =========
N_PER_CLASS = 12000
RF_TREES = 700
SEED = 42
TRAINING_YEAR = 2020  # 仅使用2020年数据进行训练

# ========= 训练随机森林模型（仅使用2020年数据）=========
print("=" * 50)
print(f"使用 {TRAINING_YEAR} 年数据训练随机森林模型...")
print("=" * 50)

year_start = f'{TRAINING_YEAR}-01-01'
year_end   = f'{TRAINING_YEAR}-12-31'
landsat_train = landsat_indices.filterDate(year_start, year_end)
print(f"\n训练总影像数量: {landsat_train.size().getInfo()} 景")

# --- 添加多时相统计特征计算 ---
print("正在计算多时相统计特征 (Percentiles & Seasonality)...")
# 1. 定义 Reducer
percentiles = [10, 50, 90]
combined_reducer = ee.Reducer.percentile(percentiles) \
    .combine(ee.Reducer.stdDev(), sharedInputs=True) \
    .combine(ee.Reducer.max(), sharedInputs=True) \
    .combine(ee.Reducer.min(), sharedInputs=True) \
    .combine(ee.Reducer.mean(), sharedInputs=True)

# 2. 计算统计值 (一次遍历)
landsat_stats = landsat_train.reduce(combined_reducer)

# 3. 提取季节性特征
# 振幅 (Amplitude)
ndvi_max = landsat_stats.select('NDVI_max').rename('NDVI_Annual_Max')
ndvi_min = landsat_stats.select('NDVI_min').rename('NDVI_Annual_Min')
ndvi_amplitude = ndvi_max.subtract(ndvi_min).rename('NDVI_Amplitude')

# 变异系数 (CoV)
ndvi_mean = landsat_stats.select('NDVI_mean')
ndvi_std = landsat_stats.select('NDVI_stdDev')
ndvi_cov = ndvi_std.divide(ndvi_mean.abs().add(0.001)).rename('NDVI_CoV')

# 季节性强度 (Seasonality)
ndvi_seasonality = ndvi_amplitude.divide(ndvi_mean.abs().add(0.001)).rename('NDVI_Seasonality')

seasonal_features = ee.Image.cat([
    ndvi_max, ndvi_min, ndvi_amplitude,
    ndvi_cov, ndvi_seasonality
]).unmask(0)

print("正在添加地形和距离特征...")

# # 1. DEM 高程和坡度 (ALOS AW3D30)
alos_demCol = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").filterBounds(buffer).select('DSM')
alos_elevation = alos_demCol.mosaic().clip(buffer).unmask(0)

# 获取原始投影以确保坡度计算正确
proj = alos_demCol.first().select(0).projection()

# 在正确的投影下计算坡度
alos_slope = ee.Terrain.slope(alos_elevation.setDefaultProjection(proj)).rename('slope')

# 合并高程和坡度作为地形特征
topo_features = alos_elevation.rename('elevation').addBands(alos_slope)
# # 2. 到常年水体的距离
jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
permWater = jrc.gte(90)  # 常水阈值:90%以上出现频率
dist2water = permWater.fastDistanceTransform(256).sqrt().multiply(30).rename('d2water').toFloat()  # 转换为米
# qualityMosaic 合成
ndvi_best = landsat_train.qualityMosaic('NDVI')
# 从 NDVI 最优场景中提取所有指数
NDVI_mosaic  = ndvi_best.select('NDVI').toFloat()
NDWI_mosaic  = ndvi_best.select('NDWI').toFloat()
EVI_mosaic   = ndvi_best.select('EVI').toFloat()
mNDWI_mosaic = ndvi_best.select('mNDWI').toFloat()
MVI_mosaic   = ndvi_best.select('MVI').toFloat()
AWEI_nsh_mosaic = ndvi_best.select('AWEI_nsh').toFloat()
AWEI_sh_mosaic = ndvi_best.select('AWEI_sh').toFloat()
NDMI_mosaic = ndvi_best.select('NDMI').toFloat()
EMVI_mosaic = ndvi_best.select('EMVI').toFloat()
CMRI_mosaic = ndvi_best.select('CMRI').toFloat()
kNDVI_mosaic = ndvi_best.select('kNDVI').toFloat()
# # 原始光谱波段(使用中值合成)
spectral_median = landsat_train.select(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']).median().toFloat()
# # 使用 spectral_median 中的 NIR 波段
nir_band = spectral_median.select('NIR').multiply(255).toInt() 
# # 计算 GLCM 纹理特征
glcm = nir_band.glcmTexture(size=5) # size=5 表示 11x11 的窗口
# # 选择一个或多个纹理特征，'contrast' 是一个常用的特征
contrast = glcm.select(['NIR_contrast', 'NIR_ent', 'NIR_corr'])
# 最终特征栈
predictors_train = ee.Image.cat([
    NDWI_mosaic, 
    NDVI_mosaic, 
    EVI_mosaic, 
    mNDWI_mosaic,
    MVI_mosaic,
    AWEI_nsh_mosaic,  
    AWEI_sh_mosaic,   
    NDMI_mosaic,
    EMVI_mosaic,
    CMRI_mosaic,
    kNDVI_mosaic,
    spectral_median,
    contrast,
    topo_features,        # 高程
    dist2water,  # 到常年水体距离
    seasonal_features, # 季节性特征
    landsat_stats.select(['NDVI_p50', 'NDVI_stdDev', 'mNDWI_p50', 'EVI_p50']) # 关键统计特征
]).clip(buffer)
snic = ee.Algorithms.Image.Segmentation.SNIC(
    image=predictors_train.select(['Red','Green','Blue','NIR','SWIR1','SWIR2']),
    size=8, compactness=1, connectivity=8, neighborhoodSize=128
)
obj_means = snic.select('.*_mean')  # 各波段的簇内均值
predictors_train = predictors_train.addBands(obj_means, None, True)
stack = predictors_train.addBands(label)

# 分层抽样（4类抽样，含硬负样本）
print(f'抽样训练数据 (4类: Other, Mangrove, Tidal, Water)...')
samples = stack.stratifiedSample(
    numPoints=0,
    classBand='class',
    classValues=[0, 1, 2, 3],
    classPoints=[6000, 12000, 4000, 4000], # 调整样本量: 背景, 红树林, 潮滩, 水体
    region=buffer.geometry(),
    scale=30,
    seed=SEED,
    geometries=False,
    dropNulls=True,
    tileScale=8
)

# --- 样本光谱清洗 ---
print("执行样本光谱清洗...")
def clean_samples(f):
    ndvi = ee.Number(f.get('NDVI')) # 使用 NDVI_mosaic (注意波段名是否匹配，mosaic中名为 NDVI)
    cls = ee.Number(f.get('class'))
    
    # 1. 红树林 (Class 1): NDVI 应该较高 (例如 > 0.25 或 0.3)
    # 降低一点阈值以防误删，设为 0.25
    keep_mangrove = cls.eq(1).And(ndvi.gt(0.25))
    
    # 2. 非植被硬负样本 (Class 2 Tidal, Class 3 Water): NDVI 应该较低 (< 0.25)
    keep_non_veg = cls.gt(1).And(ndvi.lt(0.25))
    
    # 3. 其他背景 (Class 0): 不做假设
    keep_other = cls.eq(0)
    
    keep = keep_mangrove.Or(keep_non_veg).Or(keep_other)
    return f.set('keep', ee.Algorithms.If(keep, 1, 0))

samples = samples.map(clean_samples).filter(ee.Filter.eq('keep', 1))

# --- 类别重映射 (2, 3 -> 0) ---
print("重映射类别回二分类 (0 vs 1)...")
def remap_class(f):
    cls = ee.Number(f.get('class'))
    # 将 > 1 的类别 (Tidal, Water) 归为 0 (Non-Mangrove)
    new_cls = ee.Algorithms.If(cls.gt(1), 0, cls)
    return f.set('class', new_cls)

samples = samples.map(remap_class)

# 切分训练集/验证集（80/20）
samples = samples.randomColumn('rand', SEED)
train = samples.filter(ee.Filter.lt('rand', 0.8))
test  = samples.filter(ee.Filter.gte('rand', 0.8))

# 训练随机森林分类器
print(f'训练随机森林模型...')
feat_bands = predictors_train.bandNames()
rf = ee.Classifier.smileRandomForest(
    numberOfTrees=RF_TREES,
    #minLeafPopulation=10, # 设置最小叶节点样本数以防过拟合
    bagFraction=0.6,
    seed=SEED
).train(
    features=train, 
    classProperty='class', 
    inputProperties=feat_bands
)
#importance = rf.explain()
#importance_dict = importance.getInfo()
print(f'评估模型精度...')
test_pred = test.classify(rf)
cm = test_pred.errorMatrix('class', 'classification')
cm_array = cm.array()

# 获取生产者精度和消费者精度数组
producers = cm.producersAccuracy()
consumers = cm.consumersAccuracy()

# 计算红树林类(类别1)的精确率、召回率和F1分数
# 混淆矩阵: [[TN, FP], [FN, TP]]
TP = cm_array.get([1, 1])  # 真正例(红树林正确分类为红树林)
FP = cm_array.get([0, 1])  # 假正例(非红树林错误分类为红树林)
FN = cm_array.get([1, 0])  # 假负例(红树林错误分类为非红树林)
TN = cm_array.get([0, 0])  # 真负例(非红树林正确分类为非红树林)

# 计算精确率 Precision = TP / (TP + FP)
precision_mangrove = ee.Number(TP).divide(ee.Number(TP).add(FP))

# 计算召回率 Recall = TP / (TP + FN)
recall_mangrove = ee.Number(TP).divide(ee.Number(TP).add(FN))

# 计算 F1 分数 = 2 * (Precision * Recall) / (Precision + Recall)
f1_mangrove = precision_mangrove.multiply(recall_mangrove).multiply(2) \
    .divide(precision_mangrove.add(recall_mangrove))

# ========== 计算 AUC-ROC 和 AUC-PR ==========
print('计算 AUC-ROC 和 AUC-PR...')

# 获取概率分类器
rf_prob = rf.setOutputMode('PROBABILITY')

# 对测试集进行概率预测
test_prob = test.classify(rf_prob)

# 定义阈值序列(从0到1分成25个间隔)
thresholds = ee.List.sequence(0, 1, 0.04)  # 25个阈值

def calculate_metrics_at_threshold(cutoff):
    """在特定阈值下计算各种指标"""
    cutoff = ee.Number(cutoff)
    
    # 观察到的正样本(红树林)
    pres = test_prob.filter(ee.Filter.eq('class', 1))
    # 观察到的负样本(非红树林)
    abs_samples = test_prob.filter(ee.Filter.eq('class', 0))
    
    # TP: 真正例(概率 > cutoff 且实际为正样本)
    tp = pres.filter(ee.Filter.gte('classification', cutoff)).size()
    
    # FN: 假负例(概率 < cutoff 且实际为正样本)
    fn = pres.filter(ee.Filter.lt('classification', cutoff)).size()
    
    # TN: 真负例(概率 < cutoff 且实际为负样本)
    tn = abs_samples.filter(ee.Filter.lt('classification', cutoff)).size()
    
    # FP: 假正例(概率 > cutoff 且实际为负样本)
    fp = abs_samples.filter(ee.Filter.gte('classification', cutoff)).size()
    
    # TPR (True Positive Rate) = Recall = Sensitivity = TP / (TP + FN)
    tpr = ee.Number(tp).divide(pres.size())
    
    # FPR (False Positive Rate) = FP / (FP + TN)
    fpr = ee.Number(fp).divide(abs_samples.size())
    
    # Precision = TP / (TP + FP)
    precision = ee.Number(tp).divide(ee.Number(tp).add(fp))
    
    return ee.Feature(None, {
        'cutoff': cutoff,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TPR': tpr,
        'FPR': fpr,
        'Precision': precision
    })

# 计算所有阈值下的指标
metrics_collection = ee.FeatureCollection(thresholds.map(calculate_metrics_at_threshold))

# 计算 AUC-ROC
fpr_array = ee.Array(metrics_collection.aggregate_array('FPR'))
tpr_array = ee.Array(metrics_collection.aggregate_array('TPR'))

# 使用梯形法则计算曲线下面积
fpr_diff = fpr_array.slice(0, 1).subtract(fpr_array.slice(0, 0, -1))
tpr_sum = tpr_array.slice(0, 1).add(tpr_array.slice(0, 0, -1))
auc_roc = fpr_diff.multiply(tpr_sum).multiply(0.5).reduce('sum', [0]).abs().toList().get(0)

# 计算 AUC-PR
tpr_array_pr = ee.Array(metrics_collection.aggregate_array('TPR'))
precision_array = ee.Array(metrics_collection.aggregate_array('Precision'))

# 使用梯形法则计算曲线下面积
tpr_diff = tpr_array_pr.slice(0, 1).subtract(tpr_array_pr.slice(0, 0, -1))
precision_sum = precision_array.slice(0, 1).add(precision_array.slice(0, 0, -1))
auc_pr = tpr_diff.multiply(precision_sum).multiply(0.5).reduce('sum', [0]).abs().toList().get(0)

# 创建精度指标特征集合
metrics_feature = ee.Feature(None, {
    'year': TRAINING_YEAR,
    'overall_accuracy': cm.accuracy(),
    'kappa': cm.kappa(),
    'producers_accuracy_0': producers.toList().flatten().get(0),
    'producers_accuracy_1': producers.toList().flatten().get(1),
    'consumers_accuracy_0': consumers.toList().flatten().get(0),
    'consumers_accuracy_1': consumers.toList().flatten().get(1),
    'precision_mangrove': precision_mangrove,
    'recall_mangrove': recall_mangrove,
    'f1_mangrove': f1_mangrove,
    'auc_roc': auc_roc,
    'auc_pr': auc_pr,
    'confusion_matrix_00': cm_array.get([0, 0]),
    'confusion_matrix_01': cm_array.get([0, 1]),
    'confusion_matrix_10': cm_array.get([1, 0]),
    'confusion_matrix_11': cm_array.get([1, 1])
})

# 导出训练年份的精度指标
task_cm = ee.batch.Export.table.toDrive(
    collection=ee.FeatureCollection([metrics_feature]),
    description=f'RF_metrics_training_{TRAINING_YEAR}_1000m',
    folder='GEE_Mangrove_RF',
    fileNamePrefix=f'metrics_training_{TRAINING_YEAR}_1000m',
    fileFormat='CSV'
)
#task_cm.start()
print(f'{TRAINING_YEAR} 年训练集精度指标导出任务已启动')

# ========== 使用训练好的模型预测2000-2024年红树林分布 ==========
print("=" * 50)
print("开始预测 2000-2024 年红树林分布...")
print("=" * 50)

CONFIDENCE_THRESHOLD = 0.90  # 置信度阈值 90%

# 获取概率分类器
rf_prob = rf.setOutputMode('PROBABILITY')

# 定义年份列表
years = [2000,2005,2010,2015,2020,2024]

def build_predictors_for_year(year):
    """为指定年份构建预测特征栈"""
    year_start = f'{year}-01-01'
    year_end = f'{year}-12-31'
    
    # 筛选该年份的影像
    landsat_year = landsat_indices.filterDate(year_start, year_end)
    
    # qualityMosaic 合成
    ndvi_best = landsat_year.qualityMosaic('NDVI')
    
    # 从 NDVI 最优场景中提取所有指数
    NDVI_mosaic = ndvi_best.select('NDVI').toFloat()
    NDWI_mosaic = ndvi_best.select('NDWI').toFloat()
    EVI_mosaic = ndvi_best.select('EVI').toFloat()
    mNDWI_mosaic = ndvi_best.select('mNDWI').toFloat()
    MVI_mosaic = ndvi_best.select('MVI').toFloat()
    AWEI_nsh_mosaic = ndvi_best.select('AWEI_nsh').toFloat()
    AWEI_sh_mosaic = ndvi_best.select('AWEI_sh').toFloat()
    NDMI_mosaic = ndvi_best.select('NDMI').toFloat()
    EMVI_mosaic = ndvi_best.select('EMVI').toFloat()
    CMRI_mosaic = ndvi_best.select('CMRI').toFloat()
    kNDVI_mosaic = ndvi_best.select('kNDVI').toFloat()
    
    # 原始光谱波段(使用中值合成)
    spectral_median = landsat_year.select(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']).median().toFloat()
    
    # GLCM 纹理特征
    nir_band = spectral_median.select('NIR').multiply(255).toInt()
    glcm = nir_band.glcmTexture(size=4)
    contrast = glcm.select(['NIR_contrast', 'NIR_ent', 'NIR_corr'])
    
    # --- 增加统计特征计算 (与训练保持一致) ---
    landsat_stats_year = landsat_year.reduce(combined_reducer)

    ndvi_max = landsat_stats_year.select('NDVI_max').rename('NDVI_Annual_Max')
    ndvi_min = landsat_stats_year.select('NDVI_min').rename('NDVI_Annual_Min')
    ndvi_amplitude = ndvi_max.subtract(ndvi_min).rename('NDVI_Amplitude')

    ndvi_mean = landsat_stats_year.select('NDVI_mean')
    ndvi_std = landsat_stats_year.select('NDVI_stdDev')
    ndvi_cov = ndvi_std.divide(ndvi_mean.abs().add(0.001)).rename('NDVI_CoV')
    ndvi_seasonality = ndvi_amplitude.divide(ndvi_mean.abs().add(0.001)).rename('NDVI_Seasonality')

    seasonal_features_year = ee.Image.cat([
        ndvi_max, ndvi_min, ndvi_amplitude,
        ndvi_cov, ndvi_seasonality
    ]).unmask(0)

    # 构建特征栈
    predictors = ee.Image.cat([
        NDWI_mosaic,
        NDVI_mosaic,
        EVI_mosaic,
        mNDWI_mosaic,
        MVI_mosaic,
        AWEI_nsh_mosaic,
        AWEI_sh_mosaic,
        NDMI_mosaic,
        EMVI_mosaic,
        CMRI_mosaic,
        kNDVI_mosaic,
        spectral_median,
        contrast,
        topo_features,
        dist2water,
        seasonal_features_year,
        landsat_stats_year.select(['NDVI_p50', 'NDVI_stdDev', 'mNDWI_p50', 'EVI_p50'])
    ]).clip(buffer)
    
    # SNIC 分割
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=predictors.select(['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2']),
        size=8, compactness=1, connectivity=8, neighborhoodSize=128
    )
    obj_means = snic.select('.*_mean')
    predictors = predictors.addBands(obj_means, None, True)
    
    # 返回预测特征和光谱指数
    indices_stack = ee.Image.cat([
        NDVI_mosaic.rename('NDVI'),
        EVI_mosaic.rename('EVI'),
        MVI_mosaic.rename('MVI'),
        EMVI_mosaic.rename('EMVI'),
        CMRI_mosaic.rename('CMRI'),
        kNDVI_mosaic.rename('kNDVI')
    ])
    
    return predictors, indices_stack

# 批量预测和导出
for year in years:
    print(f'\n处理 {year} 年...')
    
    # 构建该年份的预测特征
    predictors_year, indices_year = build_predictors_for_year(year)
    
    # 进行概率预测
    prob_map = predictors_year.classify(rf_prob).rename('probability')
    
    # 应用置信度阈值 (>90%)
    classified = prob_map.gte(CONFIDENCE_THRESHOLD).rename('mangrove').byte()

    # Post-processing

    # Majority Filter
    # Radius 1, kernel circle
    kernel = ee.Kernel.circle(radius=1)
    smoothed = classified.focal_mode(kernel=kernel, iterations=1)

    # Sieve (Optional per plan, implementing safe version)
    # connectedPixelCount
    min_size = 3 # Example size
    smoothed = smoothed.updateMask(smoothed.connectedPixelCount(min_size, True).gte(min_size))
    
    mangrove_prediction = smoothed
    
    # 创建红树林掩膜
    mangrove_mask = mangrove_prediction.eq(1)
    
    # 提取光谱指数（仅红树林像元有值，其他为NoData）
    year_start = f'{year}-01-01'
    year_end = f'{year}-12-31'
    landsat_year = landsat_indices.filterDate(year_start, year_end)
    ndvi_best = landsat_year.qualityMosaic('NDVI')
    
    indices_stack = ee.Image.cat([
        ndvi_best.select('NDVI').toFloat(),
        ndvi_best.select('EVI').toFloat(),
        ndvi_best.select('MVI').toFloat(),
        ndvi_best.select('EMVI').toFloat(),
        ndvi_best.select('CMRI').toFloat(),
        ndvi_best.select('kNDVI').toFloat()
    ]).updateMask(mangrove_mask)
    
    # 合并：分类结果(全域) + 光谱指数(仅红树林像元)
    combined = mangrove_prediction.toFloat().addBands(indices_stack)
    
    # 导出合并后的多波段影像
    task_combined = ee.batch.Export.image.toDrive(
        image=combined,
        description=f'Mangrove_{year}',
        folder='GEE_Mangrove_RF',
        fileNamePrefix=f'mangrove_{year}',
        region=buffer.geometry(),
        scale=30,
        crs='EPSG:4326',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    #task_combined.start()
    print(f'  {year} 年合并导出任务已启动 (波段: mangrove, NDVI, EVI, MVI, EMVI, CMRI)')

print("\n" + "=" * 50)
print("所有年份的导出任务已启动!")
print(f"共 {len(years) * 2} 个任务 (每年: 二值分类 + 光谱指数)")
print("光谱指数文件包含5个波段: NDVI, EVI, MVI, EMVI, CMRI")
print("请在 Google Earth Engine Tasks 面板查看进度")
print("文件将保存到 Google Drive 的 GEE_Mangrove_RF 文件夹")
print("=" * 50)