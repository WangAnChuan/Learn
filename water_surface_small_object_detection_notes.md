# 水面小目标检测文献笔记清单

## 1. 研究主题
**方向**：水面/海面小目标检测  
**相关场景**：
- 海上搜救
- 船只检测
- 漂浮物/障碍物检测
- 红外海面弱小目标检测
- 可见光/红外多模态检测

---

## 2. 任务特点与难点

### 2.1 主要难点
- 海浪纹理复杂，背景噪声强
- 水面反光、强散射、低对比度
- 雾霾、低照度、逆光等恶劣环境
- 目标尺度极小，通常仅几个到几十个像素
- 目标易出现在海天线附近，背景与目标混叠严重
- 不同平台（岸基、USV、UAV）成像视角差异大
- 模型容易出现漏检、误检和泛化能力差的问题

### 2.2 关键研究问题
- 如何保留小目标细节特征
- 如何增强高分辨率浅层特征
- 如何减少复杂海面背景造成的误警
- 如何提高跨场景、跨海况、跨数据集泛化能力
- 如何兼顾精度与实时性部署

---

## 3. 文献阅读主线

建议按以下四条主线阅读：

1. **综述与总体脉络**
2. **可见光海面小目标检测**
3. **红外海面弱小目标检测**
4. **多模态融合与最新趋势**

---

## 4. 必读综述

### 4.1 海事小目标检测总综述
**A Guide to Image- and Video-Based Small Object Detection Using Deep Learning: Case Study of Maritime Surveillance**  
- 来源：IEEE TITS 2025
- 作用：
  - 适合作为整体综述入口
  - 系统梳理海事小目标检测方法、数据集、评测和挑战
- 阅读重点：
  - 海事场景与通用小目标检测的差异
  - 海事数据集分类
  - 深度学习方法演化路线

### 4.2 红外小目标检测综述
**Infrared Dim Small Target Detection Networks: A Review**  
- 来源：Sensors 2024
- 作用：
  - 适合补红外弱小目标基础
  - 总结红外小目标检测中的关键问题
- 阅读重点：
  - 小目标表征
  - 虚警与漏警平衡
  - 复杂背景适应
  - 轻量化部署

### 4.3 通用小目标检测综述
**Small Object Detection: A Comprehensive Survey on Challenges, Techniques, and Real-World Applications**  
- 来源：2025
- 作用：
  - 从更广视角理解小目标检测通用技术
- 阅读重点：
  - 多尺度特征
  - 注意力机制
  - Transformer
  - 超分辨
  - 域适应
  - 实时性优化

---

## 5. 重点数据集

### 5.1 Singapore Maritime Dataset (SMD)
- 类型：经典海事检测数据集
- 特点：
  - 提供 VIS / NIR 视频
  - 常用于海事目标检测与跟踪
- 适用：
  - 早期基线复现
  - 可见光与近红外对比研究

### 5.2 SeaDronesSee
- 类型：UAV 海上搜救数据集
- 特点：
  - 包含检测、单目标跟踪、多目标跟踪任务
  - 小目标、远距离目标较多
- 适用：
  - 搜救场景
  - 远距离海面目标检测
  - 检测与跟踪联合研究

### 5.3 MODD / MODD2
- 类型：USV 海面障碍物检测数据集
- 特点：
  - 同时涉及天空/岸边/海面分割与障碍物检测
  - 强调可航区域与海天线问题
- 适用：
  - 海天线先验
  - 可航区域约束
  - 漂浮障碍物检测

### 5.4 LaRS
- 类型：多场景海事障碍物 benchmark
- 特点：
  - 覆盖 lakes / rivers / seas
  - 强调环境多样性与场景复杂度
- 适用：
  - 泛化研究
  - 多环境鲁棒性评估

### 5.5 PoLaRIS
- 类型：较新的海事检测/跟踪数据集
- 特点：
  - 强调超小目标（如 10×10 像素级）
- 适用：
  - 极小目标检测
  - 检测-跟踪联合任务

---

## 6. 代表论文清单

## 6.1 可见光/通用海面小目标检测

### [1] A Benchmark for Deep Learning Based Object Detection in Maritime Environments
- 来源：CVPRW 2019
- 定位：海事目标检测 benchmark 基础文献
- 关键词：
  - benchmark
  - maritime detection
  - baseline
- 作用：
  - 理解海事目标检测早期研究框架
  - 适合作为数据集和基线起点

### [2] S-DETR: A Transformer Model for Real-Time Detection of Marine Ships
- 来源：JMSE 2023
- 定位：Transformer/DETR 应用于海事检测
- 关键词：
  - DETR
  - Transformer
  - real-time
- 阅读重点：
  - DETR 在多尺度海上目标中的适应方式
  - 全局上下文建模能力

### [3] An Efficient Model for Small Object Detection in the Maritime Environment
- 来源：Pattern Recognition Letters 2024
- 定位：海事小目标高效检测
- 关键词：
  - small object
  - lightweight
  - real-time
- 阅读重点：
  - 小目标检测头设计
  - 浅层细节保留
  - 海天线附近目标检测优化

### [4] Spotlight on Small-scale Ship Detection: Empowering YOLO with Advanced Techniques and a Novel Dataset
- 来源：ACCV 2024
- 定位：YOLO 改进路线代表作
- 关键词：
  - YOLO
  - small-scale ship
  - dataset
- 阅读重点：
  - iShip-1 数据集
  - 小目标增强策略
  - NWD loss
  - backbone/neck 改进

### [5] MSO-DETR: A Lightweight Detection Transformer Model for Small Object Detection in Maritime Search and Rescue
- 来源：Electronics 2025
- 定位：轻量化 RT-DETR/DETR 路线
- 关键词：
  - lightweight
  - RT-DETR
  - search and rescue
- 阅读重点：
  - 轻量主干
  - 多尺度融合
  - 实时检测性能

---

## 6.2 红外海面弱小目标检测

### [6] Infrared maritime dim small target detection based on spatiotemporal cues and multidirectional morphological filtering
- 来源：2021
- 定位：传统方法代表
- 关键词：
  - infrared
  - dim small target
  - spatiotemporal
- 阅读重点：
  - 时空信息利用
  - 海杂波抑制
  - 多方向形态学滤波

### [7] Review of Infrared Sea Surface Small-target Detection Algorithm
- 来源：2022
- 定位：红外海面小目标算法综述
- 关键词：
  - infrared sea surface
  - review
- 阅读重点：
  - 传统方法与深度学习方法对比
  - 海面红外检测任务特点

### [8] Infrared maritime small target detection network based on attention and partial learning convolution (APLCnet)
- 来源：Infrared Physics & Technology 2025
- 定位：红外深度学习方法
- 关键词：
  - attention
  - infrared maritime
  - small target
- 阅读重点：
  - 注意力机制
  - 低信噪比小目标增强
  - 红外海面背景抑制

### [9] Rethinking Evaluation of Infrared Small Target Detection
- 来源：NeurIPS 2025 Poster
- 定位：评测协议反思
- 关键词：
  - evaluation
  - target-level metric
  - cross-dataset
- 阅读重点：
  - pixel-level 与 target-level 联合评估
  - cross-dataset evaluation
  - 评测体系改进思路

---

## 6.3 多模态融合方向

### 研究趋势
- 可见光 + 红外融合正在成为重要方向
- 核心问题从“简单拼接”转向：
  - 弱对齐鲁棒性
  - 选择性特征融合
  - 轻量化部署
  - 复杂天气下稳健性提升

### 适合重点关注的研究点
- visible-infrared fusion for small object detection
- 弱对齐跨模态融合
- 海面复杂天气多模态感知
- 检测与融合模块联合优化

---

## 7. 最新进展总结（适合写到综述或开题报告）

### 7.1 检测框架演进
当前主要有两条路线：
1. **YOLO 系改进**
   - 改 backbone、neck、loss
   - 增加 P2 检测头
   - 强化小目标增强
2. **DETR / RT-DETR 路线**
   - 更强的全局上下文建模
   - 更适合复杂背景和多尺度目标
   - 同时追求实时性

### 7.2 小目标友好结构成为主流
- 加入高分辨率特征层
- 增加浅层细节分支
- 强化跨层特征融合
- 更适合几像素到十几像素的小目标

### 7.3 评测方式更强调鲁棒性
研究不再只看 mAP，而是逐渐关注：
- 小目标召回率
- 虚警率/误警率
- 跨海况泛化能力
- 跨数据集测试
- 不同距离/尺度分层评测

### 7.4 数据集建设更加多样化
新趋势包括：
- 检测 + 跟踪联合
- 检测 + 分割联合
- 多场景、多海况、多平台
- 更重视真实部署场景

### 7.5 多模态是未来重点
- 低照度
- 雾霾
- 逆光
- 强反光
- 远距离小目标

这些条件下，多模态往往优于单可见光方法。

---

## 8. 可选研究切入点

### 8.1 海天线先验 + 小目标检测
- 动机：
  - 很多小目标集中出现在海天线附近
- 可做内容：
  - 海天线估计
  - 海面区域先验
  - 检测与场景先验联合建模

### 8.2 可见光-红外融合小目标检测
- 动机：
  - 单模态在低照、雾霾、强反光下性能受限
- 可做内容：
  - 跨模态对齐
  - 选择性融合
  - 模态置信度引导
  - 轻量部署

### 8.3 时序信息辅助检测
- 动机：
  - 海浪反光造成单帧大量伪目标
- 可做内容：
  - temporal fusion
  - detection + tracking
  - 短时序稳定性约束

### 8.4 面向真实场景的评测体系
- 动机：
  - 单一 mAP 不能充分反映模型实用性
- 可做内容：
  - 漏检/误检分析
  - 小目标专属指标
  - 跨数据集评测
  - 海况分层评测

---

## 9. 关键词检索清单

### 9.1 中文关键词
- 海面小目标检测
- 水面小目标检测
- 海上小目标检测
- 水面漂浮物检测 小目标
- 海上搜救 小目标检测
- 海面红外弱小目标检测
- 海事障碍物检测 小目标
- 可见光 红外 融合 海面 小目标检测

### 9.2 英文关键词
- maritime small object detection
- sea-surface small target detection
- water surface small object detection
- maritime obstacle detection small objects
- infrared maritime small target detection
- visible infrared fusion maritime small object detection
- RT-DETR maritime small object detection
- SeaDronesSee small object detection
- PoLaRIS maritime object detection

---

## 10. 推荐阅读顺序

### 第一阶段：先搭框架
1. 海事小目标检测总综述
2. 红外小目标综述
3. 通用小目标检测综述

### 第二阶段：看数据集与基线
4. SMD benchmark
5. SeaDronesSee
6. MODD / MODD2
7. LaRS
8. PoLaRIS

### 第三阶段：看方法
9. S-DETR
10. PRL 2024 高效小目标检测模型
11. ACCV 2024 小尺度船只检测
12. MSO-DETR 2025
13. APLCnet 2025

### 第四阶段：准备自己的研究设计
14. 对比 YOLO 路线与 DETR 路线
15. 总结已有方法在海天线、小尺度、多模态、时序建模上的不足
16. 确定自己的创新点与实验方案

---

## 11. 后续阅读笔记模板

### 论文标题：
### 来源：
### 发表年份：
### 任务类型：
- [ ] 可见光检测
- [ ] 红外检测
- [ ] 多模态融合
- [ ] 检测+跟踪
- [ ] 障碍物检测
- [ ] 评测方法

### 论文核心问题：
### 方法概述：
### 主要创新点：
1.
2.
3.

### 使用数据集：
### 评价指标：
### 主要实验结果：
### 优点：
### 局限性：
### 对我研究的启发：

---

## 12. 我当前的初步研究判断

### 当前领域热点
- 小目标友好的多尺度结构设计
- 轻量化实时检测
- RT-DETR/Transformer 在海事场景中的应用
- 可见光-红外融合
- 时序建模
- 更合理的评测体系

### 更有潜力的创新方向
- 海天线/海面区域先验引导的小目标检测
- 可见光-红外融合的小目标检测
- 检测与时序稳定性联合建模
- 面向真实海况的跨数据集泛化研究

---

## 13. 待办清单

### 文献阅读
- [ ] 先读 3 篇综述
- [ ] 了解 5 个核心数据集
- [ ] 精读 5 篇代表方法论文
- [ ] 记录各论文的小目标增强策略
- [ ] 总结各论文实验设置与不足

### 实验准备
- [ ] 确定主数据集
- [ ] 选择 baseline（YOLO/RT-DETR）
- [ ] 设计小目标增强模块
- [ ] 设计消融实验
- [ ] 设计跨数据集测试

### 写作准备
- [ ] 整理 related work
- [ ] 形成问题定义
- [ ] 提炼创新点
- [ ] 准备研究路线图

---