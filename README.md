1. 先运行data_loader.py来全量同步mysql中的数据
2.  
第一步：核心训练 (Foundation)
    运行文件： train_piml.py
    作用： 训练核心的物理信息神经网络 (PIML)。
    输出产物： 在 model/piml 目录下保存 best_piml_model.pth。
    重要性： 这是整个项目的基石，后续的 analysis_suite.py 依赖此文件。
第二步：深度分析与全量模型 (Deep Analysis)
    运行文件： advanced_analysis.py
    作用： 进行特征重要性分析、t-SNE 降维可视化，并使用全部数据训练一个"完整模型"。
    输出产物： 保存 piml_full_model.pth。
    重要性： 第四步的逆向设计脚本必须依赖这个 piml_full_model.pth 才能运行。
第三步：基准对比与评估 (Benchmarking)
    此步骤包含两个脚本，运行顺序不分先后，但必须在第一步之后：
    train_baseline_dnn.py：
    作用： 训练一个普通的 DNN（不带物理公式）作为对照组，证明 PIML 的优越性。
    输出： best_baseline_dnn.pth 和对比数据。
    analysis_suite.py：
    依赖： 它会直接从 train_piml.py 导入数据处理函数 (load_and_preprocess_data)，并加载第一步生成的 best_piml_model.pth。
    作用： 将 PIML 模型与随机森林 (RF)、XGBoost 进行对比，并进行虚拟筛选 (Virtual Screening)。

第四步：逆向设计与理论验证 (Discovery)
    运行文件： inverse_design_and_theory.py
    依赖： 必须读取第二步生成的 piml_full_model.pth。
    作用： 使用遗传算法 (GA) 反向设计出最优的材料配方，并验证晶格应变理论。
独立模块：实验室模拟 (Simulation)
    运行文件： lab_application_suite.py
    说明： 这是一个独立的演示套件，用于模拟“主动学习” (Active Learning) 和“不确定性量化” (UQ)。
    依赖： 它不依赖上述生成的 .pth 文件（它会在运行时快速训练临时模型），但它是对上述方法的应用演示。