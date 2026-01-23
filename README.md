1. 先运行data_loader.py来全量同步mysql中的数据
2. 
Step 0: 初始化 (Initialization)
    运行文件: 00_init_dir.py
    作用: 初始化项目的目录结构，创建 results/, models/, logs/ 等必要的输出文件夹。
Step 1: 核心训练 (Foundation)
    运行文件: 01_train_physics_model.py
    作用: 训练核心的 物理信息神经网络 (PIML)。
    输出: 在 results/checkpoint/piml/ 下保存 best_piml_model.pth。
    重要性: 整个项目的基石，后续所有分析和发现模块都依赖此模型。
Step 2: 深度分析与全量模型 (Deep Analysis)
    运行文件: 02_interpret_mechanisms.py
    作用: 进行特征重要性分析、t-SNE 降维可视化，并使用全量数据微调模型。
    输出: 保存用于解释性的图表和 piml_full_model.pth。
Step 3: 基准对比与评估 (Benchmarking)
    此步骤包含两个脚本，用于证明模型优越性：
        03a_train_baseline_model.py:
        作用: 训练普通的 DNN（无物理约束）作为对照组。
        输出: best_baseline_dnn.pth。
        03b_evaluate_benchmarks.py:
        作用: 将 PIML 与 DNN、Random Forest (RF)、XGBoost 进行横向对比，并进行虚拟筛选测试。
Step 4: 逆向设计与理论验证 (Discovery)
    运行文件: 04_discover_materials.py
    依赖: 需要 Step 1 或 Step 2 生成的模型。
    作用: 使用 遗传算法 (GA) 进行共掺杂配方的逆向设计，寻找高电导率的新材料，并验证“晶格应变理论”。
    输出: 生成最佳配方文件 results/ai_discovery_best_recipe.csv。
Step 5: 实验室模拟 (Active Learning Demo)
    运行文件: 05_simulate_lab_experiments.py
    说明: 这是一个独立的演示模块。
    作用: 模拟“主动学习” (Active Learning) 闭环和“不确定性量化” (UQ)，演示 AI 如何辅助减少实验次数。
Step 6: ML 稳定性初筛 (Stability Check)
    运行文件: 06_verify_stability.py
    作用: 使用机器学习模型快速预测新材料的相稳定性，剔除不稳定的候选配方。
Step 7: 分子动力学验证 (MD with CHGNet)
    运行文件: 07_computational_validation.py
    技术: 集成 CHGNet (通用图神经网络势函数)。
    作用: 构建超胞，在高温下 (如 1500K) 运行分子动力学模拟，通过计算氧离子的均方位移 (MSD) 直接推算扩散系数和电导率，与 PIML 预测值进行“虚实对比”。
    输出: 生成 PIML vs MD 的校验图表 results/images/paper_computational_validation.png。
Step 8: 第一性原理验证 (DFT with Quantum Espresso)
    运行文件: 08_verify_stability_dft.py
    技术: Density Functional Theory (DFT)。
    作用: 自动生成 Quantum Espresso (pw.x) 的输入文件，计算候选材料的形成能 (Formation Energy)，从量子力学层面最终确认材料的热力学稳定性。
    输出: 生成 .pw.in 输入文件及稳定性分析柱状图。