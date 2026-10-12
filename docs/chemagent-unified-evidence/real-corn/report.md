# 近红外光谱建模报告

## 1. 结论摘要
- 分析目标：Oil；参考值单位：unknown (not stated in source)。
- 最终方案：pls；预处理：SNV → 二阶导数(window=11,order=2,delta=1.0,spacing_mode=index)；成分数：15。
- 最终留出集：R²=0.8997，RMSEP=0.0479，RPD=3.2615。误差单位与参考值一致。
- 验证范围：内部独立留出，不是外部验证；质量门禁通过（good）。
- 质量达标不等同于生产批准；本报告不代表已完成审批、注册或实际部署。

## 2. 数据概览与验证协议
- 样本数：48；原始光谱变量数：700。
- 协议：sealed_single_target_holdout。
- 光谱轴范围：1100.0000–2498.0000（单位未记录）。
- 参考值范围：3.0880–3.8220（unknown (not stated in source)）。

| 分区 | 样本数 | 用途 |
|---|---:|---|
| 校准集 | 48 | 拟合预处理、变量筛选和候选模型 |
| 调参集 | 16 | 选择预处理、变量和模型方案 |
| 最终留出集 | 16 | 选择完成后的最终评估 |

- 最终拟合范围：校准集。训练拟合指标不能作为独立验证结果。
- 模型选择使用调参/CV证据；最终留出/外部验证指标不用于选择候选。

## 3. 预处理、波长与模型选择依据
- 选定流水线：SNV → 二阶导数(window=11,order=2,delta=1.0,spacing_mode=index)。
- 波长选择：未记录；保留 未记录 / 未记录 个变量。
- 未记录模型族比较；不能由当前方案推断其他方法已经执行或较差。

## 4. 分区指标与质量评估

| 评估分区 | R² | RMSE | RPD | 偏差（预测−参考） |
|---|---:|---:|---:|---:|
| Cal（校准拟合） | 0.9998 | 0.0028 | 65.6830 | 0.0000 |
| Cal CV（校准交叉验证 OOF） | 0.9045 | 0.0552 | 3.2708 | 0.0011 |
| Val（选择用验证集） | 0.9111 | 0.0565 | 3.4639 | 0.0150 |
| Test（冻结后唯一最终留出） | 0.8997 | 0.0479 | 3.2615 | 0.0085 |

- R²描述拟合解释能力；RMSE描述误差大小；RPD受参考值分布影响。三者应结合样本覆盖与偏差共同判断。
- 表中指标读取实际计算结果；未记录字段不补算。RMSEP表示最终评估分区的RMSE，不能与调参误差混称。
- 领域：food；质量等级：good；门禁通过：是。
- 使用阈值：R²≥0.8，RPD≥3.0；其他门禁约束见 metrics.json 的 quality 字段。

## 5. 可视化与诊断解读

### 调参集残差诊断（与上方最终评估图分区不同）

| 诊断字段 | 实际结果 |
|---|---|
| residual_trend | flat |
| residual_variance | low |
| outlier_count | 1 |

## 6. 适用范围与后续建议
- 预测限于与训练数据一致的仪器、光谱轴、预处理和目标单位；超出参考值覆盖范围时应单独验证。
- 使用前核对样本代表性、批次/仪器差异、偏差和误差是否满足业务要求；内部留出达标仍需针对实际使用场景复核。
- 上线需独立完成用户批准与模型注册；使用时保留漂移监测和定期参考样本复测。

## 7. 复现与交付说明
- 数据SHA-256：b461d7644db2fb16bf0dce25896b6a4ba4e4c73c40aab9f681287dbe10ede19d。
- 随机种子：42；协议：sealed_single_target_holdout。
- report.html：可离线打开的完整图文报告；delivery.zip：报告、模型、指标、诊断图和文件校验清单。
- 模型及原有安全清单保持原样；metrics.json保留完整结果与复现参数。下载包不包含原始上传数据。
- 未记录的单位、样本数或诊断保持明确缺失；请勿将缺失字段解释为零或已通过。


## 8. LLM 自主决策与终结

- 停止来源：LLM 主动停止；资源原因：未记录。
- 最终保留实验：exp_31cfe4f52b6810aa76cb966b1fc0964b；终结决策：decision_82b27f44f48e804e27e852ff9c718d90。
- 提出的候选不代表已训练；只有有成功／失败执行记录的决策才进入实际执行统计。
- 模型冻结直接使用 Cal 工件，没有 Cal+Val 重拟合；最终 Test 不再进入搜索或后续 DecisionContext。

| 决策 | LLM 选择动作 | 实际执行状态 | fits | Val RMSE | 原预期检验 |
|---|---|---|---:|---:|---|
| decision_477900f915efd4aac3163357c38ffb82 | EXPLORE | succeeded | 51 | 0.0694 | not_evaluable |
| decision_142c8c17dc25ce2405c2043949d099b8 | CHANGE_PREPROCESS | succeeded | 76 | 0.0565 | achieved |
| decision_d6a92b8f361381caedae67917496f587 | TUNE | succeeded | 26 | 0.0571 | not_achieved |
| decision_82b27f44f48e804e27e852ff9c718d90 | STOP | stopped | 0 | 未记录 | 未记录 |

### decision_477900f915efd4aac3163357c38ffb82

- 原始选择理由：The Cal profile is dominated by baseline drift while noise and spikes are negligible, so derivative-based baseline filtering is the most defensible first correction; SNV is added as a complementary scatter/scale correction that is standard for powdered corn spectra. PLS is chosen over tree/kernel families because with only 48 calibration rows and 700 strongly collinear wavelengths a low-variance linear latent model is the safest, and it is preferred over Ridge because latent compression handles collinearity while still allowing a searchable complexity knob. Snv + derivative1(w=11, order=2) with scale=False and a 1-10 component grid is the single executed arm; the other three alternatives are recorded comparisons only.。
- 原始预期：The selected arm should yield Cal 5-fold CV RMSE clearly below the target's Cal standard deviation (0.1788) with a small selected component count, and a comparable Val RMSE. Metrics are expected in the unnamed source target units; no improvement is asserted until observed.。
- 提出候选：alt_pls_snv_d1:EXPLORE, alt_pls_raw:EXPLORE, alt_pls_d1:EXPLORE, alt_ridge_snv_d1:EXPLORE；选中 alt_pls_snv_d1。
- 未选理由：{"alt_pls_d1": "Comparison only: derivative alone may miss residual multiplicative scatter, but it is the cheapest way to isolate the SNV contribution later.", "alt_pls_raw": "Kept as a comparison only: skipping all preprocessing contradicts the tagged baseline drift and risks absorbing drift into PLS components.", "alt_ridge_snv_d1": "Comparison only: Ridge may need many more effective variables and lacks the latent complexity control; deferred unless Cal CV shows PLS is unstable."}。

### decision_142c8c17dc25ce2405c2043949d099b8

- 原始选择理由：The round-1 diagnosis is baseline drift with negligible noise, and the first SG derivative already handled most drift, so the highest-value legal change is to strengthen baseline/curvature filtering while keeping the PLS family that suits 48 samples and 700 collinear wavelengths. A second-order SG derivative after SNV removes residual additive baseline and linear slope and sharpens overlapping oil bands, and because high-frequency noise is 0.00019 the usual noise-amplification penalty of second derivatives should be small. I widen the component grid to 1-15 since the round-1 optimum sat at the edge, which may indicate the search was capped. Ridge (unchanged inputs), parameter-only tuning (flat CV curve), and AsLS+first-derivative (mechanism overlap) are recorded comparisons only and are no。
- 原始预期：The executed SNV + second-derivative PLS arm should select a component count within the grid and produce Cal 5-fold CV RMSE at or below the round-1 0.0789, with Val RMSE at or below 0.0694 and Val R2 at or above 0.866; whether Val RPD reaches 3.0 is uncertain. All metrics remain in the analyte's unnamed source units and no improvement is claimed until observed.。
- 提出候选：chg_snv_d2:CHANGE_PREPROCESS, sw_ridge_d1:SWITCH_MODEL, tune_pls_scale:TUNE, exp_asls_d1:EXPLORE；选中 chg_snv_d2。
- 未选理由：{"exp_asls_d1": "AsLS baseline correction plus a first derivative largely duplicates the baseline removal the first SG derivative already performs, so it risks being an approximate replication rather than a distinct mechanism; the second derivative instead adds new curvature information.", "sw_ridge_d1": "Ridge keeps all 700 collinear derivative variables and its solution is a smoother of the same linear signal; with n=48 the round-1 latent PLS already captured most linear structure, so the more informative next test is the spectral representation (curvature) rather than another linear estimator on unchanged inputs.", "tune_pls_scale": "Parameter-only tuning around the same SNV + first-derivative representation is expected to yield marginal movement since the round-1 CV curve was flat acro。

### decision_d6a92b8f361381caedae67917496f587

- 原始选择理由：The only unresolved structural question is the edge-limited component optimum: round-2 Cal CV RMSE decreased to n_components=15, which was the grid cap, so the PLS rank search may have been truncated. PLS is retained because with 48 calibration rows and 700 strongly collinear wavelengths a low-variance latent-variable linear model is the safest and it already exceeds the R2/RPD gates. This is a pure TUNE: same family, same SNV + second-derivative preprocessing, only the discrete n_components region moves to 16-20, costing 26 of the 33 remaining fits. SVR introduces kernel hyper-parameters unsupported by 48 rows and no non-linearity evidence exists, and MSC duplicates SNV's scatter-correction role while scatter variation is only 0.064. If the new optimum is interior it is the retained candi。
- 原始预期：The extended component region should locate the Cal-CV minimum interior to or at 20. Expected Cal CV RMSE at or below the round-2 value 0.05295 with Val RMSE near 0.0565 and Val R2 near 0.911. If the minimum is interior (n_components<20) that configuration becomes the preferred retained candidate; if CV RMSE rises monotonically beyond 15 the round-2 model stays best. All metrics remain in the analyte's unnamed source units and no improvement is asserted until observed.。
- 提出候选：tune_ncomp_16_20:TUNE, switch_svr_rbf:SWITCH_MODEL, change_msc_d2:CHANGE_PREPROCESS；选中 tune_ncomp_16_20。
- 未选理由：{"change_msc_d2": "MSC addresses the same additive and multiplicative scatter mechanism as SNV, which is already selected, and the Cal scatter_variation_score is low (0.064); swapping it in is expected to be near-neutral and adds no new information about the unresolved component-optimum question.", "switch_svr_rbf": "No residual non-linearity has been evidenced in either executed round, and an RBF SVR adds kernel hyper-parameters (C, gamma, epsilon) that are hard to identify reliably from only 48 calibration rows, so it is a lower-probability source of gain than completing the component probe."}。

### decision_82b27f44f48e804e27e852ff9c718d90

- 原始选择理由：Both structural axes tested have converged: the second derivative gave a large mechanistically explained gain (Val RMSE 0.0694 to 0.0565), and extending PLS rank to 16-20 gave a flat Cal-CV plateau with slightly worse Val, so the rank optimum is interior to a plateau. The remaining 7 fits allow only one fixed single point (6 fits), which cannot fairly test a new preprocessing or family, because a point fixed at n_components=15 (a rank tuned for SNV+d2) would confound the changed axis with rank and could not be followed by a component search. The two probes also have low prior: scatter variation is only 0.064 so MSC duplicates SNV, and there is no non-linearity evidence while 48 rows cannot support SVR tuning. With convergence demonstrated and no affordable well-powered hypothesis test, ret。
- 原始预期：Stop search and retain exp_31cfe4f52b6810aa76cb966b1fc0964b (SNV + SG derivative2 w=11 order=2; PLS n_components=15, scale=false, max_iter=500, tol=1e-06). No further fits. Under automatic delivery the server freezes it, evaluates the sealed Test once as an internal holdout (not external/production validation) and publishes model, metrics and report. Metrics are in the unnamed source units.。
- 提出候选：stop_retain_round2:STOP, msc_scatter_probe:CHANGE_PREPROCESS, svr_nonlinear_probe:SWITCH_MODEL；选中 stop_retain_round2。
- 未选理由：{"msc_scatter_probe": "Low scatter (0.064) makes MSC largely redundant with SNV, and with only ~7 fits a single fixed point at n_components=15 would confound the preprocessing change with latent rank, giving an uninterpretable result.", "svr_nonlinear_probe": "No non-linearity evidence; 48 calibration rows cannot support tuning C/gamma/epsilon, so one frozen triple would be arbitrary and its outcome uninterpretable."}。
- 保留依据、覆盖及未验证方向：{"retained_advantages": ["Best Val of the three successes: RMSE 0.056535, R2 0.91110, RPD 3.464, slope 0.920, MAE 0.043199.", "More parsimonious (15 components) than round-3 (20) with equal-or-better Val; Cal-CV RMSE 0.052952 indistinguishable from 0.052859.", "Preprocessing mechanism matches the Cal diagnosis: baseline drift removed with negligible noise penalty; stable calibration pass (slope 0.9998, ~zero bias)."], "retained_limitations": ["Validation is a 16-sample internal holdout, not external or production validation.", "Residual Val bias 0.01499 in unnamed units and slope 0.920 indicate slight holdout shrinkage.", "Target unit unnamed; preprocessing coverage limited to scatter-plus-derivative pipelines."], "search_coverage": "Three successful PLS experiments over two axes: derivati。

## 9. 实际模型配置与资源

- 冻结模型 SHA-256：d63ca67b78cca9979928c5c665fef43cbb25e82a0ff39ade8a0254353b631823。
- 冻结 CV manifest：cv_f8d479ce3cdfb72c8c58f3408a8a07503dde244c240842d80a50b3c6793f1c19。
- 实际参数：{"copy": true, "max_iter": 500, "n_components": 15, "scale": false, "tol": 1e-06}。
- 有序预处理：[{"method": "snv", "params": {}}, {"method": "derivative2", "params": {"delta": 1.0, "order": 2, "spacing_mode": "index", "window": 11}}]。
- 参数优化策略：grid；完整授权空间见 metrics.json 和 decisions.json。
- 可通过现有 nir_predict 加载带完整性 manifest 的 model.joblib；预处理状态已随工件保存。
- 配置、运行时 adapter 版本、原始候选、实际参数和完整决策证据见 decisions.json／metrics.json。

| 资源字段 | 实际记录 |
|---|---|
| scope | workflow intake through finalization tool; excludes subsequent narration/approval |
| intake_model_calls_included | 2 |
| experiments_started | 3 |
| experiments_succeeded | 3 |
| experiments_failed | 0 |
| completed_parameter_candidates | 30 |
| failed_candidate_count | unavailable for interrupted numerical fits |
| actual_model_fits | 153 |
| llm_calls | 20 |
| actual_input_tokens | 728024 |
| actual_output_tokens | 42258 |
| calls_with_unknown_usage | 0 |
| admission_tokens_estimated_or_actual_max | 2921020 |
| scientific_execution_seconds | 1.4380000000819564 |
| total_elapsed_seconds | 192.9082591533661 |
| llm_seconds | 187.72100000013597 |
| monetary_cost | not supplied by provider |
| all_caller_provider_usage | {'lead_agent_tokens': 770282, 'llm_call_count': 20, 'middleware_tokens': 0, 'run_id': 'a24f38c7-39a2-4a2b-b227-d938de54ae50', 'scope': 'current_run_until_deterministic_publication', 'source': 'trusted_run_journal', 'subagent_tokens': 0, 'token_usage_by_model': {'deepseek-flash': {'input_tokens': 728024, 'output_tokens': 42258, 'total_tokens': 770282}}, 'total_input_tokens': 728024, 'total_output_tokens': 42258, 'total_tokens': 770282, 'usage_kind': 'actual_provider_usage'} |
| token_limit_enabled | False |

- input/output token 字段仅统计 provider 实际 usage；未知 usage 单独计数。admission 字段含 estimated 额度，不是准确计费。未实现金额硬上限。
- 交付时搜索已停止、模型已冻结、Test 已完成；质量门禁独立记录。用户尚未批准，模型尚未注册。后续审批状态见工作流／注册表，不回写本次交付证据。