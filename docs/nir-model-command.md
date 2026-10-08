# `/models` 显式选择预测模型

普通网页对话启用 NIR 资产库后，输入 `/models` 打开模型版本选择窗口。
支持按模型名称、版本和算法搜索，展示验证范围和可用状态，已归档版本不可挂载。

点击“挂载到会话”调用现有 owner 校验、文件哈希校验的模型挂载接口；新对话会先准备线程，
不会创建运行或发送聊天消息。输入框显示精确模型版本，支持替换和取消选择。
已添加的预测数据文件会保留；也可通过 `/datasets` 挂载已保存的数据。
手动发送预测要求时才将精确模型标识、版本及实际返回路径加入用户消息；
智能体按现有 prediction 工作流审查新数据并复用训练时保存的预处理及波长选择。
挂载信息不代表预测已执行，兼容性与漂移检查仍由现有后端执行。

关闭窗口或切换对话会取消未完成的挂载；失败不会产生已选标签，可直接重试。
清除选择不删除线程副本。发送失败保留选择；发送成功清除本次标签。
本次仅增加桌面网页操作，不修改后端注册、审批或预测门禁。

回归入口：`frontend/tests/e2e/model-command.spec.ts`、
`frontend/tests/unit/core/nir-library/mounted-model.test.ts`、
`frontend/tests/unit/core/nir-library/api.test.ts`、
`frontend/tests/unit/components/workspace/input-box-helpers.test.ts`。

2026-10-06 验证：529 项前端单元测试、14 项模型/数据集命令浏览器回归通过；
TypeScript、相关 ESLint、Prettier、生产构建和 `git diff --check` 通过。
浏览器测试使用模拟 API 验证操作和发送内容，未对用户真实数据运行新预测。
