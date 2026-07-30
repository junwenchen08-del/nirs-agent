import {
  ActivityIcon,
  ChartNoAxesCombinedIcon,
  DatabaseIcon,
  FlaskConicalIcon,
  ScanSearchIcon,
  ShieldCheckIcon,
} from "lucide-react";
import Link from "next/link";

import { Card } from "@/components/ui/card";

import { Section } from "../section";

const APPLICATIONS = [
  {
    icon: DatabaseIcon,
    title: "多格式光谱数据解析",
    description:
      "自动识别 CSV、TXT 与 MATLAB 数据结构，完成波长轴、样本、目标值和官方分区审计。",
  },
  {
    icon: ScanSearchIcon,
    title: "智能预处理与波长筛选",
    description:
      "组合 SNV、MSC、SG、导数等方法，并在训练集内评估 CARS 等波长选择策略。",
  },
  {
    icon: FlaskConicalIcon,
    title: "化学计量学自主建模",
    description:
      "以 PLS 为基线，按数据规模和调优证据比较 Ridge、SVR、Extra Trees 等模型。",
  },
  {
    icon: ChartNoAxesCombinedIcon,
    title: "单成分与多成分分析",
    description:
      "支持单目标定量、多目标共享划分及独立评估，自动生成指标、图表和分析报告。",
  },
  {
    icon: ShieldCheckIcon,
    title: "科学质量门禁",
    description:
      "检查数据泄漏、异常波长轴、重复样本、显著偏差与模型适用域，阻止不可靠模型注册。",
  },
  {
    icon: ActivityIcon,
    title: "预测审计与漂移监测",
    description:
      "验证模型完整性，对未知样品批量预测，并持续追踪 T²/Q 漂移与告警恢复状态。",
  },
];

export function CaseStudySection({ className }: { className?: string }) {
  return (
    <Section
      className={className}
      title="近红外分析全流程"
      subtitle="从原始光谱到可部署模型，每一步都有确定性工具与科学证据支撑"
    >
      <div className="container-md mt-10 grid grid-cols-1 gap-4 px-4 md:grid-cols-2 md:px-12 lg:grid-cols-3">
        {APPLICATIONS.map(({ icon: Icon, title, description }) => (
          <Link key={title} href="/workspace">
            <Card className="group relative h-full min-h-56 overflow-hidden border-white/10 bg-linear-to-br from-white/8 to-white/2 p-6 transition-all duration-300 hover:-translate-y-1 hover:border-amber-300/30 hover:shadow-[0_20px_60px_-30px_rgba(251,191,36,0.5)]">
              <div className="mb-6 flex size-12 items-center justify-center rounded-xl border border-amber-300/20 bg-amber-300/10 text-amber-300 transition-transform group-hover:scale-110">
                <Icon className="size-6" />
              </div>
              <h3 className="text-xl font-semibold text-white">{title}</h3>
              <p className="mt-3 text-sm leading-6 text-zinc-400">
                {description}
              </p>
            </Card>
          </Link>
        ))}
      </div>
    </Section>
  );
}
