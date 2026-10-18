import { DatabaseIcon, GitBranchIcon, LineChartIcon } from "lucide-react";

import { Footer } from "@/components/landing/footer";
import { Header } from "@/components/landing/header";
import { Hero } from "@/components/landing/hero";
import { getI18n } from "@/core/i18n/server";

export default async function LandingPage() {
  const { locale } = await getI18n();
  const zh = locale === "zh-CN";
  const items = [
    {
      icon: DatabaseIcon,
      title: zh ? "看清输入数据" : "Understand your data",
      body: zh
        ? "上传 CSV、TXT 或 MAT，在数据工作台检查光谱、目标、样本用途与保留区间。"
        : "Upload CSV, TXT or MAT files and inspect spectra, targets, sample roles and retained intervals.",
    },
    {
      icon: GitBranchIcon,
      title: zh ? "理解每轮建模" : "Follow every experiment",
      body: zh
        ? "沿流程节点查看实际方法、参数与评价，分清下一步建议和已经执行的实验。"
        : "Review methods, parameters and evaluation at each node. Distinguish original proposals from executed experiments.",
    },
    {
      icon: LineChartIcon,
      title: zh ? "让结果便于复用" : "Reuse recorded results",
      body: zh
        ? "查看已记录的预测图表与报告，授权保存模型后，在新对话中应用到新的数据。"
        : "Inspect recorded predictions and reports. Approve saving a model and apply it to new data in a new chat.",
    },
  ];
  return (
    <div className="bg-background text-foreground min-h-screen w-full">
      <Header homeURL="/" />
      <main>
        <Hero />
        <section className="mx-auto max-w-6xl px-6 py-12">
          <h2 className="text-2xl font-semibold">
            {zh ? "围绕一次完整的光谱分析" : "One connected analysis workflow"}
          </h2>
          <div className="mt-8 grid gap-6 md:grid-cols-3">
            {items.map(({ icon: Icon, title, body }) => (
              <article key={title} className="border-t pt-6">
                <Icon className="text-primary mb-4 size-6" />
                <h3 className="text-lg font-medium">{title}</h3>
                <p className="text-muted-foreground mt-3 text-sm leading-relaxed">
                  {body}
                </p>
              </article>
            ))}
          </div>
          <p className="text-muted-foreground mt-10 max-w-3xl text-sm leading-relaxed">
            {zh
              ? "评价范围与数据来源随结果保留。历史没有记录的图表会明确说明；内部测试结果与未来的外部验证建议分别展示。"
              : "Evaluation scopes and data sources stay with each result. Missing historical plots remain explicit; internal tests and future external validation are presented separately."}
          </p>
        </section>
      </main>
      <Footer />
    </div>
  );
}
