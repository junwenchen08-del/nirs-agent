"use client";

import { GitBranchIcon } from "lucide-react";
import Link from "next/link";

import { AuroraText } from "@/components/ui/aurora-text";
import { Button } from "@/components/ui/button";
import { BRAND } from "@/core/branding";

import { Section } from "../section";

export function CommunitySection() {
  return (
    <Section
      title={
        <AuroraText colors={["#FBBF24", "#60A5FA", "#A78BFA"]}>
          共建近红外智能分析平台
        </AuroraText>
      }
      subtitle="查看源码、提交问题或参与 ChemAgent 的能力建设"
    >
      <div className="flex justify-center">
        <Button className="text-lg" size="lg" asChild>
          <Link
            href={BRAND.repositoryUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            <GitBranchIcon />
            前往 ChemAgent 项目
          </Link>
        </Button>
      </div>
    </Section>
  );
}
